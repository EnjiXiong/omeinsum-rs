use std::collections::BTreeSet;

use super::{PlanError, StaticPlan, ValueId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveRange {
    pub value: ValueId,
    pub start: usize,
    pub end: usize,
    pub bytes: u64,
    pub alignment: u64,
}

impl LiveRange {
    pub fn new(value: ValueId, start: usize, end: usize, bytes: u64, alignment: u64) -> Self {
        Self {
            value,
            start,
            end,
            bytes,
            alignment,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaSlot {
    pub value: ValueId,
    pub offset: u64,
    pub bytes: u64,
}

impl ArenaSlot {
    pub fn new(value: ValueId, offset: u64, bytes: u64) -> Self {
        Self {
            value,
            offset,
            bytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaPlan {
    pub slots: Vec<ArenaSlot>,
    pub arena_bytes: u64,
    pub semantic_peak_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
struct ActiveAllocation {
    end: usize,
    offset: u64,
    extent: u64,
}

#[derive(Debug, Clone, Copy)]
struct FreeSpan {
    offset: u64,
    bytes: u64,
}

pub fn plan_f32_arena(plan: &StaticPlan) -> Result<ArenaPlan, PlanError> {
    plan.validate()?;
    let leaf_count = plan.leaf_values.len();
    let final_node = plan.nodes.len().saturating_sub(1);
    let mut starts = vec![0usize; plan.values.len()];
    let mut ends = vec![0usize; plan.values.len()];
    for (node_index, node) in plan.nodes.iter().enumerate() {
        starts[node.output.0] = node_index;
        ends[node.output.0] = node_index;
        ends[node.left.0] = ends[node.left.0].max(node_index);
        ends[node.right.0] = ends[node.right.0].max(node_index);
    }
    // Prepared executables enqueue the same fixed-address operator sequence
    // repeatedly without another H2D copy. Keep uploaded leaves alive for the
    // whole sequence so no computed value can overwrite the next enqueue's
    // immutable inputs.
    for leaf in &plan.leaf_values {
        ends[leaf.0] = final_node;
    }
    if plan.output.0 >= leaf_count {
        ends[plan.output.0] = ends[plan.output.0].max(final_node);
    } else {
        ends[plan.output.0] = final_node;
    }

    let ranges = plan
        .values
        .iter()
        .map(|value| {
            let elements = value
                .tensor
                .shape
                .iter()
                .try_fold(1u64, |product, dimension| {
                    let dimension = u64::try_from(*dimension).ok()?;
                    product.checked_mul(dimension)
                })
                .ok_or_else(|| PlanError::InvalidTree {
                    detail: format!("value {} element count overflows u64", value.id.0),
                })?;
            let planes = u64::try_from(value.planes.len()).map_err(|_| PlanError::InvalidTree {
                detail: format!("value {} plane count overflows u64", value.id.0),
            })?;
            let bytes = elements
                .checked_mul(planes)
                .and_then(|elements| elements.checked_mul(4))
                .ok_or_else(|| PlanError::InvalidTree {
                    detail: format!("value {} Float32 bytes overflow u64", value.id.0),
                })?;
            Ok(LiveRange::new(
                value.id,
                starts[value.id.0],
                ends[value.id.0],
                bytes,
                256,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    allocate_live_ranges(&ranges).map_err(|detail| PlanError::InvalidTree { detail })
}

pub fn allocate_live_ranges(ranges: &[LiveRange]) -> Result<ArenaPlan, String> {
    validate_ranges(ranges)?;
    let semantic_peak_bytes = semantic_peak(ranges)?;
    let mut order = (0..ranges.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|index| {
        let range = &ranges[*index];
        (range.start, range.value.0)
    });

    let mut active = Vec::<ActiveAllocation>::new();
    let mut free = Vec::<FreeSpan>::new();
    let mut arena_bytes = 0u64;
    let mut slots = Vec::with_capacity(ranges.len());
    for index in order {
        let range = &ranges[index];
        let mut retained = Vec::with_capacity(active.len());
        for allocation in active.drain(..) {
            if allocation.end < range.start {
                insert_free_span(
                    &mut free,
                    FreeSpan {
                        offset: allocation.offset,
                        bytes: allocation.extent,
                    },
                )?;
            } else {
                retained.push(allocation);
            }
        }
        active = retained;

        let extent = align_up(range.bytes, range.alignment)?;
        let offset = match take_first_fit(&mut free, extent, range.alignment)? {
            Some(offset) => offset,
            None => {
                let offset = align_up(arena_bytes, range.alignment)?;
                if offset > arena_bytes {
                    insert_free_span(
                        &mut free,
                        FreeSpan {
                            offset: arena_bytes,
                            bytes: offset - arena_bytes,
                        },
                    )?;
                }
                arena_bytes = offset
                    .checked_add(extent)
                    .ok_or_else(|| "device arena size overflows u64".to_string())?;
                offset
            }
        };
        active.push(ActiveAllocation {
            end: range.end,
            offset,
            extent,
        });
        slots.push(ArenaSlot::new(range.value, offset, range.bytes));
    }
    slots.sort_unstable_by_key(|slot| slot.value.0);
    Ok(ArenaPlan {
        slots,
        arena_bytes,
        semantic_peak_bytes,
    })
}

fn validate_ranges(ranges: &[LiveRange]) -> Result<(), String> {
    let mut values = BTreeSet::new();
    for range in ranges {
        if !values.insert(range.value) {
            return Err(format!("duplicate live range for value {}", range.value.0));
        }
        if range.start > range.end {
            return Err(format!(
                "value {} starts after it ends: {} > {}",
                range.value.0, range.start, range.end
            ));
        }
        if range.bytes == 0 {
            return Err(format!("value {} has zero bytes", range.value.0));
        }
        if !range.alignment.is_power_of_two() {
            return Err(format!(
                "value {} alignment {} is not a power of two",
                range.value.0, range.alignment
            ));
        }
    }
    Ok(())
}

fn semantic_peak(ranges: &[LiveRange]) -> Result<u64, String> {
    let mut points = ranges
        .iter()
        .flat_map(|range| [range.start, range.end])
        .collect::<Vec<_>>();
    points.sort_unstable();
    points.dedup();
    points.into_iter().try_fold(0u64, |peak, point| {
        let live = ranges
            .iter()
            .filter(|range| range.start <= point && point <= range.end)
            .try_fold(0u64, |sum, range| {
                sum.checked_add(range.bytes)
                    .ok_or_else(|| "semantic live-byte sum overflows u64".to_string())
            })?;
        Ok(peak.max(live))
    })
}

fn take_first_fit(
    free: &mut Vec<FreeSpan>,
    bytes: u64,
    alignment: u64,
) -> Result<Option<u64>, String> {
    free.sort_unstable_by_key(|span| span.offset);
    for index in 0..free.len() {
        let span = free[index];
        let offset = align_up(span.offset, alignment)?;
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| "free-span allocation overflows u64".to_string())?;
        let span_end = span
            .offset
            .checked_add(span.bytes)
            .ok_or_else(|| "free span overflows u64".to_string())?;
        if end > span_end {
            continue;
        }
        free.remove(index);
        if offset > span.offset {
            insert_free_span(
                free,
                FreeSpan {
                    offset: span.offset,
                    bytes: offset - span.offset,
                },
            )?;
        }
        if end < span_end {
            insert_free_span(
                free,
                FreeSpan {
                    offset: end,
                    bytes: span_end - end,
                },
            )?;
        }
        return Ok(Some(offset));
    }
    Ok(None)
}

fn insert_free_span(free: &mut Vec<FreeSpan>, span: FreeSpan) -> Result<(), String> {
    if span.bytes == 0 {
        return Ok(());
    }
    free.push(span);
    free.sort_unstable_by_key(|span| span.offset);
    let mut merged = Vec::<FreeSpan>::with_capacity(free.len());
    for span in free.drain(..) {
        if let Some(previous) = merged.last_mut() {
            let previous_end = previous
                .offset
                .checked_add(previous.bytes)
                .ok_or_else(|| "free span overflows u64".to_string())?;
            if span.offset < previous_end {
                return Err("free spans overlap".to_string());
            }
            if span.offset == previous_end {
                previous.bytes = previous
                    .bytes
                    .checked_add(span.bytes)
                    .ok_or_else(|| "coalesced free span overflows u64".to_string())?;
                continue;
            }
        }
        merged.push(span);
    }
    *free = merged;
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64, String> {
    if !alignment.is_power_of_two() {
        return Err(format!("alignment {alignment} is not a power of two"));
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| "alignment arithmetic overflows u64".to_string())
}
