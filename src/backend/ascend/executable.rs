#![allow(clippy::result_large_err)] // Frozen backend diagnostics retain full context.

use crate::static_plan::{
    coalesce_permutation, lower_plan_traces, plan_f32_arena, ComplexValue, ExecutionError,
    InputSet, KernelKind, PreparedExecutable, ScratchRole, StaticPlan, TensorSpec,
};

use super::capture::{configure_capture, Capture, CaptureAttempt, CaptureRuntime};
use super::operator::{PreparedOp, PreparedStep};
use super::storage::{DeviceBuffer, TensorDescriptor};
use super::{AscendMemoryStats, AscendPhaseTimings, AscendSession, CaptureStatus};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BufferKind {
    Semantic,
    Scratch,
}

#[derive(Debug, Clone)]
struct ValueLayout {
    offset: u64,
    elements: usize,
    planes: usize,
    physical_modes: Vec<i32>,
}

#[derive(Debug, Clone, Copy)]
struct LocalSlot {
    offset: u64,
    elements: usize,
}

#[derive(Debug, Clone)]
struct NodeResources {
    left_desired: Vec<i32>,
    right_desired: Vec<i32>,
    roles: Vec<(ScratchRole, LocalSlot)>,
    left_pack: Option<LocalSlot>,
    right_pack: Option<LocalSlot>,
    combine_temp: Option<LocalSlot>,
    bytes: u64,
}

#[derive(Debug, Clone, Copy)]
struct OperandSource {
    buffer: BufferKind,
    offset: u64,
    elements: usize,
    planes: usize,
}

pub(crate) struct ExecutableState<'session> {
    steps: Vec<PreparedStep<'session>>,
    workspace: DeviceBuffer<'session>,
    _scratch: DeviceBuffer<'session>,
    semantic_arena: DeviceBuffer<'session>,
    output_offset: u64,
    output_planes: usize,
    output_elements: usize,
    context: &'session super::context::Context,
    pub(crate) phase_timings: AscendPhaseTimings,
    capture: Option<Capture<'session>>,
}

impl<'session> ExecutableState<'session> {
    pub(crate) fn prepare(
        session: &'session AscendSession,
        plan: &StaticPlan,
        inputs: &InputSet<f64>,
    ) -> Result<(Self, AscendMemoryStats), ExecutionError> {
        let lower_start = Instant::now();
        plan.validate()
            .map_err(|error| ExecutionError::InvalidPlan(error.to_string()))?;
        validate_inputs(plan, inputs)?;
        // This is both the pure lowering trace and the preparation-time audit
        // against the frozen plan accounting.
        lower_plan_traces(plan)?;
        let arena =
            plan_f32_arena(plan).map_err(|error| ExecutionError::InvalidPlan(error.to_string()))?;
        let value_layouts = build_value_layouts(plan, &arena.slots)?;
        let node_resources = build_node_resources(plan, &value_layouts)?;
        let scratch_bytes = node_resources
            .iter()
            .map(|resources| resources.bytes)
            .max()
            .unwrap_or(0);
        let plan_lower_seconds = lower_start.elapsed().as_secs_f64();

        let allocation_start = Instant::now();
        let semantic_arena = session.context.allocate(arena.arena_bytes)?;
        let scratch = session.context.allocate(scratch_bytes)?;
        let mut allocation_seconds = allocation_start.elapsed().as_secs_f64();
        let h2d_start = Instant::now();
        upload_inputs(&semantic_arena, plan, inputs, &value_layouts)?;
        session.context.synchronize()?;
        let h2d_seconds = h2d_start.elapsed().as_secs_f64();

        let descriptor_start = Instant::now();
        let mut steps = Vec::new();
        for (node, resources) in plan.nodes.iter().zip(&node_resources) {
            let left = prepare_operand(
                &session.context,
                &semantic_arena,
                &scratch,
                &value_layouts[node.left.0],
                &plan.values[node.left.0].tensor,
                &resources.left_desired,
                resources.left_pack,
                node.id,
                &mut steps,
            )?;
            let right = prepare_operand(
                &session.context,
                &semantic_arena,
                &scratch,
                &value_layouts[node.right.0],
                &plan.values[node.right.0].tensor,
                &resources.right_desired,
                resources.right_pack,
                node.id,
                &mut steps,
            )?;
            lower_node(
                &session.context,
                &semantic_arena,
                &scratch,
                node,
                resources,
                left,
                right,
                &value_layouts[node.output.0],
                &mut steps,
            )?;
        }
        let workspace_bytes = steps
            .iter()
            .map(PreparedStep::workspace_bytes)
            .max()
            .unwrap_or(0);
        let descriptor_executor_prepare_seconds = descriptor_start.elapsed().as_secs_f64();
        let workspace_allocation_start = Instant::now();
        let workspace = session.context.allocate(workspace_bytes)?;
        allocation_seconds += workspace_allocation_start.elapsed().as_secs_f64();
        let peak_device_bytes = arena
            .arena_bytes
            .checked_add(scratch_bytes)
            .and_then(|bytes| bytes.checked_add(workspace_bytes))
            .ok_or_else(|| {
                ExecutionError::Unsupported(
                    "Ascend peak device-byte accounting overflowed u64".to_string(),
                )
            })?;
        let output = &value_layouts[plan.output.0];
        let stats = AscendMemoryStats {
            semantic_bytes: arena.semantic_peak_bytes,
            device_arena_bytes: arena.arena_bytes,
            scratch_bytes,
            workspace_bytes,
            peak_device_bytes,
        };
        Ok((
            Self {
                steps,
                workspace,
                _scratch: scratch,
                semantic_arena,
                output_offset: output.offset,
                output_planes: output.planes,
                output_elements: output.elements,
                context: &session.context,
                phase_timings: AscendPhaseTimings {
                    context_create_seconds: 0.0,
                    plan_lower_seconds,
                    allocation_seconds,
                    descriptor_executor_prepare_seconds,
                    h2d_seconds,
                    warmup_seconds: 0.0,
                    d2h_seconds: 0.0,
                },
                capture: None,
            },
            stats,
        ))
    }

    pub(crate) fn enqueue(&mut self) -> Result<(), ExecutionError> {
        if let Some(capture) = &mut self.capture {
            return capture.run();
        }
        self.enqueue_repeatable()
    }

    fn enqueue_repeatable(&mut self) -> Result<(), ExecutionError> {
        for step in &mut self.steps {
            step.run(&self.workspace)?;
        }
        Ok(())
    }

    pub(crate) fn configure_capture(
        &mut self,
        required: bool,
    ) -> Result<CaptureStatus, ExecutionError> {
        let mut runtime = NativeCaptureRuntime {
            context: self.context,
            steps: &mut self.steps,
            workspace: &self.workspace,
        };
        match configure_capture(&mut runtime, required)? {
            CaptureAttempt::Ready(capture) => {
                self.capture = Some(capture);
                Ok(CaptureStatus::Ready)
            }
            CaptureAttempt::SkippedUnsupported { reason } => {
                Ok(CaptureStatus::SkippedUnsupported { reason })
            }
        }
    }

    pub(crate) fn synchronize(&self) -> Result<(), ExecutionError> {
        self.context.synchronize()
    }

    pub(crate) fn output(&mut self) -> Result<ComplexValue, ExecutionError> {
        if self.output_elements != 1 {
            return Err(ExecutionError::InvalidPlan(format!(
                "Ascend output has {} elements; expected a scalar",
                self.output_elements
            )));
        }
        let d2h_start = Instant::now();
        let mut real = [0.0f32; 1];
        self.semantic_arena
            .copy_d2h(self.output_offset, &mut real)?;
        let mut imag = [0.0f32; 1];
        if self.output_planes == 2 {
            self.semantic_arena.copy_d2h(
                self.output_offset.checked_add(4).ok_or_else(|| {
                    ExecutionError::Unsupported("output plane offset overflow".to_string())
                })?,
                &mut imag,
            )?;
        }
        self.context.synchronize()?;
        self.phase_timings.d2h_seconds += d2h_start.elapsed().as_secs_f64();
        let value = ComplexValue {
            re: f64::from(real[0]),
            im: f64::from(imag[0]),
        };
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(ExecutionError::NonFiniteOutput(value));
        }
        Ok(value)
    }
}

impl Drop for ExecutableState<'_> {
    fn drop(&mut self) {
        let _ = self.context.synchronize();
        // Executors retain descriptor/scalar/array objects. Destroy them before
        // workspace, scratch, semantic storage, and finally the session.
        self.capture.take();
        self.steps.clear();
    }
}

struct NativeCaptureRuntime<'state, 'session> {
    context: &'session super::context::Context,
    steps: &'state mut [PreparedStep<'session>],
    workspace: &'state DeviceBuffer<'session>,
}

impl<'session> CaptureRuntime for NativeCaptureRuntime<'_, 'session> {
    type Handle = Capture<'session>;

    fn supported(&mut self) -> Result<bool, ExecutionError> {
        Capture::supported(self.context)
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        self.context.synchronize()
    }

    fn begin(&mut self) -> Result<(), ExecutionError> {
        Capture::begin(self.context)
    }

    fn enqueue_repeatable(&mut self) -> Result<(), ExecutionError> {
        for step in &mut *self.steps {
            step.run(self.workspace)?;
        }
        Ok(())
    }

    fn end(&mut self) -> Result<Self::Handle, ExecutionError> {
        Capture::end(self.context)
    }
}

fn validate_inputs(plan: &StaticPlan, inputs: &InputSet<f64>) -> Result<(), ExecutionError> {
    if plan.leaf_values.len() != inputs.tensors.len() {
        return Err(ExecutionError::InvalidPlan(format!(
            "plan has {} leaves but input set has {} tensors",
            plan.leaf_values.len(),
            inputs.tensors.len()
        )));
    }
    for (index, (value, input)) in plan
        .leaf_values
        .iter()
        .map(|value| &plan.values[value.0])
        .zip(&inputs.tensors)
        .enumerate()
    {
        if value.tensor != input.spec {
            return Err(ExecutionError::InvalidPlan(format!(
                "input tensor {index} does not match plan leaf geometry"
            )));
        }
        let expected = elements(&input.spec)?;
        if input.real.len() != expected || input.imag.len() != expected {
            return Err(ExecutionError::InvalidPlan(format!(
                "input tensor {index} plane length differs from its shape"
            )));
        }
    }
    Ok(())
}

fn build_value_layouts(
    plan: &StaticPlan,
    slots: &[crate::static_plan::ArenaSlot],
) -> Result<Vec<ValueLayout>, ExecutionError> {
    let mut layouts = Vec::with_capacity(plan.values.len());
    for (index, value) in plan.values.iter().enumerate() {
        let physical_modes: Vec<i32> = if index < plan.leaf_values.len() {
            value.tensor.modes.iter().rev().copied().collect()
        } else {
            let node = &plan.nodes[index - plan.leaf_values.len()];
            node.contraction
                .batch_modes
                .iter()
                .chain(&node.contraction.left_modes)
                .chain(&node.contraction.right_modes)
                .copied()
                .collect()
        };
        let mut expected = physical_modes.clone();
        let mut actual = value.tensor.modes.clone();
        expected.sort_unstable();
        actual.sort_unstable();
        if expected != actual {
            return Err(ExecutionError::InvalidPlan(format!(
                "value {index} physical mode set differs from TensorSpec"
            )));
        }
        let slot = slots.get(index).ok_or_else(|| {
            ExecutionError::InvalidPlan(format!("value {index} has no arena slot"))
        })?;
        layouts.push(ValueLayout {
            offset: slot.offset,
            elements: elements(&value.tensor)?,
            planes: value.planes.len(),
            physical_modes,
        });
    }
    Ok(layouts)
}

fn build_node_resources(
    plan: &StaticPlan,
    layouts: &[ValueLayout],
) -> Result<Vec<NodeResources>, ExecutionError> {
    plan.nodes
        .iter()
        .map(|node| {
            let mut cursor = 0u64;
            let mut roles = Vec::with_capacity(node.scratch.len());
            for spec in &node.scratch {
                let slot = allocate_local(&mut cursor, spec.elements)?;
                roles.push((spec.role.clone(), slot));
            }
            let combine_temp = (node.kind == KernelKind::Merge3M)
                .then(|| allocate_local(&mut cursor, layouts[node.output.0].elements))
                .transpose()?;
            let left_desired = node
                .contraction
                .batch_modes
                .iter()
                .chain(&node.contraction.left_modes)
                .chain(&node.contraction.contracted_modes)
                .copied()
                .collect::<Vec<_>>();
            let right_desired = node
                .contraction
                .batch_modes
                .iter()
                .chain(&node.contraction.contracted_modes)
                .chain(&node.contraction.right_modes)
                .copied()
                .collect::<Vec<_>>();
            let left_pack = if layouts[node.left.0].physical_modes != left_desired {
                validate_permutation_rank(
                    &layouts[node.left.0],
                    &plan.values[node.left.0].tensor,
                    &left_desired,
                )?;
                Some(allocate_local(
                    &mut cursor,
                    layouts[node.left.0]
                        .elements
                        .checked_mul(layouts[node.left.0].planes)
                        .ok_or_else(|| {
                            ExecutionError::Unsupported(
                                "left packing element count overflow".to_string(),
                            )
                        })?,
                )?)
            } else {
                None
            };
            let right_pack = if layouts[node.right.0].physical_modes != right_desired {
                validate_permutation_rank(
                    &layouts[node.right.0],
                    &plan.values[node.right.0].tensor,
                    &right_desired,
                )?;
                Some(allocate_local(
                    &mut cursor,
                    layouts[node.right.0]
                        .elements
                        .checked_mul(layouts[node.right.0].planes)
                        .ok_or_else(|| {
                            ExecutionError::Unsupported(
                                "right packing element count overflow".to_string(),
                            )
                        })?,
                )?)
            } else {
                None
            };
            Ok(NodeResources {
                left_desired,
                right_desired,
                roles,
                left_pack,
                right_pack,
                combine_temp,
                bytes: cursor,
            })
        })
        .collect()
}

fn validate_permutation_rank(
    layout: &ValueLayout,
    spec: &TensorSpec,
    desired: &[i32],
) -> Result<(), ExecutionError> {
    let plane_rank = usize::from(layout.planes == 2);
    let rank_limit = 8usize
        .checked_sub(plane_rank)
        .ok_or_else(|| ExecutionError::Unsupported("invalid live ACLNN rank limit".to_string()))?;
    coalesce_permutation(
        &layout.physical_modes,
        desired,
        &dimensions(spec),
        rank_limit,
    )?;
    Ok(())
}

fn allocate_local(cursor: &mut u64, elements: usize) -> Result<LocalSlot, ExecutionError> {
    *cursor = align_up(*cursor, 256)?;
    let offset = *cursor;
    let bytes = u64::try_from(elements)
        .ok()
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| {
            ExecutionError::Unsupported("scratch byte count overflows u64".to_string())
        })?;
    *cursor = cursor.checked_add(bytes).ok_or_else(|| {
        ExecutionError::Unsupported("scratch slab size overflows u64".to_string())
    })?;
    Ok(LocalSlot { offset, elements })
}

fn upload_inputs(
    arena: &DeviceBuffer<'_>,
    plan: &StaticPlan,
    inputs: &InputSet<f64>,
    layouts: &[ValueLayout],
) -> Result<(), ExecutionError> {
    for (index, input) in inputs.tensors.iter().enumerate() {
        let layout = &layouts[index];
        let real = to_f32(&input.real, index, "real")?;
        arena.copy_h2d(layout.offset, &real)?;
        if layout.planes == 2 {
            let imag = to_f32(&input.imag, index, "imag")?;
            arena.copy_h2d(plane_offset(layout, 1)?, &imag)?;
        }
        if plan.values[index].planes.len() != layout.planes {
            return Err(ExecutionError::InvalidPlan(format!(
                "leaf {index} plane layout changed during upload"
            )));
        }
    }
    Ok(())
}

fn to_f32(values: &[f64], tensor: usize, plane: &str) -> Result<Vec<f32>, ExecutionError> {
    values
        .iter()
        .enumerate()
        .map(|(position, value)| {
            let converted = *value as f32;
            converted.is_finite().then_some(converted).ok_or_else(|| {
                ExecutionError::Unsupported(format!(
                    "tensor {tensor} {plane}[{position}] is not finite in Float32"
                ))
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn prepare_operand<'session>(
    context: &'session super::context::Context,
    arena: &DeviceBuffer<'session>,
    scratch: &DeviceBuffer<'session>,
    layout: &ValueLayout,
    spec: &TensorSpec,
    desired: &[i32],
    pack: Option<LocalSlot>,
    node_id: usize,
    steps: &mut Vec<PreparedStep<'session>>,
) -> Result<OperandSource, ExecutionError> {
    let source = OperandSource {
        buffer: BufferKind::Semantic,
        offset: layout.offset,
        elements: layout.elements,
        planes: layout.planes,
    };
    let Some(pack) = pack else {
        return Ok(source);
    };
    let plane_rank = usize::from(layout.planes == 2);
    let permutation = coalesce_permutation(
        &layout.physical_modes,
        desired,
        &dimensions(spec),
        8 - plane_rank,
    )?;
    let (input_shape, output_shape, axes) = if layout.planes == 2 {
        let mut input_shape = vec![2];
        input_shape.extend(permutation.input_shape);
        let mut output_shape = vec![2];
        output_shape.extend(permutation.output_shape);
        let mut axes = vec![0];
        axes.extend(permutation.axes.into_iter().map(|axis| axis + 1));
        (input_shape, output_shape, axes)
    } else {
        (
            permutation.input_shape,
            permutation.output_shape,
            permutation.axes,
        )
    };
    let input = arena.tensor_f32(
        source.offset,
        &input_shape,
        &row_major_strides(&input_shape)?,
    )?;
    let output = scratch.tensor_f32(
        pack.offset,
        &output_shape,
        &row_major_strides(&output_shape)?,
    )?;
    steps.push(PreparedStep {
        op: PreparedOp::permute(context, &input, &axes, &output, Some(node_id))?,
    });
    Ok(OperandSource {
        buffer: BufferKind::Scratch,
        offset: pack.offset,
        elements: layout.elements,
        planes: layout.planes,
    })
}

#[allow(clippy::too_many_arguments)]
fn lower_node<'session>(
    context: &'session super::context::Context,
    arena: &DeviceBuffer<'session>,
    scratch: &DeviceBuffer<'session>,
    node: &crate::static_plan::PlanNode,
    resources: &NodeResources,
    left: OperandSource,
    right: OperandSource,
    output: &ValueLayout,
    steps: &mut Vec<PreparedStep<'session>>,
) -> Result<(), ExecutionError> {
    let batch = node.contraction.batch;
    let m = node.contraction.m;
    let k = node.contraction.k;
    let n = node.contraction.n;
    match node.kind {
        KernelKind::RealReal => {
            let left = source_matrix(arena, scratch, left, 0, &[batch, m, k])?;
            let right = source_matrix(arena, scratch, right, 0, &[batch, k, n])?;
            let output = arena.tensor_f32(
                output.offset,
                &[batch, m, n],
                &row_major_strides(&[batch, m, n])?,
            )?;
            push_matmul(context, left, right, output, node.id, steps)?;
        }
        KernelKind::RideLeft | KernelKind::RideRight => {
            let left_planes = if node.kind == KernelKind::RideLeft {
                2
            } else {
                1
            };
            let right_planes = if node.kind == KernelKind::RideRight {
                2
            } else {
                1
            };
            let left = source_batched_planes(arena, scratch, left, left_planes, &[batch, m, k])?;
            let right = source_batched_planes(arena, scratch, right, right_planes, &[batch, k, n])?;
            let shape = [2, batch, m, n];
            let output = arena.tensor_f32(output.offset, &shape, &row_major_strides(&shape)?)?;
            push_matmul(context, left, right, output, node.id, steps)?;
        }
        KernelKind::Merge3M => {
            lower_merge(
                context, arena, scratch, node, resources, left, right, output, steps,
            )?;
        }
        KernelKind::Flat4M => {
            lower_flat(
                context, arena, scratch, node, resources, left, right, output, steps,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn lower_merge<'session>(
    context: &'session super::context::Context,
    arena: &DeviceBuffer<'session>,
    scratch: &DeviceBuffer<'session>,
    node: &crate::static_plan::PlanNode,
    resources: &NodeResources,
    left: OperandSource,
    right: OperandSource,
    output: &ValueLayout,
    steps: &mut Vec<PreparedStep<'session>>,
) -> Result<(), ExecutionError> {
    let shape_left = [
        node.contraction.batch,
        node.contraction.m,
        node.contraction.k,
    ];
    let shape_right = [
        node.contraction.batch,
        node.contraction.k,
        node.contraction.n,
    ];
    let shape_output = [
        node.contraction.batch,
        node.contraction.m,
        node.contraction.n,
    ];
    let ar = source_matrix(arena, scratch, left, 0, &shape_left)?;
    let ai = source_matrix(arena, scratch, left, 1, &shape_left)?;
    let br = source_matrix(arena, scratch, right, 0, &shape_right)?;
    let bi = source_matrix(arena, scratch, right, 1, &shape_right)?;
    let ar_flat = source_flat(arena, scratch, left, 0)?;
    let ai_flat = source_flat(arena, scratch, left, 1)?;
    let br_flat = source_flat(arena, scratch, right, 0)?;
    let bi_flat = source_flat(arena, scratch, right, 1)?;
    let left_sum_slot = role(resources, ScratchRole::LeftSum)?;
    let right_sum_slot = role(resources, ScratchRole::RightSum)?;
    let p1_slot = role(resources, ScratchRole::Product1)?;
    let p2_slot = role(resources, ScratchRole::Product2)?;
    let p3_slot = role(resources, ScratchRole::Product3)?;
    let temp_slot = resources.combine_temp.ok_or_else(|| {
        ExecutionError::InvalidPlan(format!("merge node {} has no combine temp", node.id))
    })?;
    let left_sum_flat = flat_tensor(scratch, left_sum_slot)?;
    let right_sum_flat = flat_tensor(scratch, right_sum_slot)?;
    push_add(context, ar_flat, ai_flat, left_sum_flat, node.id, steps)?;
    push_add(context, br_flat, bi_flat, right_sum_flat, node.id, steps)?;
    let left_sum = shaped_tensor(scratch, left_sum_slot, &shape_left)?;
    let right_sum = shaped_tensor(scratch, right_sum_slot, &shape_right)?;
    let p1 = shaped_tensor(scratch, p1_slot, &shape_output)?;
    let p2 = shaped_tensor(scratch, p2_slot, &shape_output)?;
    let p3 = shaped_tensor(scratch, p3_slot, &shape_output)?;
    let p1_flat = flat_tensor(scratch, p1_slot)?;
    let p2_flat = flat_tensor(scratch, p2_slot)?;
    let p3_flat = flat_tensor(scratch, p3_slot)?;
    push_matmul(context, ar, br, p1.clone(), node.id, steps)?;
    push_matmul(context, ai, bi, p2.clone(), node.id, steps)?;
    push_matmul(context, left_sum, right_sum, p3.clone(), node.id, steps)?;
    let cr = output_plane_flat(arena, output, 0)?;
    let ci = output_plane_flat(arena, output, 1)?;
    let temp = flat_tensor(scratch, temp_slot)?;
    push_sub(
        context,
        p1_flat.clone(),
        p2_flat.clone(),
        cr,
        node.id,
        steps,
    )?;
    push_sub(context, p3_flat, p1_flat, temp.clone(), node.id, steps)?;
    push_sub(context, temp, p2_flat, ci, node.id, steps)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn lower_flat<'session>(
    context: &'session super::context::Context,
    arena: &DeviceBuffer<'session>,
    scratch: &DeviceBuffer<'session>,
    node: &crate::static_plan::PlanNode,
    resources: &NodeResources,
    left: OperandSource,
    right: OperandSource,
    output: &ValueLayout,
    steps: &mut Vec<PreparedStep<'session>>,
) -> Result<(), ExecutionError> {
    let shape_left = [
        node.contraction.batch,
        node.contraction.m,
        node.contraction.k,
    ];
    let shape_right = [
        node.contraction.batch,
        node.contraction.k,
        node.contraction.n,
    ];
    let shape_output = [
        node.contraction.batch,
        node.contraction.m,
        node.contraction.n,
    ];
    let ar = source_matrix(arena, scratch, left, 0, &shape_left)?;
    let ai = source_matrix(arena, scratch, left, 1, &shape_left)?;
    let br = source_matrix(arena, scratch, right, 0, &shape_right)?;
    let bi = source_matrix(arena, scratch, right, 1, &shape_right)?;
    let p1 = shaped_tensor(
        scratch,
        role(resources, ScratchRole::Product1)?,
        &shape_output,
    )?;
    let p2 = shaped_tensor(
        scratch,
        role(resources, ScratchRole::Product2)?,
        &shape_output,
    )?;
    let p3 = shaped_tensor(
        scratch,
        role(resources, ScratchRole::Product3)?,
        &shape_output,
    )?;
    let p4 = shaped_tensor(
        scratch,
        role(resources, ScratchRole::Product4)?,
        &shape_output,
    )?;
    let p1_flat = flat_tensor(scratch, role(resources, ScratchRole::Product1)?)?;
    let p2_flat = flat_tensor(scratch, role(resources, ScratchRole::Product2)?)?;
    let p3_flat = flat_tensor(scratch, role(resources, ScratchRole::Product3)?)?;
    let p4_flat = flat_tensor(scratch, role(resources, ScratchRole::Product4)?)?;
    push_matmul(context, ar.clone(), br.clone(), p1.clone(), node.id, steps)?;
    push_matmul(context, ai.clone(), bi.clone(), p2.clone(), node.id, steps)?;
    push_matmul(context, ar, bi, p3.clone(), node.id, steps)?;
    push_matmul(context, ai, br, p4.clone(), node.id, steps)?;
    let cr = output_plane_flat(arena, output, 0)?;
    let ci = output_plane_flat(arena, output, 1)?;
    push_sub(context, p1_flat, p2_flat, cr, node.id, steps)?;
    push_add(context, p3_flat, p4_flat, ci, node.id, steps)?;
    Ok(())
}

fn push_matmul<'session>(
    context: &'session super::context::Context,
    left: TensorDescriptor<'session>,
    right: TensorDescriptor<'session>,
    output: TensorDescriptor<'session>,
    node_id: usize,
    steps: &mut Vec<PreparedStep<'session>>,
) -> Result<(), ExecutionError> {
    steps.push(PreparedStep {
        op: PreparedOp::matmul(context, &left, &right, &output, Some(node_id))?,
    });
    Ok(())
}

fn push_add<'session>(
    context: &'session super::context::Context,
    left: TensorDescriptor<'session>,
    right: TensorDescriptor<'session>,
    output: TensorDescriptor<'session>,
    node_id: usize,
    steps: &mut Vec<PreparedStep<'session>>,
) -> Result<(), ExecutionError> {
    steps.push(PreparedStep {
        op: PreparedOp::add(context, &left, &right, &output, Some(node_id))?,
    });
    Ok(())
}

fn push_sub<'session>(
    context: &'session super::context::Context,
    left: TensorDescriptor<'session>,
    right: TensorDescriptor<'session>,
    output: TensorDescriptor<'session>,
    node_id: usize,
    steps: &mut Vec<PreparedStep<'session>>,
) -> Result<(), ExecutionError> {
    steps.push(PreparedStep {
        op: PreparedOp::sub(context, &left, &right, &output, Some(node_id))?,
    });
    Ok(())
}

fn source_matrix<'session>(
    arena: &DeviceBuffer<'session>,
    scratch: &DeviceBuffer<'session>,
    source: OperandSource,
    plane: usize,
    shape: &[usize],
) -> Result<TensorDescriptor<'session>, ExecutionError> {
    if plane >= source.planes {
        return Err(ExecutionError::InvalidPlan(format!(
            "requested plane {plane} from {}-plane operand",
            source.planes
        )));
    }
    let offset = source
        .offset
        .checked_add(plane_bytes(source.elements, plane)?)
        .ok_or_else(|| ExecutionError::Unsupported("operand offset overflow".to_string()))?;
    buffer(source.buffer, arena, scratch).tensor_f32(offset, shape, &row_major_strides(shape)?)
}

fn source_batched_planes<'session>(
    arena: &DeviceBuffer<'session>,
    scratch: &DeviceBuffer<'session>,
    source: OperandSource,
    visible_planes: usize,
    matrix_shape: &[usize],
) -> Result<TensorDescriptor<'session>, ExecutionError> {
    if visible_planes > source.planes {
        return Err(ExecutionError::InvalidPlan(format!(
            "requested {visible_planes} visible planes from {}-plane operand",
            source.planes
        )));
    }
    let mut shape = vec![visible_planes];
    shape.extend(matrix_shape);
    buffer(source.buffer, arena, scratch).tensor_f32(
        source.offset,
        &shape,
        &row_major_strides(&shape)?,
    )
}

fn source_flat<'session>(
    arena: &DeviceBuffer<'session>,
    scratch: &DeviceBuffer<'session>,
    source: OperandSource,
    plane: usize,
) -> Result<TensorDescriptor<'session>, ExecutionError> {
    if plane >= source.planes {
        return Err(ExecutionError::InvalidPlan(format!(
            "requested plane {plane} from {}-plane operand",
            source.planes
        )));
    }
    let offset = source
        .offset
        .checked_add(plane_bytes(source.elements, plane)?)
        .ok_or_else(|| ExecutionError::Unsupported("operand offset overflow".to_string()))?;
    buffer(source.buffer, arena, scratch).tensor_f32(offset, &[source.elements], &[1])
}

fn shaped_tensor<'session>(
    buffer: &DeviceBuffer<'session>,
    slot: LocalSlot,
    shape: &[usize],
) -> Result<TensorDescriptor<'session>, ExecutionError> {
    let expected = shape
        .iter()
        .try_fold(1usize, |product, dimension| product.checked_mul(*dimension));
    if expected != Some(slot.elements) {
        return Err(ExecutionError::InvalidPlan(format!(
            "scratch slot has {} elements but shaped descriptor needs {expected:?}",
            slot.elements
        )));
    }
    buffer.tensor_f32(slot.offset, shape, &row_major_strides(shape)?)
}

fn flat_tensor<'session>(
    buffer: &DeviceBuffer<'session>,
    slot: LocalSlot,
) -> Result<TensorDescriptor<'session>, ExecutionError> {
    buffer.tensor_f32(slot.offset, &[slot.elements], &[1])
}

fn output_plane_flat<'session>(
    arena: &DeviceBuffer<'session>,
    output: &ValueLayout,
    plane: usize,
) -> Result<TensorDescriptor<'session>, ExecutionError> {
    arena.tensor_f32(plane_offset(output, plane)?, &[output.elements], &[1])
}

fn role(resources: &NodeResources, role: ScratchRole) -> Result<LocalSlot, ExecutionError> {
    resources
        .roles
        .iter()
        .find_map(|(candidate, slot)| (*candidate == role).then_some(*slot))
        .ok_or_else(|| {
            ExecutionError::InvalidPlan(format!("node scratch is missing role {role:?}"))
        })
}

fn buffer<'a, 'session>(
    kind: BufferKind,
    arena: &'a DeviceBuffer<'session>,
    scratch: &'a DeviceBuffer<'session>,
) -> &'a DeviceBuffer<'session> {
    match kind {
        BufferKind::Semantic => arena,
        BufferKind::Scratch => scratch,
    }
}

fn elements(spec: &TensorSpec) -> Result<usize, ExecutionError> {
    spec.shape
        .iter()
        .try_fold(1usize, |product, dimension| product.checked_mul(*dimension))
        .ok_or_else(|| {
            ExecutionError::InvalidPlan("tensor element count overflows usize".to_string())
        })
}

fn dimensions(spec: &TensorSpec) -> Vec<(i32, usize)> {
    spec.modes
        .iter()
        .copied()
        .zip(spec.shape.iter().copied())
        .collect()
}

fn row_major_strides(shape: &[usize]) -> Result<Vec<i64>, ExecutionError> {
    let mut strides = vec![1i64; shape.len()];
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = i64::try_from(stride)
            .map_err(|_| ExecutionError::Unsupported("physical stride exceeds i64".to_string()))?;
        stride = stride.checked_mul(shape[axis]).ok_or_else(|| {
            ExecutionError::Unsupported("physical stride product overflows usize".to_string())
        })?;
    }
    Ok(strides)
}

fn plane_offset(layout: &ValueLayout, plane: usize) -> Result<u64, ExecutionError> {
    if plane >= layout.planes {
        return Err(ExecutionError::InvalidPlan(format!(
            "value has {} planes but plane {plane} was requested",
            layout.planes
        )));
    }
    layout
        .offset
        .checked_add(plane_bytes(layout.elements, plane)?)
        .ok_or_else(|| ExecutionError::Unsupported("plane offset overflow".to_string()))
}

fn plane_bytes(elements: usize, plane: usize) -> Result<u64, ExecutionError> {
    u64::try_from(elements)
        .ok()
        .and_then(|elements| elements.checked_mul(4))
        .and_then(|bytes| bytes.checked_mul(plane as u64))
        .ok_or_else(|| ExecutionError::Unsupported("plane byte offset overflow".to_string()))
}

fn align_up(value: u64, alignment: u64) -> Result<u64, ExecutionError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| ExecutionError::Unsupported("alignment overflow".to_string()))
}

impl PreparedExecutable for super::AscendExecutable<'_> {
    fn representation(&self) -> crate::static_plan::Representation {
        self.representation.clone()
    }

    fn enqueue(&mut self) -> Result<(), ExecutionError> {
        self.state.enqueue()
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        self.state.synchronize()
    }

    fn output(&mut self) -> Result<ComplexValue, ExecutionError> {
        self.state.output()
    }
}
