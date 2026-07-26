use super::{
    ComplexNetwork, ComplexValue, InputSet, InputTensor, LeafClass, LeafPreprocessing,
    PhaseCanonicalizationReport, PlanError,
};

#[derive(Clone, Copy)]
struct UnitPhase {
    re: f64,
    im: f64,
}

#[derive(Clone, Copy)]
struct PhaseInterval {
    lower: f64,
    upper: f64,
}

impl UnitPhase {
    const ONE: Self = Self { re: 1.0, im: 0.0 };

    fn normalized(re: f64, im: f64) -> Option<Self> {
        let scale = re.abs().max(im.abs());
        if scale == 0.0 || !scale.is_finite() {
            return None;
        }
        let scaled_re = re / scale;
        let scaled_im = im / scale;
        let norm = scaled_re.hypot(scaled_im);
        let phase = Self {
            re: scaled_re / norm,
            im: scaled_im / norm,
        };
        (phase.re.is_finite() && phase.im.is_finite()).then_some(phase)
    }

    fn multiply(self, right: Self) -> Option<Self> {
        Self::normalized(
            self.re * right.re - self.im * right.im,
            self.re * right.im + self.im * right.re,
        )
    }
}

pub(super) fn classify_inputs(
    network: &ComplexNetwork<f64>,
    realness_tol: f64,
    leaf_preprocessing: LeafPreprocessing,
) -> Result<(InputSet<f64>, Option<PhaseCanonicalizationReport>), PlanError> {
    let inputs = classify_raw_inputs(network, realness_tol)?;
    match leaf_preprocessing {
        LeafPreprocessing::Raw => Ok((inputs, None)),
        LeafPreprocessing::PhaseCanonicalized => {
            let (inputs, report) = phase_canonicalize(inputs, realness_tol)?;
            Ok((inputs, Some(report)))
        }
    }
}

fn classify_raw_inputs(
    network: &ComplexNetwork<f64>,
    realness_tol: f64,
) -> Result<InputSet<f64>, PlanError> {
    if !realness_tol.is_finite() || realness_tol <= 0.0 {
        return Err(PlanError::InvalidNetwork(format!(
            "realness tolerance must be positive and finite, got {realness_tol}"
        )));
    }
    let tensors = network
        .tensors
        .iter()
        .enumerate()
        .map(|(index, tensor)| {
            let elements = tensor
                .spec
                .shape
                .iter()
                .try_fold(1usize, |product, size| product.checked_mul(*size));
            let elements = elements.ok_or_else(|| PlanError::InvalidTensor {
                index,
                detail: "shape product overflows usize".to_string(),
            })?;
            if tensor.real.len() != elements || tensor.imag.len() != elements {
                return Err(PlanError::InvalidTensor {
                    index,
                    detail: format!(
                        "shape has {elements} elements but real/imag lengths are {}/{}",
                        tensor.real.len(),
                        tensor.imag.len()
                    ),
                });
            }
            if let Some((plane, position, value)) = tensor
                .real
                .iter()
                .enumerate()
                .map(|(position, value)| ("real", position, *value))
                .chain(
                    tensor
                        .imag
                        .iter()
                        .enumerate()
                        .map(|(position, value)| ("imag", position, *value)),
                )
                .find(|(_, _, value)| !value.is_finite())
            {
                return Err(PlanError::InvalidTensor {
                    index,
                    detail: format!("{plane}[{position}] is non-finite: {value}"),
                });
            }
            let imag_max = observed_imag_max(&tensor.imag);
            let class = classify(imag_max, realness_tol);
            let imag = if class == LeafClass::Real {
                vec![0.0; elements]
            } else {
                tensor.imag.clone()
            };
            Ok(InputTensor {
                spec: tensor.spec.clone(),
                real: tensor.real.clone(),
                imag,
                class,
                imag_max,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(InputSet { tensors })
}

fn phase_canonicalize(
    mut inputs: InputSet<f64>,
    realness_tol: f64,
) -> Result<(InputSet<f64>, PhaseCanonicalizationReport), PlanError> {
    let source_real_leaf_count = inputs
        .tensors
        .iter()
        .filter(|tensor| tensor.class == LeafClass::Real)
        .count();
    let source_complex_leaf_count = inputs.tensors.len() - source_real_leaf_count;
    let mut accumulated_phase = UnitPhase::ONE;
    let mut canonicalized_indices = Vec::new();
    let mut genuinely_complex_anchor = None;

    for (index, tensor) in inputs.tensors.iter_mut().enumerate() {
        if tensor.class == LeafClass::Real {
            continue;
        }
        match phase_real_form(tensor, realness_tol, index)? {
            Some((real, phase, rotated_imag_max)) => {
                tensor.real = real;
                tensor.imag.fill(0.0);
                tensor.class = LeafClass::Real;
                tensor.imag_max = rotated_imag_max;
                accumulated_phase =
                    accumulated_phase
                        .multiply(phase)
                        .ok_or_else(|| PlanError::InvalidTensor {
                            index,
                            detail: "accumulated leaf phase is non-finite".to_string(),
                        })?;
                canonicalized_indices.push(index);
            }
            None => {
                genuinely_complex_anchor.get_or_insert(index);
            }
        }
    }

    let phase_anchor = if canonicalized_indices.is_empty() {
        None
    } else {
        let anchor = genuinely_complex_anchor
            .or_else(|| {
                canonicalized_indices.iter().copied().find(|index| {
                    inputs.tensors[*index]
                        .real
                        .iter()
                        .any(|value| *value != 0.0)
                })
            })
            .ok_or_else(|| {
                PlanError::InvalidNetwork(
                    "phase-canonicalized inputs have no nonzero phase anchor".to_string(),
                )
            })?;
        apply_phase(
            &mut inputs.tensors[anchor],
            accumulated_phase,
            realness_tol,
            anchor,
        )?;
        Some(anchor)
    };

    Ok((
        inputs,
        PhaseCanonicalizationReport {
            source_real_leaf_count,
            source_complex_leaf_count,
            canonicalized_leaf_count: canonicalized_indices.len(),
            phase_anchor,
            accumulated_phase: ComplexValue {
                re: accumulated_phase.re,
                im: accumulated_phase.im,
            },
        },
    ))
}

fn phase_real_form(
    tensor: &InputTensor<f64>,
    realness_tol: f64,
    index: usize,
) -> Result<Option<(Vec<f64>, UnitPhase, f64)>, PlanError> {
    let candidates = feasible_phase_candidates(tensor, realness_tol);
    if candidates.is_empty() {
        return Ok(None);
    }
    let mut best = None;
    let mut saw_finite_rotation = false;
    for phase in candidates {
        let Some(rotated_imag_max) = rotated_imag_max(tensor, phase) else {
            continue;
        };
        saw_finite_rotation = true;
        if best
            .as_ref()
            .is_none_or(|(_, best_imag_max)| rotated_imag_max < *best_imag_max)
        {
            best = Some((phase, rotated_imag_max));
        }
    }
    let Some((phase, rotated_imag_max)) = best else {
        if !saw_finite_rotation {
            return Err(PlanError::InvalidTensor {
                index,
                detail: "phase rotation produced non-finite data".to_string(),
            });
        }
        return Ok(None);
    };
    if rotated_imag_max > realness_tol {
        return Ok(None);
    }

    let mut real = Vec::with_capacity(tensor.real.len());
    for (position, (source_re, source_im)) in tensor.real.iter().zip(&tensor.imag).enumerate() {
        let rotated_re = phase.re * source_re + phase.im * source_im;
        let rotated_im = phase.re * source_im - phase.im * source_re;
        if !rotated_re.is_finite() || !rotated_im.is_finite() {
            return Err(PlanError::InvalidTensor {
                index,
                detail: format!("phase rotation produced non-finite data at element {position}"),
            });
        }
        real.push(rotated_re);
    }
    Ok(Some((real, phase, rotated_imag_max)))
}

fn feasible_phase_candidates(tensor: &InputTensor<f64>, realness_tol: f64) -> Vec<UnitPhase> {
    const PERIOD: f64 = std::f64::consts::PI;
    const INTERVAL_ROUNDOFF: f64 = 8.0 * f64::EPSILON;

    let mut feasible = vec![PhaseInterval {
        lower: 0.0,
        upper: PERIOD,
    }];
    let mut constrained = false;
    for (real, imag) in tensor.real.iter().zip(&tensor.imag) {
        let magnitude = real.hypot(*imag);
        if magnitude <= realness_tol {
            continue;
        }
        constrained = true;
        let ratio = if magnitude.is_infinite() {
            0.0
        } else {
            (realness_tol / magnitude).clamp(0.0, 1.0)
        };
        let half_width = (ratio.asin() + INTERVAL_ROUNDOFF).min(std::f64::consts::FRAC_PI_2);
        if half_width == std::f64::consts::FRAC_PI_2 {
            continue;
        }
        let center = imag.atan2(*real).rem_euclid(PERIOD);
        let allowed = circular_interval(center, half_width);
        feasible = intersect_intervals(&feasible, &allowed);
        if feasible.is_empty() {
            return Vec::new();
        }
    }
    if !constrained {
        return Vec::new();
    }

    let mut candidates = Vec::with_capacity(feasible.len() * 3);
    for interval in feasible {
        for angle in [
            (interval.lower + interval.upper) * 0.5,
            interval.lower,
            interval.upper,
        ] {
            let (im, re) = angle.sin_cos();
            candidates.push(UnitPhase { re, im });
        }
    }
    candidates
}

fn circular_interval(center: f64, half_width: f64) -> Vec<PhaseInterval> {
    let period = std::f64::consts::PI;
    let lower = center - half_width;
    let upper = center + half_width;
    if lower < 0.0 {
        vec![
            PhaseInterval { lower: 0.0, upper },
            PhaseInterval {
                lower: lower + period,
                upper: period,
            },
        ]
    } else if upper > period {
        vec![
            PhaseInterval {
                lower: 0.0,
                upper: upper - period,
            },
            PhaseInterval {
                lower,
                upper: period,
            },
        ]
    } else {
        vec![PhaseInterval { lower, upper }]
    }
}

fn intersect_intervals(left: &[PhaseInterval], right: &[PhaseInterval]) -> Vec<PhaseInterval> {
    let mut intersections = Vec::new();
    for left in left {
        for right in right {
            let lower = left.lower.max(right.lower);
            let upper = left.upper.min(right.upper);
            if lower <= upper {
                intersections.push(PhaseInterval { lower, upper });
            }
        }
    }
    intersections.sort_by(|left, right| left.lower.total_cmp(&right.lower));
    let mut merged: Vec<PhaseInterval> = Vec::with_capacity(intersections.len());
    for interval in intersections {
        if let Some(previous) = merged.last_mut() {
            if interval.lower <= previous.upper {
                previous.upper = previous.upper.max(interval.upper);
                continue;
            }
        }
        merged.push(interval);
    }
    merged
}

fn rotated_imag_max(tensor: &InputTensor<f64>, phase: UnitPhase) -> Option<f64> {
    let mut maximum = 0.0_f64;
    for (real, imag) in tensor.real.iter().zip(&tensor.imag) {
        let rotated_imag = phase.re * imag - phase.im * real;
        if !rotated_imag.is_finite() {
            return None;
        }
        maximum = maximum.max(rotated_imag.abs());
    }
    Some(maximum)
}

fn apply_phase(
    tensor: &mut InputTensor<f64>,
    phase: UnitPhase,
    realness_tol: f64,
    index: usize,
) -> Result<(), PlanError> {
    for (position, (real, imag)) in tensor.real.iter_mut().zip(&mut tensor.imag).enumerate() {
        let rotated_re = phase.re * *real - phase.im * *imag;
        let rotated_im = phase.re * *imag + phase.im * *real;
        if !rotated_re.is_finite() || !rotated_im.is_finite() {
            return Err(PlanError::InvalidTensor {
                index,
                detail: format!(
                    "phase-anchor rotation produced non-finite data at element {position}"
                ),
            });
        }
        *real = rotated_re;
        *imag = rotated_im;
    }
    tensor.imag_max = observed_imag_max(&tensor.imag);
    tensor.class = classify(tensor.imag_max, realness_tol);
    if tensor.class == LeafClass::Real {
        tensor.imag.fill(0.0);
    }
    Ok(())
}

fn classify(imag_max: f64, realness_tol: f64) -> LeafClass {
    if imag_max <= realness_tol {
        LeafClass::Real
    } else {
        LeafClass::Complex
    }
}

fn observed_imag_max(imag: &[f64]) -> f64 {
    imag.iter().map(|value| value.abs()).fold(0.0, f64::max)
}
