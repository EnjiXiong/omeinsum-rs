#![allow(clippy::result_large_err)] // Frozen backend diagnostics intentionally carry full context.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::{Duration, Instant};

use rand::seq::SliceRandom;
use rand::SeedableRng;

use super::{ComplexValue, InputUpdate, LeafClass, Representation, StaticPlan};

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionError {
    InvalidPlan(String),
    Unsupported(String),
    Backend {
        stage: String,
        node_id: Option<usize>,
        operation: String,
        status_code: Option<i64>,
        logical_shape: Vec<usize>,
        physical_strides: Vec<i64>,
        device: Option<String>,
        detail: String,
    },
    NonFiniteOutput(ComplexValue),
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan(detail) => write!(f, "invalid static plan: {detail}"),
            Self::Unsupported(detail) => write!(f, "unsupported execution request: {detail}"),
            Self::Backend {
                stage,
                node_id,
                operation,
                status_code,
                detail,
                ..
            } => write!(
                f,
                "backend failure during {stage}, node {node_id:?}, operation {operation}, status {status_code:?}: {detail}"
            ),
            Self::NonFiniteOutput(value) => {
                write!(f, "execution produced non-finite output {value:?}")
            }
        }
    }
}

impl std::error::Error for ExecutionError {}

pub trait PreparedExecutable {
    fn representation(&self) -> Representation;
    fn update_inputs(&mut self, updates: &[InputUpdate<f64>]) -> Result<(), ExecutionError>;
    fn enqueue(&mut self) -> Result<(), ExecutionError>;
    fn synchronize(&mut self) -> Result<(), ExecutionError>;
    fn output(&mut self) -> Result<ComplexValue, ExecutionError>;
}

impl<T: PreparedExecutable + ?Sized> PreparedExecutable for Box<T> {
    fn representation(&self) -> Representation {
        (**self).representation()
    }

    fn update_inputs(&mut self, updates: &[InputUpdate<f64>]) -> Result<(), ExecutionError> {
        (**self).update_inputs(updates)
    }

    fn enqueue(&mut self) -> Result<(), ExecutionError> {
        (**self).enqueue()
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        (**self).synchronize()
    }

    fn output(&mut self) -> Result<ComplexValue, ExecutionError> {
        (**self).output()
    }
}

pub(crate) fn validate_input_updates(
    plan: &StaticPlan,
    leaf_classes: &[LeafClass],
    updates: &[InputUpdate<f64>],
) -> Result<(), ExecutionError> {
    if leaf_classes.len() != plan.leaf_values.len() {
        return Err(ExecutionError::InvalidPlan(
            "prepared leaf-class metadata is incomplete".to_string(),
        ));
    }
    let mut seen = vec![false; plan.leaf_values.len()];
    for update in updates {
        if update.index >= plan.leaf_values.len() {
            return Err(ExecutionError::InvalidPlan(format!(
                "input update index {} is outside {} leaves",
                update.index,
                plan.leaf_values.len()
            )));
        }
        if std::mem::replace(&mut seen[update.index], true) {
            return Err(ExecutionError::InvalidPlan(format!(
                "input update index {} appears more than once",
                update.index
            )));
        }
        let value = &plan.values[plan.leaf_values[update.index].0];
        if update.tensor.spec != value.tensor {
            return Err(ExecutionError::InvalidPlan(format!(
                "input update {} geometry differs from its prepared leaf",
                update.index
            )));
        }
        let elements = update
            .tensor
            .spec
            .shape
            .iter()
            .try_fold(1usize, |product, dimension| product.checked_mul(*dimension))
            .ok_or_else(|| {
                ExecutionError::InvalidPlan(format!(
                    "input update {} element count overflows usize",
                    update.index
                ))
            })?;
        if update.tensor.real.len() != elements || update.tensor.imag.len() != elements {
            return Err(ExecutionError::InvalidPlan(format!(
                "input update {} plane length differs from its geometry",
                update.index
            )));
        }
        if update.tensor.class != leaf_classes[update.index] {
            return Err(ExecutionError::InvalidPlan(format!(
                "input update {} changes the frozen leaf class",
                update.index
            )));
        }
        if update.tensor.class == LeafClass::Real
            && update.tensor.imag.iter().any(|value| *value != 0.0)
        {
            return Err(ExecutionError::InvalidPlan(format!(
                "input update {} real leaf has a nonzero imaginary plane",
                update.index
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub warmups: usize,
    pub samples: usize,
    pub min_sample_ms: u64,
    pub measurement_order_seed: u64,
}

pub struct BenchmarkTarget<'a> {
    pub backend: &'a str,
    pub dtype: &'a str,
    pub execution_mode: &'a str,
    pub executable: &'a mut dyn PreparedExecutable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkDiagnostics {
    pub warmup_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModeTiming {
    pub representation: Representation,
    pub backend: String,
    pub dtype: String,
    pub execution_mode: String,
    pub warmups: usize,
    pub samples: usize,
    pub min_sample_ms: u64,
    pub measurement_order_seed: u64,
    pub inner_iterations: usize,
    pub raw_seconds_per_contraction: Vec<f64>,
    pub best_seconds: f64,
    pub median_seconds: f64,
    pub iqr_seconds: f64,
    pub output: ComplexValue,
}

pub fn benchmark_prepared(
    targets: &mut [BenchmarkTarget<'_>],
    config: &BenchmarkConfig,
) -> Result<Vec<ModeTiming>, ExecutionError> {
    benchmark_prepared_with_diagnostics(targets, config).map(|(timings, _)| timings)
}

pub fn benchmark_prepared_with_diagnostics(
    targets: &mut [BenchmarkTarget<'_>],
    config: &BenchmarkConfig,
) -> Result<(Vec<ModeTiming>, BenchmarkDiagnostics), ExecutionError> {
    if targets.is_empty() {
        return Err(ExecutionError::Unsupported(
            "at least one benchmark target is required".to_string(),
        ));
    }
    if config.warmups == 0 || config.samples == 0 || config.min_sample_ms == 0 {
        return Err(ExecutionError::Unsupported(
            "warmups, samples, and min_sample_ms must all be positive".to_string(),
        ));
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(config.measurement_order_seed);
    let mut order: Vec<_> = (0..targets.len()).collect();
    let warmup_start = Instant::now();
    for _ in 0..config.warmups {
        order.shuffle(&mut rng);
        for index in order.iter().copied() {
            targets[index].executable.enqueue()?;
            targets[index].executable.synchronize()?;
        }
    }
    let warmup_seconds = warmup_start.elapsed().as_secs_f64();

    let threshold = Duration::from_millis(config.min_sample_ms);
    let mut inner_iterations = Vec::with_capacity(targets.len());
    for target in targets.iter_mut() {
        let mut iterations = 1usize;
        loop {
            let start = Instant::now();
            for _ in 0..iterations {
                target.executable.enqueue()?;
            }
            target.executable.synchronize()?;
            if start.elapsed() >= threshold {
                break;
            }
            iterations = iterations.checked_mul(2).ok_or_else(|| {
                ExecutionError::Unsupported(
                    "calibration iteration count overflowed usize".to_string(),
                )
            })?;
        }
        inner_iterations.push(iterations);
    }

    let mut raw = vec![Vec::with_capacity(config.samples); targets.len()];
    for _ in 0..config.samples {
        order.shuffle(&mut rng);
        for index in order.iter().copied() {
            let iterations = inner_iterations[index];
            let start = Instant::now();
            for _ in 0..iterations {
                targets[index].executable.enqueue()?;
            }
            targets[index].executable.synchronize()?;
            raw[index].push(start.elapsed().as_secs_f64() / iterations as f64);
        }
    }

    let timings = targets
        .iter_mut()
        .zip(inner_iterations)
        .zip(raw)
        .map(
            |((target, inner_iterations), raw_seconds_per_contraction)| {
                let output = target.executable.output()?;
                if !output.re.is_finite() || !output.im.is_finite() {
                    return Err(ExecutionError::NonFiniteOutput(output));
                }
                let (best_seconds, median_seconds, iqr_seconds) =
                    summarize(&raw_seconds_per_contraction)?;
                Ok(ModeTiming {
                    representation: target.executable.representation(),
                    backend: target.backend.to_string(),
                    dtype: target.dtype.to_string(),
                    execution_mode: target.execution_mode.to_string(),
                    warmups: config.warmups,
                    samples: config.samples,
                    min_sample_ms: config.min_sample_ms,
                    measurement_order_seed: config.measurement_order_seed,
                    inner_iterations,
                    raw_seconds_per_contraction,
                    best_seconds,
                    median_seconds,
                    iqr_seconds,
                    output,
                })
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    Ok((timings, BenchmarkDiagnostics { warmup_seconds }))
}

fn summarize(samples: &[f64]) -> Result<(f64, f64, f64), ExecutionError> {
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return Err(ExecutionError::Unsupported(
            "timing samples must be non-empty and finite".to_string(),
        ));
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = slice_median(&sorted);
    let midpoint = sorted.len() / 2;
    let (q1, q3) = if sorted.len() == 1 {
        (sorted[0], sorted[0])
    } else if sorted.len().is_multiple_of(2) {
        (
            slice_median(&sorted[..midpoint]),
            slice_median(&sorted[midpoint..]),
        )
    } else {
        (
            slice_median(&sorted[..midpoint]),
            slice_median(&sorted[midpoint + 1..]),
        )
    };
    Ok((sorted[0], median, q3 - q1))
}

fn slice_median(sorted: &[f64]) -> f64 {
    let midpoint = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[midpoint - 1] + sorted[midpoint]) / 2.0
    } else {
        sorted[midpoint]
    }
}
