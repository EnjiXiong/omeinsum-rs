#![allow(clippy::result_large_err)] // Frozen backend diagnostics intentionally carry full context.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::{Duration, Instant};

use rand::seq::SliceRandom;
use rand::SeedableRng;

use super::{ComplexValue, Representation};

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
    fn enqueue(&mut self) -> Result<(), ExecutionError>;
    fn synchronize(&mut self) -> Result<(), ExecutionError>;
    fn output(&mut self) -> Result<ComplexValue, ExecutionError>;
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
    for _ in 0..config.warmups {
        order.shuffle(&mut rng);
        for index in order.iter().copied() {
            targets[index].executable.enqueue()?;
            targets[index].executable.synchronize()?;
        }
    }

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

    targets
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
        .collect()
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
