#![allow(clippy::result_large_err)] // Frozen backend diagnostics intentionally carry full context.

use serde::{Deserialize, Serialize};
use std::fmt;

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
