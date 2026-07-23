use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum PlanError {
    InvalidFormat(String),
    InvalidNetwork(String),
    MissingContractionOrder,
    NonBinaryNode {
        children: usize,
    },
    NonScalarOutput(Vec<i32>),
    InvalidTensor {
        index: usize,
        detail: String,
    },
    InvalidTree {
        detail: String,
    },
    UnsupportedTrace {
        node: usize,
        left_trace_modes: Vec<i32>,
        right_trace_modes: Vec<i32>,
    },
    Hash(String),
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFormat(detail) => write!(f, "invalid plan format: {detail}"),
            Self::InvalidNetwork(detail) => write!(f, "invalid tensor network: {detail}"),
            Self::MissingContractionOrder => write!(f, "missing contraction order"),
            Self::NonBinaryNode { children } => {
                write!(f, "contraction tree node has {children} children; expected 2")
            }
            Self::NonScalarOutput(modes) => {
                write!(f, "static plan output is not scalar; modes: {modes:?}")
            }
            Self::InvalidTensor { index, detail } => {
                write!(f, "invalid input tensor {index}: {detail}")
            }
            Self::InvalidTree { detail } => write!(f, "invalid contraction tree: {detail}"),
            Self::UnsupportedTrace {
                node,
                left_trace_modes,
                right_trace_modes,
            } => write!(
                f,
                "node {node} requires unsupported trace reductions: left={left_trace_modes:?}, right={right_trace_modes:?}"
            ),
            Self::Hash(detail) => write!(f, "failed to hash static plan: {detail}"),
        }
    }
}

impl std::error::Error for PlanError {}
