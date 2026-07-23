use std::collections::HashSet;

use omeinsum::static_plan::{
    contract_complex64, prepare_cpu_f32, prepare_cpu_f64, ComplexValue, PlanBundle,
    PreparedExecutable, Representation, StaticPlan,
};
use serde::Serialize;

#[derive(Debug, Clone, Copy)]
pub(crate) enum CpuDtype {
    F64,
    F32,
}

impl CpuDtype {
    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        match value {
            "f64" => Ok(Self::F64),
            "f32" => Ok(Self::F32),
            _ => Err(format!(
                "unsupported dtype {value:?}; expected one of: f64, f32"
            )),
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::F32 => "f32",
        }
    }
}

#[derive(Serialize)]
struct ExecutionCheck {
    format: &'static str,
    tree_hash: String,
    reference_complex64: ComplexValue,
    executions: Vec<ExecutionResult>,
}

#[derive(Serialize)]
struct ExecutionResult {
    representation: Representation,
    backend: &'static str,
    dtype: &'static str,
    execution_mode: &'static str,
    output: ComplexValue,
}

pub(crate) fn run(
    plan_path: &str,
    backend: &str,
    representations: &str,
    dtype: &str,
    output: Option<&str>,
    pretty: Option<bool>,
) -> Result<(), String> {
    validate_cpu_backend(backend)?;
    let dtype = CpuDtype::parse(dtype)?;
    let bundle = read_bundle(plan_path)?;
    let selected = select_plans(&bundle, representations)?;
    let reference_complex64 = contract_complex64(&bundle.realified_rank3, &bundle.inputs)
        .map_err(|error| error.to_string())?;

    let mut executions = Vec::with_capacity(selected.len());
    for plan in selected {
        let mut executable = prepare_cpu(plan, &bundle, dtype)?;
        executable.enqueue().map_err(|error| error.to_string())?;
        executable
            .synchronize()
            .map_err(|error| error.to_string())?;
        let value = executable.output().map_err(|error| error.to_string())?;
        executions.push(ExecutionResult {
            representation: executable.representation(),
            backend: "cpu",
            dtype: dtype.name(),
            execution_mode: "prepared",
            output: value,
        });
    }

    let report = ExecutionCheck {
        format: "omeinsum-execution-check-v1",
        tree_hash: bundle.tree_hash,
        reference_complex64,
        executions,
    };
    crate::common::write_json_output(&report, output, pretty)
}

pub(crate) fn validate_cpu_backend(backend: &str) -> Result<(), String> {
    if backend == "cpu" {
        Ok(())
    } else {
        Err(format!(
            "unsupported backend {backend:?}; this build supports: cpu"
        ))
    }
}

pub(crate) fn read_bundle(path: &str) -> Result<PlanBundle, String> {
    let json = std::fs::read_to_string(path)
        .map_err(|error| format!("Failed to read '{path}': {error}"))?;
    let bundle: PlanBundle = serde_json::from_str(&json)
        .map_err(|error| format!("Failed to parse static plan JSON: {error}"))?;
    bundle.validate().map_err(|error| error.to_string())?;
    Ok(bundle)
}

pub(crate) fn select_plans<'a>(
    bundle: &'a PlanBundle,
    representations: &str,
) -> Result<Vec<&'a StaticPlan>, String> {
    let mut selected = Vec::new();
    let mut seen = HashSet::new();
    for token in representations.split(',').map(str::trim) {
        let plan = match token {
            "real-skeleton" => &bundle.real_skeleton,
            "flat-4m" => &bundle.flat_4m,
            "realified-rank3" => &bundle.realified_rank3,
            "" => {
                return Err("representations must contain at least one non-empty name".to_string())
            }
            _ => {
                return Err(format!(
                    "unsupported representation {token:?}; expected one of: \
                     real-skeleton, flat-4m, realified-rank3"
                ))
            }
        };
        if !seen.insert(token) {
            return Err(format!("duplicate representation {token:?}"));
        }
        selected.push(plan);
    }
    if selected.is_empty() {
        return Err("at least one representation is required".to_string());
    }
    Ok(selected)
}

pub(crate) fn prepare_cpu(
    plan: &StaticPlan,
    bundle: &PlanBundle,
    dtype: CpuDtype,
) -> Result<Box<dyn PreparedExecutable>, String> {
    match dtype {
        CpuDtype::F64 => prepare_cpu_f64(plan, &bundle.inputs),
        CpuDtype::F32 => prepare_cpu_f32(plan, &bundle.inputs),
    }
    .map_err(|error| error.to_string())
}
