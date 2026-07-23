use std::collections::HashSet;

use omeinsum::static_plan::{
    contract_complex64, prepare_cpu_f32, prepare_cpu_f64, ComplexValue, PlanBundle,
    PreparedExecutable, Representation, StaticPlan,
};
use serde::Serialize;

#[cfg(feature = "ascend")]
use omeinsum::backend::ascend::{
    AscendExecutable, AscendExecutableConfig, AscendExecutionMode, AscendMemoryStats,
    AscendPrecisionMode, AscendSession, AscendSessionConfig,
};

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
    #[cfg(feature = "ascend")]
    device: Option<omeinsum::backend::ascend::AscendDeviceInfo>,
    #[cfg(feature = "ascend")]
    precision_mode: Option<AscendPrecisionMode>,
    #[cfg(feature = "ascend")]
    lowering: Vec<(Representation, Vec<omeinsum::static_plan::LoweredNodeTrace>)>,
}

#[derive(Serialize)]
struct ExecutionResult {
    representation: Representation,
    backend: String,
    dtype: String,
    execution_mode: String,
    output: ComplexValue,
    #[cfg(feature = "ascend")]
    memory: Option<AscendMemoryStats>,
}

#[cfg(feature = "ascend")]
type AscendExecutionOutcome = (
    Vec<ExecutionResult>,
    Option<omeinsum::backend::ascend::AscendDeviceInfo>,
    Option<AscendPrecisionMode>,
);

#[cfg(not(feature = "ascend"))]
type AscendExecutionOutcome = (Vec<ExecutionResult>, Option<()>, Option<()>);

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    plan_path: &str,
    backend: &str,
    representations: &str,
    dtype: &str,
    device_id: Option<i32>,
    precision_mode: &str,
    output: Option<&str>,
    pretty: Option<bool>,
) -> Result<(), String> {
    let bundle = read_bundle(plan_path)?;
    let selected = select_plans(&bundle, representations)?;
    #[cfg(feature = "ascend")]
    let lowering = if backend == "ascend" {
        selected
            .iter()
            .map(|plan| {
                omeinsum::static_plan::lower_plan_traces(plan)
                    .map(|trace| (plan.representation.clone(), trace))
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![]
    };
    let reference_complex64 = contract_complex64(&bundle.realified_rank3, &bundle.inputs)
        .map_err(|error| error.to_string())?;

    let (executions, _device, _precision) = match backend {
        "cpu" => {
            let dtype = CpuDtype::parse(dtype)?;
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
                    backend: "cpu".to_string(),
                    dtype: dtype.name().to_string(),
                    execution_mode: "prepared".to_string(),
                    output: value,
                    #[cfg(feature = "ascend")]
                    memory: None,
                });
            }
            (executions, None, None)
        }
        "ascend" => execute_ascend(selected, &bundle, dtype, device_id, precision_mode)?,
        _ => {
            return Err(format!(
                "unsupported backend {backend:?}; expected cpu or ascend"
            ))
        }
    };

    let report = ExecutionCheck {
        format: "omeinsum-execution-check-v1",
        tree_hash: bundle.tree_hash,
        reference_complex64,
        executions,
        #[cfg(feature = "ascend")]
        device: _device,
        #[cfg(feature = "ascend")]
        precision_mode: _precision,
        #[cfg(feature = "ascend")]
        lowering,
    };
    crate::common::write_json_output(&report, output, pretty)
}

#[cfg(feature = "ascend")]
fn execute_ascend(
    selected: Vec<&StaticPlan>,
    bundle: &PlanBundle,
    dtype: &str,
    device_id: Option<i32>,
    precision_mode: &str,
) -> Result<AscendExecutionOutcome, String> {
    if dtype != "f32" {
        return Err("Ascend execution requires --dtype f32".to_string());
    }
    let precision = parse_ascend_precision(precision_mode)?;
    let device_id =
        device_id.ok_or_else(|| "--device-id is required with --backend ascend".to_string())?;
    let session = AscendSession::new(&AscendSessionConfig {
        device_id,
        precision_mode: precision,
    })
    .map_err(|error| error.to_string())?;
    let mut executions = Vec::with_capacity(selected.len());
    for plan in selected {
        let mut executable = AscendExecutable::prepare(
            &session,
            plan,
            &bundle.inputs,
            &AscendExecutableConfig {
                execution_mode: AscendExecutionMode::RepeatableAclnn,
            },
        )
        .map_err(|error| error.to_string())?;
        let memory = executable.memory_stats().clone();
        executable.enqueue().map_err(|error| error.to_string())?;
        executable
            .synchronize()
            .map_err(|error| error.to_string())?;
        let value = executable.output().map_err(|error| error.to_string())?;
        executions.push(ExecutionResult {
            representation: executable.representation(),
            backend: "ascend".to_string(),
            dtype: "f32".to_string(),
            execution_mode: "repeatable-aclnn".to_string(),
            output: value,
            memory: Some(memory),
        });
    }
    Ok((
        executions,
        Some(session.device_info().clone()),
        Some(precision),
    ))
}

#[cfg(not(feature = "ascend"))]
fn execute_ascend(
    _selected: Vec<&StaticPlan>,
    _bundle: &PlanBundle,
    _dtype: &str,
    _device_id: Option<i32>,
    _precision_mode: &str,
) -> Result<AscendExecutionOutcome, String> {
    Err("Ascend backend unavailable; rebuild with --features ascend".to_string())
}

#[cfg(feature = "ascend")]
pub(crate) fn parse_ascend_precision(value: &str) -> Result<AscendPrecisionMode, String> {
    match value {
        "keep-dtype" => Ok(AscendPrecisionMode::KeepDtype),
        _ => Err(format!(
            "unsupported Ascend precision mode {value:?}; expected keep-dtype"
        )),
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
