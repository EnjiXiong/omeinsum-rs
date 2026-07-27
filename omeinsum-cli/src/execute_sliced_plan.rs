use omeinsum::static_plan::{
    contract_complex64, gray_assignments, slice_inputs, ComplexValue, PreparedExecutable,
    Representation, SliceSpec, SlicedExecutable, SlicedPlanBundle, SlicedVolume,
};
use serde::Serialize;

use crate::execute_plan::{prepare_cpu, select_plans, CpuDtype};
use crate::sliced_reference_policy::SlicedReferencePolicy;

#[cfg(feature = "ascend")]
use crate::execute_plan::parse_ascend_precision;
#[cfg(feature = "ascend")]
use omeinsum::backend::ascend::{
    AscendDeviceInfo, AscendExecutable, AscendExecutableConfig, AscendExecutionMode,
    AscendMemoryStats, AscendPhaseTimings, AscendPrecisionMode, AscendSession, AscendSessionConfig,
};

#[derive(Serialize)]
struct SlicedExecutionCheck {
    format: &'static str,
    source_tree_hash: String,
    reduced_tree_hash: String,
    source_plan_hashes: Vec<(Representation, String)>,
    reduced_plan_hashes: Vec<(Representation, String)>,
    slicing: SliceSpec,
    aggregate_volumes: Vec<SlicedVolume>,
    reference_policy: &'static str,
    correctness_admission: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_complex64: Option<ComplexValue>,
    executions: Vec<SlicedExecutionResult>,
    #[cfg(feature = "ascend")]
    device: Option<AscendDeviceInfo>,
    #[cfg(feature = "ascend")]
    precision_mode: Option<AscendPrecisionMode>,
    #[cfg(feature = "ascend")]
    memory: Vec<(Representation, AscendMemoryStats)>,
    #[cfg(feature = "ascend")]
    phases: Vec<(Representation, AscendPhaseTimings)>,
    #[cfg(feature = "ascend")]
    lowering: Vec<(Representation, Vec<omeinsum::static_plan::LoweredNodeTrace>)>,
}

#[derive(Serialize)]
struct SlicedExecutionResult {
    representation: Representation,
    backend: String,
    dtype: String,
    execution_mode: String,
    output: ComplexValue,
}

struct SlicedExecutionOutcome {
    executions: Vec<SlicedExecutionResult>,
    #[cfg(feature = "ascend")]
    device: Option<AscendDeviceInfo>,
    #[cfg(feature = "ascend")]
    precision_mode: Option<AscendPrecisionMode>,
    #[cfg(feature = "ascend")]
    memory: Vec<(Representation, AscendMemoryStats)>,
    #[cfg(feature = "ascend")]
    phases: Vec<(Representation, AscendPhaseTimings)>,
    #[cfg(feature = "ascend")]
    lowering: Vec<(Representation, Vec<omeinsum::static_plan::LoweredNodeTrace>)>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    plan_path: &str,
    backend: &str,
    representations: &str,
    dtype: &str,
    device_id: Option<i32>,
    precision_mode: &str,
    reference_policy: &str,
    output: Option<&str>,
    pretty: Option<bool>,
) -> Result<(), String> {
    let bundle = read_sliced_bundle(plan_path)?;
    let selected = select_plans(&bundle.reduced, representations)?;
    let reference_policy = SlicedReferencePolicy::parse(reference_policy)?;
    let reference_complex64 =
        reference_policy.evaluate(backend, || sliced_reference_complex64(&bundle))?;
    let outcome = match backend {
        "cpu" => {
            if device_id.is_some() {
                return Err("--device-id is only valid with --backend ascend".to_string());
            }
            if precision_mode != "keep-dtype" {
                return Err(
                    "CPU sliced execution accepts only --precision-mode keep-dtype".to_string(),
                );
            }
            SlicedExecutionOutcome {
                executions: execute_cpu(&bundle, selected, CpuDtype::parse(dtype)?)?,
                #[cfg(feature = "ascend")]
                device: None,
                #[cfg(feature = "ascend")]
                precision_mode: None,
                #[cfg(feature = "ascend")]
                memory: vec![],
                #[cfg(feature = "ascend")]
                phases: vec![],
                #[cfg(feature = "ascend")]
                lowering: vec![],
            }
        }
        "ascend" => execute_ascend(&bundle, selected, dtype, device_id, precision_mode)?,
        _ => {
            return Err(format!(
                "unsupported backend {backend:?}; expected cpu or ascend"
            ));
        }
    };
    let reduced_plan_hashes = [
        &bundle.reduced.real_skeleton,
        &bundle.reduced.flat_4m,
        &bundle.reduced.realified_rank3,
    ]
    .into_iter()
    .map(|plan| (plan.representation.clone(), plan.plan_hash.clone()))
    .collect();
    let report = SlicedExecutionCheck {
        format: "omeinsum-sliced-execution-check-v1",
        source_tree_hash: bundle.source_tree_hash,
        reduced_tree_hash: bundle.reduced.tree_hash,
        source_plan_hashes: bundle.source_plan_hashes,
        reduced_plan_hashes,
        slicing: bundle.slice,
        aggregate_volumes: bundle.aggregate_volumes,
        reference_policy: reference_policy.label(),
        correctness_admission: reference_policy.correctness_admission(),
        reference_complex64,
        executions: outcome.executions,
        #[cfg(feature = "ascend")]
        device: outcome.device,
        #[cfg(feature = "ascend")]
        precision_mode: outcome.precision_mode,
        #[cfg(feature = "ascend")]
        memory: outcome.memory,
        #[cfg(feature = "ascend")]
        phases: outcome.phases,
        #[cfg(feature = "ascend")]
        lowering: outcome.lowering,
    };
    crate::common::write_json_output(&report, output, pretty)
}

fn execute_cpu(
    bundle: &SlicedPlanBundle,
    selected: Vec<&omeinsum::static_plan::StaticPlan>,
    dtype: CpuDtype,
) -> Result<Vec<SlicedExecutionResult>, String> {
    selected
        .into_iter()
        .map(|plan| {
            let inner = prepare_cpu(plan, &bundle.reduced, dtype)?;
            let mut executable = SlicedExecutable::new(inner, &bundle.source_inputs, bundle)
                .map_err(|error| error.to_string())?;
            executable.enqueue().map_err(|error| error.to_string())?;
            executable
                .synchronize()
                .map_err(|error| error.to_string())?;
            let output = executable.output().map_err(|error| error.to_string())?;
            Ok(SlicedExecutionResult {
                representation: executable.representation(),
                backend: "cpu".to_string(),
                dtype: dtype.name().to_string(),
                execution_mode: executable.execution_mode_label().to_string(),
                output,
            })
        })
        .collect()
}

#[cfg(feature = "ascend")]
fn execute_ascend(
    bundle: &SlicedPlanBundle,
    selected: Vec<&omeinsum::static_plan::StaticPlan>,
    dtype: &str,
    device_id: Option<i32>,
    precision_mode: &str,
) -> Result<SlicedExecutionOutcome, String> {
    if dtype != "f32" {
        return Err("Ascend sliced execution requires --dtype f32".to_string());
    }
    let precision = parse_ascend_precision(precision_mode)?;
    let device_id =
        device_id.ok_or_else(|| "--device-id is required with --backend ascend".to_string())?;
    let session = AscendSession::new(&AscendSessionConfig {
        device_id,
        precision_mode: precision,
    })
    .map_err(|error| error.to_string())?;
    let lowering = selected
        .iter()
        .map(|plan| {
            omeinsum::static_plan::lower_plan_traces(plan)
                .map(|trace| (plan.representation.clone(), trace))
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut executions = Vec::with_capacity(selected.len());
    let mut memory = Vec::with_capacity(selected.len());
    let mut phases = Vec::with_capacity(selected.len());
    for plan in selected {
        let inner = AscendExecutable::prepare(
            &session,
            plan,
            &bundle.reduced.inputs,
            &AscendExecutableConfig {
                execution_mode: AscendExecutionMode::RepeatableAclnn,
            },
        )
        .map_err(|error| error.to_string())?;
        let mut executable = SlicedExecutable::new(inner, &bundle.source_inputs, bundle)
            .map_err(|error| error.to_string())?;
        executable.enqueue().map_err(|error| error.to_string())?;
        executable
            .synchronize()
            .map_err(|error| error.to_string())?;
        let output = executable.output().map_err(|error| error.to_string())?;
        let representation = executable.representation();
        memory.push((
            representation.clone(),
            executable.inner().memory_stats().clone(),
        ));
        phases.push((representation.clone(), executable.inner().phase_timings()));
        executions.push(SlicedExecutionResult {
            representation,
            backend: "ascend".to_string(),
            dtype: "f32".to_string(),
            execution_mode: executable.execution_mode_label().to_string(),
            output,
        });
    }
    Ok(SlicedExecutionOutcome {
        executions,
        device: Some(session.device_info().clone()),
        precision_mode: Some(precision),
        memory,
        phases,
        lowering,
    })
}

#[cfg(not(feature = "ascend"))]
fn execute_ascend(
    _bundle: &SlicedPlanBundle,
    _selected: Vec<&omeinsum::static_plan::StaticPlan>,
    _dtype: &str,
    _device_id: Option<i32>,
    _precision_mode: &str,
) -> Result<SlicedExecutionOutcome, String> {
    Err("Ascend backend unavailable; rebuild with --features ascend".to_string())
}

pub(crate) fn read_sliced_bundle(path: &str) -> Result<SlicedPlanBundle, String> {
    let json = std::fs::read_to_string(path)
        .map_err(|error| format!("Failed to read '{path}': {error}"))?;
    let bundle: SlicedPlanBundle = serde_json::from_str(&json)
        .map_err(|error| format!("Failed to parse sliced plan JSON: {error}"))?;
    bundle.validate().map_err(|error| error.to_string())?;
    Ok(bundle)
}

pub(crate) fn sliced_reference_complex64(
    bundle: &SlicedPlanBundle,
) -> Result<ComplexValue, String> {
    let assignments = gray_assignments(&bundle.slice).map_err(|error| error.to_string())?;
    let mut sum = ComplexValue { re: 0.0, im: 0.0 };
    for assignment in assignments {
        let inputs = slice_inputs(&bundle.source_inputs, &bundle.slice, &assignment)
            .map_err(|error| error.to_string())?;
        let value = contract_complex64(&bundle.reduced.realified_rank3, &inputs)
            .map_err(|error| error.to_string())?;
        sum.re += value.re;
        sum.im += value.im;
        if !sum.re.is_finite() || !sum.im.is_finite() {
            return Err("sliced Complex64 reference is non-finite".to_string());
        }
    }
    Ok(sum)
}
