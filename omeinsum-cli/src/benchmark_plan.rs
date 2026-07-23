use omeinsum::static_plan::{
    benchmark_prepared, contract_complex64, BenchmarkConfig, BenchmarkTarget, ComplexValue,
    ModeTiming, PreparedExecutable,
};
use serde::{Deserialize, Serialize};

use crate::execute_plan::{prepare_cpu, read_bundle, select_plans, CpuDtype};

#[cfg(feature = "ascend")]
use crate::execute_plan::parse_ascend_precision;
#[cfg(feature = "ascend")]
use omeinsum::backend::ascend::{
    AscendDeviceInfo, AscendExecutable, AscendExecutableConfig, AscendExecutionMode,
    AscendMemoryStats, AscendPhaseTimings, AscendPrecisionMode, AscendSession, AscendSessionConfig,
    CaptureStatus,
};
#[cfg(feature = "ascend")]
use omeinsum::static_plan::benchmark_prepared_with_diagnostics;

#[derive(Serialize, Deserialize)]
struct BenchmarkPlanReport {
    format: String,
    tree_hash: String,
    reference_complex64: ComplexValue,
    timings: Vec<ModeTiming>,
    #[cfg(feature = "ascend")]
    device: Option<AscendDeviceInfo>,
    #[cfg(feature = "ascend")]
    memory: Vec<(omeinsum::static_plan::Representation, AscendMemoryStats)>,
    #[cfg(feature = "ascend")]
    capture: CaptureStatus,
    #[cfg(feature = "ascend")]
    precision_mode: Option<AscendPrecisionMode>,
    #[cfg(feature = "ascend")]
    phases: Option<AscendPhaseTimings>,
    #[cfg(feature = "ascend")]
    lowering: Vec<(
        omeinsum::static_plan::Representation,
        Vec<omeinsum::static_plan::LoweredNodeTrace>,
    )>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    plan_path: &str,
    backend: &str,
    representations: &str,
    dtype: &str,
    device_id: Option<i32>,
    precision_mode: &str,
    capture_realified: &str,
    warmups: usize,
    samples: usize,
    min_sample_ms: u64,
    measurement_order_seed: u64,
    output: Option<&str>,
    pretty: Option<bool>,
) -> Result<(), String> {
    let bundle = read_bundle(plan_path)?;
    let selected = select_plans(&bundle, representations)?;
    let reference_complex64 = contract_complex64(&bundle.realified_rank3, &bundle.inputs)
        .map_err(|error| error.to_string())?;
    let config = BenchmarkConfig {
        warmups,
        samples,
        min_sample_ms,
        measurement_order_seed,
    };

    let report = match backend {
        "cpu" => {
            let dtype = CpuDtype::parse(dtype)?;
            let mut executables = selected
                .into_iter()
                .map(|plan| prepare_cpu(plan, &bundle, dtype))
                .collect::<Result<Vec<Box<dyn PreparedExecutable>>, _>>()?;
            let mut targets = executables
                .iter_mut()
                .map(|executable| BenchmarkTarget {
                    backend: "cpu",
                    dtype: dtype.name(),
                    execution_mode: "prepared",
                    executable: executable.as_mut(),
                })
                .collect::<Vec<_>>();
            let timings =
                benchmark_prepared(&mut targets, &config).map_err(|error| error.to_string())?;
            BenchmarkPlanReport {
                format: "omeinsum-benchmark-report-v1".to_string(),
                tree_hash: bundle.tree_hash,
                reference_complex64,
                timings,
                #[cfg(feature = "ascend")]
                device: None,
                #[cfg(feature = "ascend")]
                memory: vec![],
                #[cfg(feature = "ascend")]
                capture: CaptureStatus::NotRequested,
                #[cfg(feature = "ascend")]
                precision_mode: None,
                #[cfg(feature = "ascend")]
                phases: None,
                #[cfg(feature = "ascend")]
                lowering: vec![],
            }
        }
        "ascend" => benchmark_ascend(
            &bundle,
            selected,
            reference_complex64,
            dtype,
            device_id,
            precision_mode,
            capture_realified,
            &config,
        )?,
        _ => {
            return Err(format!(
                "unsupported backend {backend:?}; expected cpu or ascend"
            ))
        }
    };
    crate::common::write_json_output(&report, output, pretty)
}

#[cfg(feature = "ascend")]
#[allow(clippy::too_many_arguments)]
fn benchmark_ascend(
    bundle: &omeinsum::static_plan::PlanBundle,
    selected: Vec<&omeinsum::static_plan::StaticPlan>,
    reference_complex64: ComplexValue,
    dtype: &str,
    device_id: Option<i32>,
    precision_mode: &str,
    capture_realified: &str,
    config: &BenchmarkConfig,
) -> Result<BenchmarkPlanReport, String> {
    if dtype != "f32" {
        return Err("Ascend benchmarking requires --dtype f32".to_string());
    }
    let capture_policy = RealifiedCapturePolicy::parse(capture_realified)?;
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
    let mut executables = selected
        .into_iter()
        .map(|plan| {
            if plan.representation == omeinsum::static_plan::Representation::RealifiedRank3 {
                match capture_policy {
                    RealifiedCapturePolicy::Off => AscendExecutable::prepare(
                        &session,
                        plan,
                        &bundle.inputs,
                        &AscendExecutableConfig {
                            execution_mode: AscendExecutionMode::RepeatableAclnn,
                        },
                    ),
                    RealifiedCapturePolicy::Auto => {
                        AscendExecutable::prepare_auto_capture(&session, plan, &bundle.inputs)
                    }
                    RealifiedCapturePolicy::Required => AscendExecutable::prepare(
                        &session,
                        plan,
                        &bundle.inputs,
                        &AscendExecutableConfig {
                            execution_mode: AscendExecutionMode::CapturedModel,
                        },
                    ),
                }
            } else {
                AscendExecutable::prepare(
                    &session,
                    plan,
                    &bundle.inputs,
                    &AscendExecutableConfig {
                        execution_mode: AscendExecutionMode::RepeatableAclnn,
                    },
                )
            }
            .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let capture = executables
        .iter()
        .find(|executable| {
            executable.representation() == omeinsum::static_plan::Representation::RealifiedRank3
        })
        .map(|executable| executable.capture_status().clone())
        .unwrap_or(CaptureStatus::NotRequested);
    let memory = executables
        .iter()
        .map(|executable| {
            (
                executable.representation(),
                executable.memory_stats().clone(),
            )
        })
        .collect::<Vec<_>>();
    let execution_modes = executables
        .iter()
        .map(|executable| match executable.execution_mode() {
            AscendExecutionMode::RepeatableAclnn => "repeatable-aclnn",
            AscendExecutionMode::CapturedModel => "captured-model",
        })
        .collect::<Vec<_>>();
    let mut targets = executables
        .iter_mut()
        .zip(&execution_modes)
        .map(|(executable, execution_mode)| BenchmarkTarget {
            backend: "ascend",
            dtype: "f32",
            execution_mode,
            executable,
        })
        .collect::<Vec<_>>();
    let (timings, diagnostics) = benchmark_prepared_with_diagnostics(&mut targets, config)
        .map_err(|error| error.to_string())?;
    drop(targets);

    let mut phases = AscendPhaseTimings {
        context_create_seconds: session.context_create_seconds(),
        warmup_seconds: diagnostics.warmup_seconds,
        ..AscendPhaseTimings::default()
    };
    for executable in &executables {
        let timing = executable.phase_timings();
        phases.plan_lower_seconds += timing.plan_lower_seconds;
        phases.allocation_seconds += timing.allocation_seconds;
        phases.descriptor_executor_prepare_seconds += timing.descriptor_executor_prepare_seconds;
        phases.h2d_seconds += timing.h2d_seconds;
        phases.d2h_seconds += timing.d2h_seconds;
    }
    Ok(BenchmarkPlanReport {
        format: "omeinsum-benchmark-report-v1".to_string(),
        tree_hash: bundle.tree_hash.clone(),
        reference_complex64,
        timings,
        device: Some(session.device_info().clone()),
        memory,
        capture,
        precision_mode: Some(precision),
        phases: Some(phases),
        lowering,
    })
}

#[cfg(feature = "ascend")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RealifiedCapturePolicy {
    Off,
    Auto,
    Required,
}

#[cfg(feature = "ascend")]
impl RealifiedCapturePolicy {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "off" => Ok(Self::Off),
            "auto" => Ok(Self::Auto),
            "required" => Ok(Self::Required),
            _ => Err(format!(
                "unsupported capture policy {value:?}; expected off, auto, or required"
            )),
        }
    }
}

#[cfg(not(feature = "ascend"))]
#[allow(clippy::too_many_arguments)]
fn benchmark_ascend(
    _bundle: &omeinsum::static_plan::PlanBundle,
    _selected: Vec<&omeinsum::static_plan::StaticPlan>,
    _reference_complex64: ComplexValue,
    _dtype: &str,
    _device_id: Option<i32>,
    _precision_mode: &str,
    _capture_realified: &str,
    _config: &BenchmarkConfig,
) -> Result<BenchmarkPlanReport, String> {
    Err("Ascend backend unavailable; rebuild with --features ascend".to_string())
}
