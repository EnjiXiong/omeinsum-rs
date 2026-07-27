#![allow(clippy::result_large_err)] // Frozen backend diagnostics retain full context.

use std::time::Instant;

use omeinsum::static_plan::{
    benchmark_sequential_resident, BenchmarkConfig, ExecutionError, ModeTiming, PreparedExecutable,
    Representation, SequentialBenchmarkTarget, SliceSpec, SlicedExecutable, SlicedPlanBundle,
    SlicedVolume,
};
use serde::Serialize;

use crate::execute_plan::{prepare_cpu, select_plans, CpuDtype};
use crate::execute_sliced_plan::{read_sliced_bundle, sliced_reference_complex64};
use crate::sliced_reference_policy::SlicedReferencePolicy;

#[cfg(feature = "ascend")]
use crate::execute_plan::parse_ascend_precision;
#[cfg(feature = "ascend")]
use omeinsum::backend::ascend::{
    AscendDeviceInfo, AscendExecutable, AscendExecutableConfig, AscendExecutionMode,
    AscendMemoryStats, AscendPhaseTimings, AscendPrecisionMode, AscendSession, AscendSessionConfig,
    CaptureStatus,
};

const SLICED_PROTOCOL: &str = "complete-sliced-contraction";
const SLICED_RESIDENCY: &str = "sequential-single-representation";

#[derive(Serialize)]
struct SlicedBenchmarkReport {
    format: &'static str,
    protocol: &'static str,
    residency: &'static str,
    timed_region: &'static str,
    source_tree_hash: String,
    reduced_tree_hash: String,
    source_plan_hashes: Vec<(Representation, String)>,
    reduced_plan_hashes: Vec<(Representation, String)>,
    slicing: SliceSpec,
    aggregate_volumes: Vec<SlicedVolume>,
    reference_policy: &'static str,
    correctness_admission: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_complex64: Option<omeinsum::static_plan::ComplexValue>,
    timings: Vec<ModeTiming>,
    #[cfg(feature = "ascend")]
    device: Option<AscendDeviceInfo>,
    #[cfg(feature = "ascend")]
    precision_mode: Option<AscendPrecisionMode>,
    #[cfg(feature = "ascend")]
    capture: CaptureStatus,
    #[cfg(feature = "ascend")]
    session_context_seconds: Option<f64>,
    #[cfg(feature = "ascend")]
    target_diagnostics: Vec<SlicedTargetDiagnostics>,
    #[cfg(feature = "ascend")]
    capture_preflight: Option<SlicedCapturePreflight>,
    #[cfg(feature = "ascend")]
    lowering: Vec<(Representation, Vec<omeinsum::static_plan::LoweredNodeTrace>)>,
}

struct SlicedBenchmarkOutcome {
    timings: Vec<ModeTiming>,
    #[cfg(feature = "ascend")]
    device: Option<AscendDeviceInfo>,
    #[cfg(feature = "ascend")]
    precision_mode: Option<AscendPrecisionMode>,
    #[cfg(feature = "ascend")]
    capture: CaptureStatus,
    #[cfg(feature = "ascend")]
    session_context_seconds: Option<f64>,
    #[cfg(feature = "ascend")]
    target_diagnostics: Vec<SlicedTargetDiagnostics>,
    #[cfg(feature = "ascend")]
    capture_preflight: Option<SlicedCapturePreflight>,
    #[cfg(feature = "ascend")]
    lowering: Vec<(Representation, Vec<omeinsum::static_plan::LoweredNodeTrace>)>,
}

#[cfg(feature = "ascend")]
#[derive(Serialize)]
struct SlicedTargetDiagnostics {
    representation: Representation,
    execution_mode: String,
    preparations: usize,
    memory: AscendMemoryStats,
    phases: AscendPhaseTimings,
}

#[cfg(feature = "ascend")]
#[derive(Serialize)]
struct SlicedCapturePreflight {
    memory: AscendMemoryStats,
    phases: AscendPhaseTimings,
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
    capture_realified: &str,
    warmups: usize,
    samples: usize,
    min_sample_ms: u64,
    measurement_order_seed: u64,
    output: Option<&str>,
    pretty: Option<bool>,
) -> Result<(), String> {
    let bundle = read_sliced_bundle(plan_path)?;
    let selected = select_plans(&bundle.reduced, representations)?;
    let reference_policy = SlicedReferencePolicy::parse(reference_policy)?;
    let reference_complex64 =
        reference_policy.evaluate(backend, || sliced_reference_complex64(&bundle))?;
    let config = BenchmarkConfig {
        warmups,
        samples,
        min_sample_ms,
        measurement_order_seed,
    };
    let outcome = match backend {
        "cpu" => {
            if device_id.is_some() {
                return Err("--device-id is only valid with --backend ascend".to_string());
            }
            if precision_mode != "keep-dtype" {
                return Err(
                    "CPU sliced benchmarking accepts only --precision-mode keep-dtype".to_string(),
                );
            }
            if capture_realified != "off" {
                return Err(
                    "CPU sliced benchmarking accepts only --capture-realified off".to_string(),
                );
            }
            SlicedBenchmarkOutcome {
                timings: benchmark_cpu(&bundle, &selected, CpuDtype::parse(dtype)?, &config)?,
                #[cfg(feature = "ascend")]
                device: None,
                #[cfg(feature = "ascend")]
                precision_mode: None,
                #[cfg(feature = "ascend")]
                capture: CaptureStatus::NotRequested,
                #[cfg(feature = "ascend")]
                session_context_seconds: None,
                #[cfg(feature = "ascend")]
                target_diagnostics: vec![],
                #[cfg(feature = "ascend")]
                capture_preflight: None,
                #[cfg(feature = "ascend")]
                lowering: vec![],
            }
        }
        "ascend" => benchmark_ascend(
            &bundle,
            selected,
            dtype,
            device_id,
            precision_mode,
            capture_realified,
            &config,
        )?,
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
    let report = SlicedBenchmarkReport {
        format: "omeinsum-sliced-benchmark-report-v1",
        protocol: SLICED_PROTOCOL,
        residency: SLICED_RESIDENCY,
        timed_region: SLICED_PROTOCOL,
        source_tree_hash: bundle.source_tree_hash,
        reduced_tree_hash: bundle.reduced.tree_hash,
        source_plan_hashes: bundle.source_plan_hashes,
        reduced_plan_hashes,
        slicing: bundle.slice,
        aggregate_volumes: bundle.aggregate_volumes,
        reference_policy: reference_policy.label(),
        correctness_admission: reference_policy.correctness_admission(),
        reference_complex64,
        timings: outcome.timings,
        #[cfg(feature = "ascend")]
        device: outcome.device,
        #[cfg(feature = "ascend")]
        precision_mode: outcome.precision_mode,
        #[cfg(feature = "ascend")]
        capture: outcome.capture,
        #[cfg(feature = "ascend")]
        session_context_seconds: outcome.session_context_seconds,
        #[cfg(feature = "ascend")]
        target_diagnostics: outcome.target_diagnostics,
        #[cfg(feature = "ascend")]
        capture_preflight: outcome.capture_preflight,
        #[cfg(feature = "ascend")]
        lowering: outcome.lowering,
    };
    crate::common::write_json_output(&report, output, pretty)
}

fn benchmark_cpu(
    bundle: &SlicedPlanBundle,
    selected: &[&omeinsum::static_plan::StaticPlan],
    dtype: CpuDtype,
    config: &BenchmarkConfig,
) -> Result<Vec<ModeTiming>, String> {
    let targets = selected
        .iter()
        .map(|plan| SequentialBenchmarkTarget {
            representation: plan.representation.clone(),
            backend: "cpu",
            dtype: dtype.name(),
            execution_mode: "sliced-host-f64-accumulated",
        })
        .collect::<Vec<_>>();
    benchmark_sequential_resident(&targets, config, |target_index, inner_iterations| {
        let plan = selected[target_index];
        let inner =
            prepare_cpu(plan, &bundle.reduced, dtype).map_err(ExecutionError::Unsupported)?;
        let mut executable = SlicedExecutable::new(inner, &bundle.source_inputs, bundle)?;
        let started = Instant::now();
        for _ in 0..inner_iterations {
            executable.enqueue()?;
            executable.synchronize()?;
        }
        let elapsed = started.elapsed().as_secs_f64();
        let output = executable.output()?;
        Ok((elapsed, output))
    })
    .map_err(|error| error.to_string())
}

#[cfg(feature = "ascend")]
#[derive(Clone, Copy)]
enum SlicedTargetMode {
    Repeatable,
    AutoCaptured,
    RequiredCaptured,
}

#[cfg(feature = "ascend")]
impl SlicedTargetMode {
    fn execution_mode_label(self) -> &'static str {
        match self {
            Self::Repeatable => "sliced-host-f64-accumulated",
            Self::AutoCaptured | Self::RequiredCaptured => {
                "sliced-host-f64-accumulated-captured-model"
            }
        }
    }
}

#[cfg(feature = "ascend")]
struct SlicedTarget<'plan> {
    plan: &'plan omeinsum::static_plan::StaticPlan,
    mode: SlicedTargetMode,
}

#[cfg(feature = "ascend")]
#[derive(Default)]
struct SlicedTargetAccumulator {
    preparations: usize,
    memory: Option<AscendMemoryStats>,
    phases: AscendPhaseTimings,
}

#[cfg(feature = "ascend")]
#[allow(clippy::too_many_arguments)]
fn benchmark_ascend(
    bundle: &SlicedPlanBundle,
    selected: Vec<&omeinsum::static_plan::StaticPlan>,
    dtype: &str,
    device_id: Option<i32>,
    precision_mode: &str,
    capture_realified: &str,
    config: &BenchmarkConfig,
) -> Result<SlicedBenchmarkOutcome, String> {
    if dtype != "f32" {
        return Err("Ascend sliced benchmarking requires --dtype f32".to_string());
    }
    let capture_policy = SlicedCapturePolicy::parse(capture_realified)?;
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
    let mut targets = selected
        .iter()
        .copied()
        .map(|plan| SlicedTarget {
            plan,
            mode: SlicedTargetMode::Repeatable,
        })
        .collect::<Vec<_>>();
    let (capture, capture_preflight) =
        configure_sliced_capture_target(&session, bundle, &selected, capture_policy, &mut targets)?;
    let benchmark_targets = targets
        .iter()
        .map(|target| SequentialBenchmarkTarget {
            representation: target.plan.representation.clone(),
            backend: "ascend",
            dtype: "f32",
            execution_mode: target.mode.execution_mode_label(),
        })
        .collect::<Vec<_>>();
    let mut diagnostics = (0..targets.len())
        .map(|_| SlicedTargetAccumulator::default())
        .collect::<Vec<_>>();
    let timings = benchmark_sequential_resident(
        &benchmark_targets,
        config,
        |target_index, inner_iterations| {
            let target = &targets[target_index];
            let inner = prepare_sliced_ascend_target(&session, bundle, target)?;
            let mut executable = SlicedExecutable::new(inner, &bundle.source_inputs, bundle)?;
            let started = Instant::now();
            for _ in 0..inner_iterations {
                executable.enqueue()?;
                executable.synchronize()?;
            }
            let elapsed = started.elapsed().as_secs_f64();
            let output = executable.output()?;
            record_target_diagnostics(
                &mut diagnostics[target_index],
                executable.inner().memory_stats().clone(),
                executable.inner().phase_timings(),
            )?;
            Ok((elapsed, output))
        },
    )
    .map_err(|error| error.to_string())?;
    let target_diagnostics = targets
        .iter()
        .zip(diagnostics)
        .map(|(target, diagnostics)| {
            let memory = diagnostics.memory.ok_or_else(|| {
                format!(
                    "{:?} {} target was never prepared",
                    target.plan.representation,
                    target.mode.execution_mode_label()
                )
            })?;
            Ok(SlicedTargetDiagnostics {
                representation: target.plan.representation.clone(),
                execution_mode: target.mode.execution_mode_label().to_string(),
                preparations: diagnostics.preparations,
                memory,
                phases: diagnostics.phases,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(SlicedBenchmarkOutcome {
        timings,
        device: Some(session.device_info().clone()),
        precision_mode: Some(precision),
        capture,
        session_context_seconds: Some(session.context_create_seconds()),
        target_diagnostics,
        capture_preflight,
        lowering,
    })
}

#[cfg(feature = "ascend")]
fn configure_sliced_capture_target<'plan>(
    session: &AscendSession,
    bundle: &SlicedPlanBundle,
    selected: &[&'plan omeinsum::static_plan::StaticPlan],
    policy: SlicedCapturePolicy,
    targets: &mut Vec<SlicedTarget<'plan>>,
) -> Result<(CaptureStatus, Option<SlicedCapturePreflight>), String> {
    if policy == SlicedCapturePolicy::Off {
        return Ok((CaptureStatus::NotRequested, None));
    }
    let plan = selected
        .iter()
        .copied()
        .find(|plan| plan.representation == Representation::RealifiedRank3)
        .ok_or_else(|| {
            "--capture-realified requires realified-rank3 in --representations".to_string()
        })?;
    let candidate = match policy {
        SlicedCapturePolicy::Off => unreachable!("handled above"),
        SlicedCapturePolicy::Auto => {
            AscendExecutable::prepare_auto_capture(session, plan, &bundle.reduced.inputs)
        }
        SlicedCapturePolicy::Required => AscendExecutable::prepare(
            session,
            plan,
            &bundle.reduced.inputs,
            &AscendExecutableConfig {
                execution_mode: AscendExecutionMode::CapturedModel,
            },
        ),
    }
    .map_err(|error| error.to_string())?;
    let status = candidate.capture_status().clone();
    let preflight = SlicedCapturePreflight {
        memory: candidate.memory_stats().clone(),
        phases: candidate.phase_timings(),
    };
    match (&status, policy) {
        (CaptureStatus::Ready, SlicedCapturePolicy::Auto) => {
            targets.push(SlicedTarget {
                plan,
                mode: SlicedTargetMode::AutoCaptured,
            });
        }
        (CaptureStatus::Ready, SlicedCapturePolicy::Required) => {
            targets.push(SlicedTarget {
                plan,
                mode: SlicedTargetMode::RequiredCaptured,
            });
        }
        (CaptureStatus::SkippedUnsupported { .. }, SlicedCapturePolicy::Auto) => {}
        (CaptureStatus::SkippedUnsupported { reason }, SlicedCapturePolicy::Required) => {
            return Err(format!("required sliced capture is unavailable: {reason}"));
        }
        (CaptureStatus::NotRequested, _) => {
            return Err("capture preflight did not attempt capture".to_string());
        }
        (_, SlicedCapturePolicy::Off) => unreachable!("handled above"),
    }
    Ok((status, Some(preflight)))
}

#[cfg(feature = "ascend")]
fn prepare_sliced_ascend_target<'session>(
    session: &'session AscendSession,
    bundle: &SlicedPlanBundle,
    target: &SlicedTarget<'_>,
) -> Result<AscendExecutable<'session>, ExecutionError> {
    let executable = match target.mode {
        SlicedTargetMode::Repeatable => AscendExecutable::prepare(
            session,
            target.plan,
            &bundle.reduced.inputs,
            &AscendExecutableConfig {
                execution_mode: AscendExecutionMode::RepeatableAclnn,
            },
        )?,
        SlicedTargetMode::AutoCaptured => {
            AscendExecutable::prepare_auto_capture(session, target.plan, &bundle.reduced.inputs)?
        }
        SlicedTargetMode::RequiredCaptured => AscendExecutable::prepare(
            session,
            target.plan,
            &bundle.reduced.inputs,
            &AscendExecutableConfig {
                execution_mode: AscendExecutionMode::CapturedModel,
            },
        )?,
    };
    if !matches!(target.mode, SlicedTargetMode::Repeatable)
        && executable.capture_status() != &CaptureStatus::Ready
    {
        return Err(ExecutionError::Unsupported(
            "sliced capture readiness changed after preflight".to_string(),
        ));
    }
    Ok(executable)
}

#[cfg(feature = "ascend")]
fn record_target_diagnostics(
    aggregate: &mut SlicedTargetAccumulator,
    memory: AscendMemoryStats,
    mut phases: AscendPhaseTimings,
) -> Result<(), ExecutionError> {
    if let Some(previous) = &aggregate.memory {
        if previous != &memory {
            return Err(ExecutionError::InvalidPlan(
                "sequential target device memory changed across preparations".to_string(),
            ));
        }
    } else {
        aggregate.memory = Some(memory);
    }
    aggregate.preparations += 1;
    phases.context_create_seconds = 0.0;
    aggregate.phases.plan_lower_seconds += phases.plan_lower_seconds;
    aggregate.phases.allocation_seconds += phases.allocation_seconds;
    aggregate.phases.descriptor_executor_prepare_seconds +=
        phases.descriptor_executor_prepare_seconds;
    aggregate.phases.h2d_seconds += phases.h2d_seconds;
    aggregate.phases.warmup_seconds += phases.warmup_seconds;
    aggregate.phases.d2h_seconds += phases.d2h_seconds;
    Ok(())
}

#[cfg(feature = "ascend")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum SlicedCapturePolicy {
    Off,
    Auto,
    Required,
}

#[cfg(feature = "ascend")]
impl SlicedCapturePolicy {
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
    _bundle: &SlicedPlanBundle,
    _selected: Vec<&omeinsum::static_plan::StaticPlan>,
    _dtype: &str,
    _device_id: Option<i32>,
    _precision_mode: &str,
    _capture_realified: &str,
    _config: &BenchmarkConfig,
) -> Result<SlicedBenchmarkOutcome, String> {
    Err("Ascend backend unavailable; rebuild with --features ascend".to_string())
}
