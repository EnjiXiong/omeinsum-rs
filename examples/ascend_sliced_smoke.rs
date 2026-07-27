use omeinsum::backend::ascend::{
    AscendExecutable, AscendExecutableConfig, AscendExecutionMode, AscendMemoryStats,
    AscendPhaseTimings, AscendPrecisionMode, AscendSession, AscendSessionConfig,
};
use omeinsum::static_plan::{
    build_plan_bundle_with_preprocessing, build_sliced_plan_bundle, contract_complex64,
    gray_assignments, slice_inputs, BinaryContractionTree, ComplexNetwork, ComplexTensor,
    ComplexValue, LeafPreprocessing, PreparedExecutable, Representation, SlicedExecutable,
    SlicedPlanBundle, StaticPlan, TensorSpec,
};
use serde::Serialize;

const REPEAT_COUNT: usize = 2;
const SCALED_ERROR_LIMIT: f64 = 1e-3;

#[derive(Serialize)]
struct SmokeExecution {
    representation: Representation,
    execution_mode: AscendExecutionMode,
    execution_mode_label: &'static str,
    outputs: Vec<ComplexValue>,
    max_scaled_error: f64,
    memory: AscendMemoryStats,
    phases: AscendPhaseTimings,
}

#[derive(Serialize)]
struct SmokeReport {
    format: &'static str,
    status: &'static str,
    slice_count: usize,
    repeat_count: usize,
    assignment_order: &'static str,
    slice_modes: Vec<i32>,
    source_tree_hash: String,
    reduced_tree_hash: String,
    device_id: i32,
    soc_name: String,
    precision_mode: AscendPrecisionMode,
    reference_complex64: ComplexValue,
    executions: Vec<SmokeExecution>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let device_id = parse_device_id()?;
    let source = build_plan_bundle_with_preprocessing(
        &phase_canonicalized_network(),
        1e-12,
        LeafPreprocessing::PhaseCanonicalized,
    )?;
    let sliced = build_sliced_plan_bundle(&source, &[0])?;
    if sliced.slice.slice_count != 2 {
        return Err(format!(
            "sliced smoke expected two assignments, got {}",
            sliced.slice.slice_count
        )
        .into());
    }
    let reference_complex64 = sliced_reference(&sliced)?;
    let unsliced_reference = contract_complex64(&source.flat_4m, &source.inputs)?;
    if scaled_error(reference_complex64, unsliced_reference) > 1e-12 {
        return Err("sliced Complex64 reference differs from the unsliced scalar".into());
    }

    let session = AscendSession::new(&AscendSessionConfig {
        device_id,
        precision_mode: AscendPrecisionMode::KeepDtype,
    })?;
    let flat = run_representation(
        &session,
        &sliced.reduced.flat_4m,
        &sliced,
        reference_complex64,
    )?;
    let rank3 = run_representation(
        &session,
        &sliced.reduced.realified_rank3,
        &sliced,
        reference_complex64,
    )?;
    for (flat_output, rank3_output) in flat.outputs.iter().zip(&rank3.outputs) {
        if scaled_error(*flat_output, *rank3_output) > SCALED_ERROR_LIMIT {
            return Err("Flat-4M and rank-3 sliced outputs disagree".into());
        }
    }

    println!(
        "{}",
        serde_json::to_string(&SmokeReport {
            format: "omeinsum-ascend-sliced-smoke-v1",
            status: "passed",
            slice_count: sliced.slice.slice_count,
            repeat_count: REPEAT_COUNT,
            assignment_order: "binary-reflected-gray",
            slice_modes: sliced.slice.modes.clone(),
            source_tree_hash: sliced.source_tree_hash.clone(),
            reduced_tree_hash: sliced.reduced.tree_hash.clone(),
            device_id,
            soc_name: session.device_info().soc_name.clone(),
            precision_mode: AscendPrecisionMode::KeepDtype,
            reference_complex64,
            executions: vec![flat, rank3],
        })?
    );
    Ok(())
}

fn parse_device_id() -> Result<i32, Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let mut device_id = None;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--device-id" => {
                let value = arguments.next().ok_or("--device-id requires a value")?;
                device_id = Some(value.parse()?);
            }
            _ => return Err(format!("unrecognized argument {argument:?}").into()),
        }
    }
    device_id.ok_or_else(|| "--device-id is required".into())
}

fn run_representation(
    session: &AscendSession,
    plan: &StaticPlan,
    sliced: &SlicedPlanBundle,
    reference: ComplexValue,
) -> Result<SmokeExecution, Box<dyn std::error::Error>> {
    let inner = AscendExecutable::prepare(
        session,
        plan,
        &sliced.reduced.inputs,
        &AscendExecutableConfig {
            execution_mode: AscendExecutionMode::RepeatableAclnn,
        },
    )?;
    let mut executable = SlicedExecutable::new(inner, &sliced.source_inputs, sliced)?;
    let mut outputs = Vec::with_capacity(REPEAT_COUNT);
    let mut max_scaled_error = 0.0_f64;
    for _ in 0..REPEAT_COUNT {
        executable.enqueue()?;
        executable.synchronize()?;
        let output = executable.output()?;
        let error = scaled_error(output, reference);
        if error > SCALED_ERROR_LIMIT {
            return Err(format!(
                "{:?} sliced output {output:?} differs from {reference:?} by {error}",
                plan.representation
            )
            .into());
        }
        max_scaled_error = max_scaled_error.max(error);
        outputs.push(output);
    }
    let inner = executable.inner();
    if inner.memory_stats().peak_device_bytes == 0 {
        return Err(format!("{:?} reported zero device memory", plan.representation).into());
    }
    Ok(SmokeExecution {
        representation: plan.representation.clone(),
        execution_mode: inner.execution_mode(),
        execution_mode_label: executable.execution_mode_label(),
        outputs,
        max_scaled_error,
        memory: inner.memory_stats().clone(),
        phases: inner.phase_timings(),
    })
}

fn sliced_reference(sliced: &SlicedPlanBundle) -> Result<ComplexValue, Box<dyn std::error::Error>> {
    let mut sum = ComplexValue { re: 0.0, im: 0.0 };
    for assignment in gray_assignments(&sliced.slice)? {
        let inputs = slice_inputs(&sliced.source_inputs, &sliced.slice, &assignment)?;
        let value = contract_complex64(&sliced.reduced.flat_4m, &inputs)?;
        sum.re += value.re;
        sum.im += value.im;
    }
    Ok(sum)
}

fn scaled_error(observed: ComplexValue, expected: ComplexValue) -> f64 {
    let scale = 1.0_f64.max(expected.re.abs()).max(expected.im.abs());
    ((observed.re - expected.re).powi(2) + (observed.im - expected.im).powi(2)).sqrt() / scale
}

fn phase_canonicalized_network() -> ComplexNetwork<f64> {
    let left_phase = num_complex::Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    let left = [left_phase, -left_phase];
    let right = [
        num_complex::Complex64::new(0.5, 0.75),
        num_complex::Complex64::new(-0.25, 0.125),
    ];
    ComplexNetwork {
        tensors: vec![
            ComplexTensor {
                spec: TensorSpec {
                    modes: vec![0],
                    shape: vec![2],
                },
                real: left.iter().map(|value| value.re).collect(),
                imag: left.iter().map(|value| value.im).collect(),
            },
            ComplexTensor {
                spec: TensorSpec {
                    modes: vec![0],
                    shape: vec![2],
                },
                real: right.iter().map(|value| value.re).collect(),
                imag: right.iter().map(|value| value.im).collect(),
            },
        ],
        output_modes: vec![],
        size_dict: vec![(0, 2)],
        tree: BinaryContractionTree::Node {
            output_modes: vec![],
            left: Box::new(BinaryContractionTree::Leaf { tensor_index: 0 }),
            right: Box::new(BinaryContractionTree::Leaf { tensor_index: 1 }),
        },
    }
}
