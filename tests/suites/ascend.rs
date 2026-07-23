use omeinsum::backend::ascend::{
    AscendExecutable, AscendExecutableConfig, AscendExecutionMode, AscendPrecisionMode,
    AscendSession, AscendSessionConfig, CaptureStatus,
};
use omeinsum::static_plan::{
    build_plan_bundle, prepare_cpu_f32, BinaryContractionTree, ComplexNetwork, ComplexTensor,
    InputSet, PreparedExecutable, StaticPlan, TensorSpec,
};

fn two_leaf_network(left_imag: f64, right_imag: f64) -> ComplexNetwork<f64> {
    ComplexNetwork {
        tensors: vec![
            ComplexTensor {
                spec: TensorSpec {
                    modes: vec![0],
                    shape: vec![2],
                },
                real: vec![1.0, 2.0],
                imag: vec![left_imag, 0.0],
            },
            ComplexTensor {
                spec: TensorSpec {
                    modes: vec![0],
                    shape: vec![2],
                },
                real: vec![3.0, 4.0],
                imag: vec![right_imag, 0.0],
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

fn three_leaf_network() -> ComplexNetwork<f64> {
    ComplexNetwork {
        tensors: vec![
            ComplexTensor {
                spec: TensorSpec {
                    modes: vec![0, 1],
                    shape: vec![2, 2],
                },
                real: vec![1.0, 2.0, 3.0, 4.0],
                imag: vec![0.25, 0.125, 0.0, -0.25],
            },
            ComplexTensor {
                spec: TensorSpec {
                    modes: vec![1, 2],
                    shape: vec![2, 2],
                },
                real: vec![0.5, -1.0, 2.0, 3.0],
                imag: vec![-0.5, 0.0, 0.125, -0.5],
            },
            ComplexTensor {
                spec: TensorSpec {
                    modes: vec![0, 2],
                    shape: vec![2, 2],
                },
                real: vec![1.0, -0.5, 0.25, 2.0],
                imag: vec![0.0; 4],
            },
        ],
        output_modes: vec![],
        size_dict: vec![(0, 2), (1, 2), (2, 2)],
        tree: BinaryContractionTree::Node {
            output_modes: vec![],
            left: Box::new(BinaryContractionTree::Node {
                output_modes: vec![0, 2],
                left: Box::new(BinaryContractionTree::Leaf { tensor_index: 0 }),
                right: Box::new(BinaryContractionTree::Leaf { tensor_index: 1 }),
            }),
            right: Box::new(BinaryContractionTree::Leaf { tensor_index: 2 }),
        },
    }
}

fn compare_plan(session: &AscendSession, plan: &StaticPlan, inputs: &InputSet<f64>) {
    let mut cpu = prepare_cpu_f32(plan, inputs).unwrap();
    cpu.enqueue().unwrap();
    cpu.synchronize().unwrap();
    let expected = cpu.output().unwrap();

    let mut npu = AscendExecutable::prepare(
        session,
        plan,
        inputs,
        &AscendExecutableConfig {
            execution_mode: AscendExecutionMode::RepeatableAclnn,
        },
    )
    .unwrap();
    for iteration in 0..3 {
        npu.enqueue().unwrap();
        npu.synchronize().unwrap();
        let actual = npu.output().unwrap();
        let scale = 1.0f64.max(expected.re.abs()).max(expected.im.abs());
        let error =
            ((actual.re - expected.re).powi(2) + (actual.im - expected.im).powi(2)).sqrt() / scale;
        assert!(
            error <= 1e-3,
            "{:?} iteration {iteration}: expected {expected:?}, actual {actual:?}, \
             scaled error {error}",
            plan.representation
        );
    }
}

#[test]
#[ignore = "requires a live Ascend device and CANN runtime"]
fn ascend_static_plan_conformance_real_ride_merge_and_flat() {
    let device_id = std::env::var("OME_ASCEND_TEST_DEVICE_ID")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let session = AscendSession::new(&AscendSessionConfig {
        device_id,
        precision_mode: AscendPrecisionMode::KeepDtype,
    })
    .unwrap();

    let real = build_plan_bundle(&two_leaf_network(0.0, 0.0), 1e-12).unwrap();
    compare_plan(&session, &real.realified_rank3, &real.inputs);

    let ride_left = build_plan_bundle(&two_leaf_network(0.25, 0.0), 1e-12).unwrap();
    compare_plan(&session, &ride_left.realified_rank3, &ride_left.inputs);

    let ride_right = build_plan_bundle(&two_leaf_network(0.0, -0.5), 1e-12).unwrap();
    compare_plan(&session, &ride_right.realified_rank3, &ride_right.inputs);

    let merge = build_plan_bundle(&two_leaf_network(0.25, -0.5), 1e-12).unwrap();
    compare_plan(&session, &merge.realified_rank3, &merge.inputs);
    compare_plan(&session, &merge.flat_4m, &merge.inputs);
}

#[test]
#[ignore = "requires a live Ascend device and CANN runtime"]
fn ascend_repeatable_preserves_uploaded_inputs_across_enqueues() {
    let device_id = std::env::var("OME_ASCEND_TEST_DEVICE_ID")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let session = AscendSession::new(&AscendSessionConfig {
        device_id,
        precision_mode: AscendPrecisionMode::KeepDtype,
    })
    .unwrap();
    let bundle = build_plan_bundle(&three_leaf_network(), 1e-12).unwrap();
    compare_plan(&session, &bundle.real_skeleton, &bundle.inputs);
    compare_plan(&session, &bundle.flat_4m, &bundle.inputs);
    compare_plan(&session, &bundle.realified_rank3, &bundle.inputs);
}

#[test]
#[ignore = "requires a live Ascend device, CANN runtime, and OME_ASCEND_ENABLE_CAPTURE=1"]
fn ascend_captured_rank3_matches_repeatable() {
    let device_id = std::env::var("OME_ASCEND_TEST_DEVICE_ID")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let session = AscendSession::new(&AscendSessionConfig {
        device_id,
        precision_mode: AscendPrecisionMode::KeepDtype,
    })
    .unwrap();
    let bundle = build_plan_bundle(&three_leaf_network(), 1e-12).unwrap();

    let mut repeatable = AscendExecutable::prepare(
        &session,
        &bundle.realified_rank3,
        &bundle.inputs,
        &AscendExecutableConfig {
            execution_mode: AscendExecutionMode::RepeatableAclnn,
        },
    )
    .unwrap();
    let mut captured = AscendExecutable::prepare(
        &session,
        &bundle.realified_rank3,
        &bundle.inputs,
        &AscendExecutableConfig {
            execution_mode: AscendExecutionMode::CapturedModel,
        },
    )
    .unwrap();
    assert_eq!(captured.capture_status(), &CaptureStatus::Ready);

    for iteration in 0..3 {
        repeatable.enqueue().unwrap();
        repeatable.synchronize().unwrap();
        let expected = repeatable.output().unwrap();
        captured.enqueue().unwrap();
        captured.synchronize().unwrap();
        let actual = captured.output().unwrap();
        let scale = 1.0f64.max(expected.re.abs()).max(expected.im.abs());
        let error =
            ((actual.re - expected.re).powi(2) + (actual.im - expected.im).powi(2)).sqrt() / scale;
        assert!(
            error <= 1e-3,
            "captured rank-3 iteration {iteration} differs from repeatable: \
             expected {expected:?}, actual {actual:?}, scaled error {error}"
        );
    }
}
