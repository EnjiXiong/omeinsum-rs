#![allow(clippy::result_large_err)] // Frozen execution diagnostics retain full context.

use omeinsum::static_plan::{
    allocate_live_ranges, benchmark_prepared, benchmark_sequential_resident, build_geometry_plan,
    build_plan_bundle, build_plan_bundle_with_preprocessing, build_sliced_plan_bundle,
    coalesce_permutation, contract_complex64, decompose_permutation, gray_assignments,
    green_operand_plane_batches, lower_plan_traces, plan_f32_arena, prepare_cpu_f32,
    prepare_cpu_f64, slice_inputs, ArenaSlot, BenchmarkConfig, BenchmarkTarget,
    BinaryContractionTree, ComplexNetwork, ComplexTensor, ComplexValue, ExecutionError, InputSet,
    InputTensor, InputUpdate, KernelKind, LeafClass, LeafPreprocessing, LiveRange, PlanBundle,
    PlanError, PlanStats, Plane, PreparedExecutable, Representation, ScratchRole,
    SequentialBenchmarkTarget, SliceAssignmentOrder, SliceSpec, SlicedExecutable, StaticPlan,
    TensorSpec, ValueId, ValueSpec,
};
use rand::{Rng, SeedableRng};
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn one_leaf_plan(
    representation: Representation,
    plan_hash: &str,
    planes: Vec<Plane>,
) -> StaticPlan {
    StaticPlan {
        representation,
        tree_hash: "tree".to_string(),
        plan_hash: plan_hash.to_string(),
        leaf_values: vec![ValueId(0)],
        values: vec![ValueSpec {
            id: ValueId(0),
            tensor: TensorSpec {
                modes: vec![],
                shape: vec![],
            },
            planes,
        }],
        nodes: vec![],
        output: ValueId(0),
        stats: PlanStats {
            real_leaf_count: 1,
            complex_leaf_count: 0,
            real_real_nodes: 0,
            ride_left_nodes: 0,
            ride_right_nodes: 0,
            merge_3m_nodes: 0,
            flat_4m_nodes: 0,
            real_skeleton_volume: 0,
            real_matmul_volume: 0,
            realification_cost: None,
        },
    }
}

fn canonical_geometry_network(size_dict: Vec<(i32, usize)>) -> ComplexNetwork<f64> {
    ComplexNetwork {
        tensors: vec![
            ComplexTensor {
                spec: TensorSpec {
                    modes: vec![0, 1],
                    shape: vec![2, 3],
                },
                real: vec![0.0; 6],
                imag: vec![0.0; 6],
            },
            ComplexTensor {
                spec: TensorSpec {
                    modes: vec![1, 2],
                    shape: vec![3, 4],
                },
                real: vec![0.0; 12],
                imag: vec![0.0; 12],
            },
            ComplexTensor {
                spec: TensorSpec {
                    modes: vec![0, 2],
                    shape: vec![2, 4],
                },
                real: vec![0.0; 8],
                imag: vec![0.0; 8],
            },
        ],
        output_modes: vec![],
        size_dict,
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

fn two_leaf_scalar_network(left_imag: f64, right_imag: f64) -> ComplexNetwork<f64> {
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

fn phase_real_two_leaf_network(
    left_phase: f64,
    right_phase: f64,
    right_is_genuinely_complex: bool,
) -> ComplexNetwork<f64> {
    let left_phase = num_complex::Complex64::from_polar(1.0, left_phase);
    let right_phase = num_complex::Complex64::from_polar(1.0, right_phase);
    let left_real = [1.0, -1.0];
    let right_base = if right_is_genuinely_complex {
        [
            num_complex::Complex64::new(0.5, 0.75),
            num_complex::Complex64::new(-0.25, 0.125),
        ]
    } else {
        [
            num_complex::Complex64::new(3.0, 0.0),
            num_complex::Complex64::new(4.0, 0.0),
        ]
    };
    let left = left_real.map(|value| left_phase * value);
    let right = right_base.map(|value| right_phase * value);
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

fn matrix_scalar_network(left_imag: f64, right_imag: f64) -> ComplexNetwork<f64> {
    ComplexNetwork {
        tensors: vec![
            ComplexTensor {
                spec: TensorSpec {
                    modes: vec![0, 1],
                    shape: vec![2, 2],
                },
                real: vec![1.0, 2.0, 3.0, 4.0],
                imag: vec![left_imag, 0.5 * left_imag, 0.0, -left_imag],
            },
            ComplexTensor {
                spec: TensorSpec {
                    modes: vec![1, 2],
                    shape: vec![2, 2],
                },
                real: vec![0.5, -1.0, 2.0, 3.0],
                imag: vec![right_imag, 0.0, -0.25 * right_imag, right_imag],
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

fn vector_chain_tree(leaves: usize) -> BinaryContractionTree {
    assert!(leaves >= 2);
    let first_output = if leaves == 2 { vec![] } else { vec![0] };
    let mut tree = BinaryContractionTree::Node {
        output_modes: first_output,
        left: Box::new(BinaryContractionTree::Leaf { tensor_index: 0 }),
        right: Box::new(BinaryContractionTree::Leaf { tensor_index: 1 }),
    };
    for tensor_index in 2..leaves {
        tree = BinaryContractionTree::Node {
            output_modes: if tensor_index + 1 == leaves {
                vec![]
            } else {
                vec![0]
            },
            left: Box::new(tree),
            right: Box::new(BinaryContractionTree::Leaf { tensor_index }),
        };
    }
    tree
}

struct FakeExecutable {
    id: usize,
    delay: Duration,
    value: ComplexValue,
    enqueues_since_sync: usize,
    batches: Vec<usize>,
    order: Arc<Mutex<Vec<usize>>>,
}

impl FakeExecutable {
    fn new(id: usize, delay: Duration, value: ComplexValue, order: Arc<Mutex<Vec<usize>>>) -> Self {
        Self {
            id,
            delay,
            value,
            enqueues_since_sync: 0,
            batches: vec![],
            order,
        }
    }
}

impl PreparedExecutable for FakeExecutable {
    fn representation(&self) -> Representation {
        Representation::RealSkeleton
    }

    fn update_inputs(&mut self, _updates: &[InputUpdate<f64>]) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn enqueue(&mut self) -> Result<(), ExecutionError> {
        self.enqueues_since_sync += 1;
        self.order.lock().unwrap().push(self.id);
        if !self.delay.is_zero() {
            std::thread::sleep(self.delay);
        }
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        self.batches.push(self.enqueues_since_sync);
        self.enqueues_since_sync = 0;
        Ok(())
    }

    fn output(&mut self) -> Result<ComplexValue, ExecutionError> {
        Ok(self.value)
    }
}

#[derive(Default)]
struct SlicedFakeState {
    updates: Vec<Vec<InputUpdate<f64>>>,
    enqueue_count: usize,
    sync_count: usize,
}

struct SlicedFakeExecutable {
    state: Arc<Mutex<SlicedFakeState>>,
    outputs: Vec<ComplexValue>,
}

impl PreparedExecutable for SlicedFakeExecutable {
    fn representation(&self) -> Representation {
        Representation::RealifiedRank3
    }

    fn update_inputs(&mut self, updates: &[InputUpdate<f64>]) -> Result<(), ExecutionError> {
        self.state.lock().unwrap().updates.push(updates.to_vec());
        Ok(())
    }

    fn enqueue(&mut self) -> Result<(), ExecutionError> {
        let mut state = self.state.lock().unwrap();
        if state.enqueue_count >= self.outputs.len() {
            return Err(ExecutionError::InvalidPlan(
                "sliced fake received too many enqueues".to_string(),
            ));
        }
        state.enqueue_count += 1;
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        self.state.lock().unwrap().sync_count += 1;
        Ok(())
    }

    fn output(&mut self) -> Result<ComplexValue, ExecutionError> {
        let enqueue_count = self.state.lock().unwrap().enqueue_count;
        self.outputs
            .get(enqueue_count.saturating_sub(1))
            .copied()
            .ok_or_else(|| {
                ExecutionError::InvalidPlan(
                    "sliced fake output requested before enqueue".to_string(),
                )
            })
    }
}

#[test]
fn plan_bundle_round_trips_json() {
    let bundle = PlanBundle {
        format: "omeinsum-static-plan-v1".to_string(),
        realness_tol: 1e-12,
        leaf_preprocessing: LeafPreprocessing::Raw,
        phase_canonicalization: None,
        tree_hash: "tree".to_string(),
        inputs: InputSet {
            tensors: vec![InputTensor {
                spec: TensorSpec {
                    modes: vec![],
                    shape: vec![],
                },
                real: vec![1.0],
                imag: vec![0.0],
                class: LeafClass::Real,
                imag_max: 0.0,
                classification_imag_max: None,
            }],
        },
        real_skeleton: one_leaf_plan(Representation::RealSkeleton, "real", vec![Plane::Real]),
        flat_4m: one_leaf_plan(
            Representation::Flat4M,
            "flat",
            vec![Plane::Real, Plane::Imag],
        ),
        realified_rank3: one_leaf_plan(Representation::RealifiedRank3, "rank3", vec![Plane::Real]),
    };

    let json = serde_json::to_string(&bundle).unwrap();
    assert!(json.contains("\"real-skeleton\""));
    assert!(json.contains("\"flat-4m\""));
    assert!(json.contains("\"realified-rank3\""));

    let round_trip: PlanBundle = serde_json::from_str(&json).unwrap();
    assert_eq!(round_trip, bundle);
    assert_eq!(round_trip.format, "omeinsum-static-plan-v1");
}

#[test]
fn sliced_classification_provenance_is_optional_and_freezes_source_class() {
    let mut bundle = build_plan_bundle(&two_leaf_scalar_network(0.5, 0.0), 1e-12).unwrap();
    let source_imag_max = bundle.inputs.tensors[0].imag_max;

    let ordinary_json = serde_json::to_value(&bundle).unwrap();
    assert!(ordinary_json["inputs"]["tensors"][0]
        .get("classification_imag_max")
        .is_none());

    let sliced = &mut bundle.inputs.tensors[0];
    sliced.imag.fill(0.0);
    sliced.imag_max = 0.0;
    sliced.classification_imag_max = Some(source_imag_max);

    assert_eq!(sliced.class, LeafClass::Complex);
    assert_eq!(sliced.imag_max, 0.0);
    assert_eq!(sliced.classification_imag_max, Some(0.5));
    bundle.validate().unwrap();
    let sliced_json = serde_json::to_value(&bundle).unwrap();
    assert_eq!(
        sliced_json["inputs"]["tensors"][0]["classification_imag_max"],
        0.5
    );
}

#[test]
fn sliced_gray_schedule_has_literal_binary_reflected_order() {
    let spec = SliceSpec {
        modes: vec![7, 9],
        dimensions: vec![2, 2],
        assignment_order: SliceAssignmentOrder::BinaryReflectedGray,
        slice_count: 4,
    };

    assert_eq!(
        gray_assignments(&spec).unwrap(),
        vec![vec![0, 0], vec![0, 1], vec![1, 1], vec![1, 0]]
    );
}

#[test]
fn sliced_inputs_select_column_major_values_and_preserve_source_classes() {
    let inputs = InputSet {
        tensors: vec![
            InputTensor {
                spec: TensorSpec {
                    modes: vec![1, 2, 3],
                    shape: vec![2, 3, 2],
                },
                real: (0..12).map(f64::from).collect(),
                imag: (100..112).map(f64::from).collect(),
                class: LeafClass::Complex,
                imag_max: 111.0,
                classification_imag_max: None,
            },
            InputTensor {
                spec: TensorSpec {
                    modes: vec![2],
                    shape: vec![3],
                },
                real: vec![20.0, 21.0, 22.0],
                imag: vec![0.0, 0.0, 0.0],
                class: LeafClass::Real,
                imag_max: 0.0,
                classification_imag_max: None,
            },
            InputTensor {
                spec: TensorSpec {
                    modes: vec![1],
                    shape: vec![2],
                },
                real: vec![30.0, 31.0],
                imag: vec![0.0, 0.0],
                class: LeafClass::Real,
                imag_max: 0.0,
                classification_imag_max: None,
            },
        ],
    };
    let spec = SliceSpec {
        modes: vec![1, 3],
        dimensions: vec![2, 2],
        assignment_order: SliceAssignmentOrder::BinaryReflectedGray,
        slice_count: 4,
    };

    let sliced = slice_inputs(&inputs, &spec, &[1, 0]).unwrap();

    assert_eq!(sliced.tensors[0].spec.modes, vec![1, 2, 3]);
    assert_eq!(sliced.tensors[0].spec.shape, vec![1, 3, 1]);
    assert_eq!(sliced.tensors[0].real, vec![1.0, 3.0, 5.0]);
    assert_eq!(sliced.tensors[0].imag, vec![101.0, 103.0, 105.0]);
    assert_eq!(sliced.tensors[0].imag_max, 105.0);
    assert_eq!(sliced.tensors[0].classification_imag_max, Some(111.0));
    assert_eq!(sliced.tensors[0].class, LeafClass::Complex);
    assert_eq!(sliced.tensors[1], inputs.tensors[1]);
    assert_eq!(sliced.tensors[2].spec.shape, vec![1]);
    assert_eq!(sliced.tensors[2].real, vec![31.0]);
    assert_eq!(sliced.tensors[2].imag, vec![0.0]);
}

#[test]
fn sliced_inputs_reject_a_declared_dimension_that_disagrees_with_the_source() {
    let inputs = InputSet {
        tensors: vec![InputTensor {
            spec: TensorSpec {
                modes: vec![7],
                shape: vec![3],
            },
            real: vec![1.0, 2.0, 3.0],
            imag: vec![0.0; 3],
            class: LeafClass::Real,
            imag_max: 0.0,
            classification_imag_max: None,
        }],
    };
    let spec = SliceSpec {
        modes: vec![7],
        dimensions: vec![2],
        assignment_order: SliceAssignmentOrder::BinaryReflectedGray,
        slice_count: 2,
    };

    let error = slice_inputs(&inputs, &spec, &[1]).unwrap_err();

    match error {
        PlanError::InvalidTensor { index, detail } => {
            assert_eq!(index, 0);
            assert_eq!(
                detail,
                "slice mode 7 has source dimension 3 but the slice spec declares 2"
            );
        }
        other => panic!("expected an invalid tensor error, got {other:?}"),
    }
}

#[test]
fn sliced_plan_rebuilds_geometry_without_changing_tree_or_leaf_classes() {
    let source = build_plan_bundle(&matrix_scalar_network(0.5, 0.25), 1e-12).unwrap();

    let sliced = build_sliced_plan_bundle(&source, &[0, 2]).unwrap();

    assert_eq!(sliced.format, "omeinsum-sliced-plan-v1");
    assert_eq!(sliced.source_tree_hash, source.tree_hash);
    assert_eq!(sliced.reduced.tree_hash, source.tree_hash);
    assert_eq!(sliced.slice.modes, vec![0, 2]);
    assert_eq!(sliced.slice.dimensions, vec![2, 2]);
    assert_eq!(sliced.slice.slice_count, 4);
    assert_eq!(
        sliced
            .source_inputs
            .tensors
            .iter()
            .map(|tensor| tensor.class.clone())
            .collect::<Vec<_>>(),
        sliced
            .reduced
            .inputs
            .tensors
            .iter()
            .map(|tensor| tensor.class.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        sliced
            .reduced
            .realified_rank3
            .nodes
            .iter()
            .map(|node| node.kind.clone())
            .collect::<Vec<_>>(),
        source
            .realified_rank3
            .nodes
            .iter()
            .map(|node| node.kind.clone())
            .collect::<Vec<_>>()
    );
    assert_ne!(
        sliced.reduced.realified_rank3.plan_hash,
        source.realified_rank3.plan_hash
    );
    assert_eq!(sliced.reduced.inputs.tensors[0].spec.shape, vec![1, 2]);
    assert_eq!(sliced.reduced.inputs.tensors[1].spec.shape, vec![2, 1]);
    assert_eq!(sliced.reduced.inputs.tensors[2].spec.shape, vec![1, 1]);
    assert_eq!(
        sliced
            .aggregate_volumes
            .iter()
            .map(|volume| (
                volume.representation.clone(),
                volume.source_real_matmul_volume,
                volume.sliced_real_matmul_volume,
            ))
            .collect::<Vec<_>>(),
        vec![
            (Representation::RealSkeleton, 12, 12),
            (Representation::Flat4M, 48, 48),
            (Representation::RealifiedRank3, 32, 32),
        ]
    );
    sliced.validate().unwrap();
}

#[test]
fn sliced_complex_flat_and_rank3_sums_match_unsliced_values() {
    let source = build_plan_bundle(&matrix_scalar_network(0.5, 0.25), 1e-12).unwrap();
    let sliced = build_sliced_plan_bundle(&source, &[0, 2]).unwrap();
    let unsliced_reference = contract_complex64(&source.flat_4m, &source.inputs).unwrap();
    let mut sliced_reference = ComplexValue { re: 0.0, im: 0.0 };
    let mut sliced_flat = ComplexValue { re: 0.0, im: 0.0 };
    let mut sliced_rank3 = ComplexValue { re: 0.0, im: 0.0 };

    for assignment in gray_assignments(&sliced.slice).unwrap() {
        let inputs = slice_inputs(&sliced.source_inputs, &sliced.slice, &assignment).unwrap();
        let reference = contract_complex64(&sliced.reduced.flat_4m, &inputs).unwrap();
        sliced_reference.re += reference.re;
        sliced_reference.im += reference.im;

        for (plan, sum) in [
            (&sliced.reduced.flat_4m, &mut sliced_flat),
            (&sliced.reduced.realified_rank3, &mut sliced_rank3),
        ] {
            let mut executable = prepare_cpu_f64(plan, &inputs).unwrap();
            executable.enqueue().unwrap();
            executable.synchronize().unwrap();
            let output = executable.output().unwrap();
            sum.re += output.re;
            sum.im += output.im;
        }
    }

    for observed in [sliced_reference, sliced_flat, sliced_rank3] {
        assert!((observed.re - unsliced_reference.re).abs() <= 1e-12);
        assert!((observed.im - unsliced_reference.im).abs() <= 1e-12);
    }
}

#[test]
fn sliced_phase_canonicalized_plan_preserves_preprocessing_classes_and_scalar() {
    let network = phase_real_two_leaf_network(std::f64::consts::FRAC_PI_4, 0.0, true);
    let source = build_plan_bundle_with_preprocessing(
        &network,
        1e-12,
        LeafPreprocessing::PhaseCanonicalized,
    )
    .unwrap();
    let sliced = build_sliced_plan_bundle(&source, &[0]).unwrap();

    assert_eq!(
        sliced.reduced.leaf_preprocessing,
        LeafPreprocessing::PhaseCanonicalized
    );
    assert_eq!(
        sliced.reduced.phase_canonicalization,
        source.phase_canonicalization
    );
    assert_eq!(
        sliced
            .reduced
            .inputs
            .tensors
            .iter()
            .map(|tensor| tensor.class.clone())
            .collect::<Vec<_>>(),
        vec![LeafClass::Real, LeafClass::Complex]
    );
    assert_eq!(
        sliced
            .reduced
            .realified_rank3
            .nodes
            .iter()
            .map(|node| node.kind.clone())
            .collect::<Vec<_>>(),
        source
            .realified_rank3
            .nodes
            .iter()
            .map(|node| node.kind.clone())
            .collect::<Vec<_>>()
    );

    let expected = contract_complex64(&source.flat_4m, &source.inputs).unwrap();
    let mut observed = ComplexValue { re: 0.0, im: 0.0 };
    for assignment in gray_assignments(&sliced.slice).unwrap() {
        let inputs = slice_inputs(&sliced.source_inputs, &sliced.slice, &assignment).unwrap();
        let mut executable = prepare_cpu_f64(&sliced.reduced.realified_rank3, &inputs).unwrap();
        executable.enqueue().unwrap();
        executable.synchronize().unwrap();
        let value = executable.output().unwrap();
        observed.re += value.re;
        observed.im += value.im;
    }

    assert!((observed.re - expected.re).abs() <= 1e-12);
    assert!((observed.im - expected.im).abs() <= 1e-12);
}

#[test]
fn sliced_plan_rejects_duplicate_missing_or_nonbinary_physical_modes() {
    let source = build_plan_bundle(&matrix_scalar_network(0.5, 0.25), 1e-12).unwrap();
    assert!(matches!(
        build_sliced_plan_bundle(&source, &[0, 0]),
        Err(PlanError::InvalidNetwork(detail)) if detail == "slice modes must be unique"
    ));
    assert!(matches!(
        build_sliced_plan_bundle(&source, &[99]),
        Err(PlanError::InvalidNetwork(detail)) if detail == "slice mode 99 is absent"
    ));

    let mut dimension_one = two_leaf_scalar_network(0.5, 0.25);
    dimension_one.tensors[0].spec.shape = vec![1];
    dimension_one.tensors[0].real.truncate(1);
    dimension_one.tensors[0].imag.truncate(1);
    dimension_one.tensors[1].spec.shape = vec![1];
    dimension_one.tensors[1].real.truncate(1);
    dimension_one.tensors[1].imag.truncate(1);
    dimension_one.size_dict = vec![(0, 1)];
    let dimension_one = build_plan_bundle(&dimension_one, 1e-12).unwrap();
    assert!(matches!(
        build_sliced_plan_bundle(&dimension_one, &[0]),
        Err(PlanError::InvalidNetwork(detail))
            if detail == "slice mode 0 has dimension 1; expected 2"
    ));
}

#[test]
fn sliced_gray_schedule_rejects_inconsistent_and_overflowing_specs() {
    let inconsistent = SliceSpec {
        modes: vec![7, 9],
        dimensions: vec![2],
        assignment_order: SliceAssignmentOrder::BinaryReflectedGray,
        slice_count: 4,
    };
    assert!(matches!(
        gray_assignments(&inconsistent),
        Err(PlanError::InvalidNetwork(detail))
            if detail == "slice modes and dimensions must have the same nonzero length"
    ));

    let overflow = SliceSpec {
        modes: (0..usize::BITS).map(|mode| mode as i32).collect(),
        dimensions: vec![2; usize::BITS as usize],
        assignment_order: SliceAssignmentOrder::BinaryReflectedGray,
        slice_count: 0,
    };
    assert!(matches!(
        gray_assignments(&overflow),
        Err(PlanError::InvalidNetwork(detail)) if detail == "slice count overflows usize"
    ));
}

#[test]
fn prepared_input_update_changes_a_cpu_leaf_without_repreparing() {
    let bundle = build_plan_bundle(&two_leaf_scalar_network(0.0, 0.0), 1e-12).unwrap();
    let mut executable = prepare_cpu_f64(&bundle.realified_rank3, &bundle.inputs).unwrap();
    executable.enqueue().unwrap();
    executable.synchronize().unwrap();
    assert_eq!(
        executable.output().unwrap(),
        ComplexValue { re: 11.0, im: 0.0 }
    );

    let mut tensor = bundle.inputs.tensors[0].clone();
    tensor.real = vec![5.0, 6.0];
    executable
        .update_inputs(&[InputUpdate { index: 0, tensor }])
        .unwrap();
    executable.enqueue().unwrap();
    executable.synchronize().unwrap();

    assert_eq!(
        executable.output().unwrap(),
        ComplexValue { re: 39.0, im: 0.0 }
    );
}

#[test]
fn prepared_input_update_rejects_duplicate_index_geometry_class_and_plane_drift() {
    let bundle = build_plan_bundle(&two_leaf_scalar_network(0.0, 0.0), 1e-12).unwrap();
    let unchanged = bundle.inputs.tensors[0].clone();

    let mut duplicate = prepare_cpu_f64(&bundle.realified_rank3, &bundle.inputs).unwrap();
    let duplicate_error = duplicate
        .update_inputs(&[
            InputUpdate {
                index: 0,
                tensor: unchanged.clone(),
            },
            InputUpdate {
                index: 0,
                tensor: unchanged.clone(),
            },
        ])
        .unwrap_err();
    assert!(matches!(
        duplicate_error,
        ExecutionError::InvalidPlan(detail)
            if detail == "input update index 0 appears more than once"
    ));

    let mut out_of_range = prepare_cpu_f64(&bundle.realified_rank3, &bundle.inputs).unwrap();
    let range_error = out_of_range
        .update_inputs(&[InputUpdate {
            index: 2,
            tensor: unchanged.clone(),
        }])
        .unwrap_err();
    assert!(matches!(
        range_error,
        ExecutionError::InvalidPlan(detail)
            if detail == "input update index 2 is outside 2 leaves"
    ));

    let mut wrong_geometry = unchanged.clone();
    wrong_geometry.spec.shape = vec![1];
    wrong_geometry.real.truncate(1);
    wrong_geometry.imag.truncate(1);
    let mut geometry = prepare_cpu_f64(&bundle.realified_rank3, &bundle.inputs).unwrap();
    let geometry_error = geometry
        .update_inputs(&[InputUpdate {
            index: 0,
            tensor: wrong_geometry,
        }])
        .unwrap_err();
    assert!(matches!(
        geometry_error,
        ExecutionError::InvalidPlan(detail)
            if detail == "input update 0 geometry differs from its prepared leaf"
    ));

    let mut wrong_class = unchanged;
    wrong_class.class = LeafClass::Complex;
    wrong_class.imag = vec![1.0, 0.0];
    wrong_class.imag_max = 1.0;
    wrong_class.classification_imag_max = Some(1.0);
    let mut class = prepare_cpu_f64(&bundle.realified_rank3, &bundle.inputs).unwrap();
    let class_error = class
        .update_inputs(&[InputUpdate {
            index: 0,
            tensor: wrong_class,
        }])
        .unwrap_err();
    assert!(matches!(
        class_error,
        ExecutionError::InvalidPlan(detail)
            if detail == "input update 0 changes the frozen leaf class"
    ));

    let mut wrong_plane = bundle.inputs.tensors[0].clone();
    wrong_plane.imag = vec![1.0, 0.0];
    wrong_plane.imag_max = 1.0;
    let mut plane = prepare_cpu_f64(&bundle.realified_rank3, &bundle.inputs).unwrap();
    let plane_error = plane
        .update_inputs(&[InputUpdate {
            index: 0,
            tensor: wrong_plane,
        }])
        .unwrap_err();
    assert!(matches!(
        plane_error,
        ExecutionError::InvalidPlan(detail)
            if detail == "input update 0 real leaf has a nonzero imaginary plane"
    ));
}

#[test]
fn sliced_executable_runs_every_gray_edge_and_updates_only_affected_leaves() {
    let source = build_plan_bundle(&matrix_scalar_network(0.5, 0.25), 1e-12).unwrap();
    let sliced = build_sliced_plan_bundle(&source, &[0, 2]).unwrap();
    let state = Arc::new(Mutex::new(SlicedFakeState::default()));
    let inner = SlicedFakeExecutable {
        state: Arc::clone(&state),
        outputs: vec![
            ComplexValue { re: 1.0, im: 0.5 },
            ComplexValue { re: 2.0, im: 1.0 },
            ComplexValue { re: 3.0, im: 1.5 },
            ComplexValue { re: 4.0, im: 2.0 },
        ],
    };
    let mut executable = SlicedExecutable::new(inner, &sliced.source_inputs, &sliced).unwrap();

    executable.enqueue().unwrap();
    executable.synchronize().unwrap();

    assert_eq!(
        executable.output().unwrap(),
        ComplexValue { re: 10.0, im: 5.0 }
    );
    assert_eq!(
        executable.execution_mode_label(),
        "sliced-host-f64-accumulated"
    );
    let state = state.lock().unwrap();
    assert_eq!(state.enqueue_count, 4);
    assert_eq!(state.sync_count, 4);
    assert_eq!(
        state
            .updates
            .iter()
            .map(|batch| batch.iter().map(|update| update.index).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        vec![vec![0, 2], vec![1, 2], vec![0, 2], vec![1, 2]]
    );
    assert_eq!(state.updates[0][0].tensor.real, vec![1.0, 3.0]);
    assert_eq!(state.updates[0][1].tensor.real, vec![1.0]);
    assert_eq!(state.updates[1][0].tensor.real, vec![2.0, 3.0]);
    assert_eq!(state.updates[1][1].tensor.real, vec![0.25]);
    assert_eq!(state.updates[2][0].tensor.real, vec![2.0, 4.0]);
    assert_eq!(state.updates[2][1].tensor.real, vec![2.0]);
    assert_eq!(state.updates[3][0].tensor.real, vec![0.5, -1.0]);
    assert_eq!(state.updates[3][1].tensor.real, vec![-0.5]);
}

#[test]
fn sliced_executable_cpu_matches_the_phase_canonicalized_complex_reference() {
    let source = build_plan_bundle_with_preprocessing(
        &phase_real_two_leaf_network(std::f64::consts::FRAC_PI_4, 0.0, true),
        1e-12,
        LeafPreprocessing::PhaseCanonicalized,
    )
    .unwrap();
    let expected = contract_complex64(&source.flat_4m, &source.inputs).unwrap();
    let sliced = build_sliced_plan_bundle(&source, &[0]).unwrap();
    let inner = prepare_cpu_f64(&sliced.reduced.realified_rank3, &sliced.reduced.inputs).unwrap();
    let mut executable = SlicedExecutable::new(inner, &sliced.source_inputs, &sliced).unwrap();

    executable.enqueue().unwrap();
    executable.synchronize().unwrap();
    let observed = executable.output().unwrap();

    assert!((observed.re - expected.re).abs() <= 1e-12);
    assert!((observed.im - expected.im).abs() <= 1e-12);
}

#[test]
fn canonical_geometry_preserves_tree_postorder() {
    let network = canonical_geometry_network(vec![(0, 2), (1, 3), (2, 4)]);

    let plan = build_geometry_plan(&network).unwrap();
    assert_eq!(
        plan.nodes.iter().map(|node| node.id).collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(plan.nodes[0].left, ValueId(0));
    assert_eq!(plan.nodes[0].right, ValueId(1));
    assert_eq!(plan.nodes[0].output, ValueId(3));
    assert_eq!(plan.nodes[1].left, ValueId(3));
    assert_eq!(plan.nodes[1].right, ValueId(2));
    assert_eq!(plan.nodes[1].output, ValueId(4));
    assert_eq!(plan.output, ValueId(4));
    assert!(plan.values[plan.output.0].tensor.modes.is_empty());

    let first = &plan.nodes[0].contraction;
    assert_eq!((first.batch, first.m, first.k, first.n), (1, 2, 3, 4));
    assert_eq!(first.left_permutation, vec![0, 1]);
    assert_eq!(first.right_permutation, vec![0, 1]);
    assert_eq!(first.output_permutation, None);
}

#[test]
fn canonical_hashes_are_deterministic_and_cover_node_kinds() {
    let forward =
        build_geometry_plan(&canonical_geometry_network(vec![(0, 2), (1, 3), (2, 4)])).unwrap();
    let reverse =
        build_geometry_plan(&canonical_geometry_network(vec![(2, 4), (1, 3), (0, 2)])).unwrap();

    assert_eq!(forward.tree_hash, reverse.tree_hash);
    assert_eq!(forward.plan_hash, reverse.plan_hash);

    let mut mutated = forward.clone();
    mutated.nodes[0].kind = omeinsum::static_plan::KernelKind::Flat4M;
    let mutated_hash = mutated.recompute_plan_hash().unwrap();
    assert_eq!(mutated.tree_hash, forward.tree_hash);
    assert_ne!(mutated_hash, forward.plan_hash);
}

#[test]
fn leaf_classification_uses_inclusive_f64_tolerance_and_rejects_nonfinite() {
    let source = two_leaf_scalar_network(0.0, 0.0);
    let mut network = ComplexNetwork {
        tensors: vec![
            ComplexTensor {
                imag: vec![1e-12 - 1e-16, 0.0],
                ..source.tensors[0].clone()
            },
            ComplexTensor {
                imag: vec![1e-12, 0.0],
                ..source.tensors[0].clone()
            },
            ComplexTensor {
                imag: vec![1e-12 + 1e-16, 0.0],
                ..source.tensors[0].clone()
            },
        ],
        output_modes: vec![],
        size_dict: vec![(0, 2)],
        tree: BinaryContractionTree::Node {
            output_modes: vec![],
            left: Box::new(BinaryContractionTree::Node {
                output_modes: vec![0],
                left: Box::new(BinaryContractionTree::Leaf { tensor_index: 0 }),
                right: Box::new(BinaryContractionTree::Leaf { tensor_index: 1 }),
            }),
            right: Box::new(BinaryContractionTree::Leaf { tensor_index: 2 }),
        },
    };

    let bundle = build_plan_bundle(&network, 1e-12).unwrap();
    assert_eq!(
        bundle
            .inputs
            .tensors
            .iter()
            .map(|tensor| tensor.class.clone())
            .collect::<Vec<_>>(),
        vec![LeafClass::Real, LeafClass::Real, LeafClass::Complex]
    );
    assert!(bundle.inputs.tensors[0]
        .imag
        .iter()
        .all(|value| *value == 0.0));
    assert!(bundle.inputs.tensors[1]
        .imag
        .iter()
        .all(|value| *value == 0.0));
    assert_eq!(bundle.inputs.tensors[2].imag_max, 1e-12 + 1e-16);

    network.tensors[1].real[0] = f64::INFINITY;
    assert!(build_plan_bundle(&network, 1e-12).is_err());
}

#[test]
fn phase_canonicalization_makes_sqrty_shaped_leaf_real_and_preserves_scalar() {
    let network = phase_real_two_leaf_network(std::f64::consts::FRAC_PI_4, 0.0, true);
    let raw = build_plan_bundle(&network, 1e-12).unwrap();
    assert_eq!(raw.realified_rank3.stats.complex_leaf_count, 2);
    let raw_json = serde_json::to_value(&raw).unwrap();
    assert!(raw_json.get("leaf_preprocessing").is_none());
    assert!(raw_json.get("phase_canonicalization").is_none());

    let canonical = build_plan_bundle_with_preprocessing(
        &network,
        1e-12,
        LeafPreprocessing::PhaseCanonicalized,
    )
    .unwrap();
    assert_eq!(
        canonical.leaf_preprocessing,
        LeafPreprocessing::PhaseCanonicalized
    );
    assert_eq!(canonical.realified_rank3.stats.complex_leaf_count, 1);
    assert_eq!(canonical.inputs.tensors[0].class, LeafClass::Real);
    assert_eq!(canonical.inputs.tensors[0].real, vec![1.0, -1.0]);
    assert_eq!(canonical.inputs.tensors[0].imag, vec![0.0, 0.0]);
    assert_eq!(canonical.inputs.tensors[1].class, LeafClass::Complex);
    let report = canonical.phase_canonicalization.as_ref().unwrap();
    assert_eq!(report.source_real_leaf_count, 0);
    assert_eq!(report.source_complex_leaf_count, 2);
    assert_eq!(report.canonicalized_leaf_count, 1);
    assert_eq!(report.phase_anchor, Some(1));
    assert!((report.accumulated_phase.re - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-15);
    assert!((report.accumulated_phase.im - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-15);

    let raw_value = contract_complex64(&raw.flat_4m, &raw.inputs).unwrap();
    let canonical_value = contract_complex64(&canonical.flat_4m, &canonical.inputs).unwrap();
    assert!((raw_value.re - canonical_value.re).abs() <= 1e-12);
    assert!((raw_value.im - canonical_value.im).abs() <= 1e-12);
    let mut rank3 = prepare_cpu_f64(&canonical.realified_rank3, &canonical.inputs).unwrap();
    rank3.enqueue().unwrap();
    rank3.synchronize().unwrap();
    let rank3_value = rank3.output().unwrap();
    assert!((raw_value.re - rank3_value.re).abs() <= 1e-12);
    assert!((raw_value.im - rank3_value.im).abs() <= 1e-12);

    let mut invalid = canonical.clone();
    invalid
        .phase_canonicalization
        .as_mut()
        .unwrap()
        .phase_anchor = Some(invalid.inputs.tensors.len());
    assert!(invalid.validate().is_err());
    let mut invalid_no_op = canonical.clone();
    let invalid_no_op_report = invalid_no_op.phase_canonicalization.as_mut().unwrap();
    invalid_no_op_report.canonicalized_leaf_count = 0;
    invalid_no_op_report.phase_anchor = None;
    invalid_no_op_report.accumulated_phase = ComplexValue { re: 0.0, im: 1.0 };
    assert!(invalid_no_op.validate().is_err());
    let mut invalid = canonical;
    invalid.phase_canonicalization = None;
    assert!(invalid.validate().is_err());
}

#[test]
fn phase_canonicalization_solves_the_full_tolerance_feasibility_interval() {
    let tolerance = 1e-12;
    let mut network = phase_real_two_leaf_network(std::f64::consts::FRAC_PI_4, 0.0, true);
    let phase = num_complex::Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    let noisy_real = [
        phase * num_complex::Complex64::new(1.0, tolerance),
        phase * num_complex::Complex64::new(1.0, -tolerance),
    ];
    network.tensors[0].real = noisy_real.iter().map(|value| value.re).collect();
    network.tensors[0].imag = noisy_real.iter().map(|value| value.im).collect();

    let canonical = build_plan_bundle_with_preprocessing(
        &network,
        tolerance,
        LeafPreprocessing::PhaseCanonicalized,
    )
    .unwrap();

    assert_eq!(canonical.inputs.tensors[0].class, LeafClass::Real);
    assert_eq!(canonical.realified_rank3.stats.complex_leaf_count, 1);
    assert_eq!(
        canonical
            .phase_canonicalization
            .as_ref()
            .unwrap()
            .canonicalized_leaf_count,
        1
    );
}

#[test]
fn phase_canonicalization_retains_one_anchor_for_all_phase_real_network() {
    let network = phase_real_two_leaf_network(
        std::f64::consts::FRAC_PI_4,
        std::f64::consts::FRAC_PI_6,
        false,
    );
    let raw = build_plan_bundle(&network, 1e-12).unwrap();
    let canonical = build_plan_bundle_with_preprocessing(
        &network,
        1e-12,
        LeafPreprocessing::PhaseCanonicalized,
    )
    .unwrap();

    assert_eq!(raw.realified_rank3.stats.complex_leaf_count, 2);
    assert_eq!(canonical.realified_rank3.stats.complex_leaf_count, 1);
    assert_eq!(
        canonical
            .inputs
            .tensors
            .iter()
            .filter(|tensor| tensor.class == LeafClass::Real)
            .count(),
        1
    );
    let report = canonical.phase_canonicalization.as_ref().unwrap();
    assert_eq!(report.canonicalized_leaf_count, 2);
    assert_eq!(report.phase_anchor, Some(0));

    let raw_value = contract_complex64(&raw.flat_4m, &raw.inputs).unwrap();
    let canonical_value = contract_complex64(&canonical.flat_4m, &canonical.inputs).unwrap();
    assert!((raw_value.re - canonical_value.re).abs() <= 1e-12);
    assert!((raw_value.im - canonical_value.im).abs() <= 1e-12);
}

#[test]
fn phase_canonicalization_allows_zero_complex_leaves_when_phases_cancel() {
    let network = phase_real_two_leaf_network(
        std::f64::consts::FRAC_PI_4,
        -std::f64::consts::FRAC_PI_4,
        false,
    );
    let raw = build_plan_bundle(&network, 1e-12).unwrap();
    let canonical = build_plan_bundle_with_preprocessing(
        &network,
        1e-12,
        LeafPreprocessing::PhaseCanonicalized,
    )
    .unwrap();

    assert_eq!(raw.realified_rank3.stats.complex_leaf_count, 2);
    assert_eq!(canonical.realified_rank3.stats.complex_leaf_count, 0);
    assert_eq!(
        canonical
            .inputs
            .tensors
            .iter()
            .filter(|tensor| tensor.class == LeafClass::Real)
            .count(),
        2
    );
    let report = canonical.phase_canonicalization.as_ref().unwrap();
    assert_eq!(report.canonicalized_leaf_count, 2);
    assert_eq!(report.phase_anchor, Some(0));
    assert!((report.accumulated_phase.re.abs() - 1.0).abs() <= 1e-15);
    assert!(report.accumulated_phase.im.abs() <= 1e-15);

    let raw_value = contract_complex64(&raw.flat_4m, &raw.inputs).unwrap();
    let canonical_value =
        contract_complex64(&canonical.realified_rank3, &canonical.inputs).unwrap();
    assert!((raw_value.re - canonical_value.re).abs() <= 1e-12);
    assert!((raw_value.im - canonical_value.im).abs() <= 1e-12);
}

#[test]
fn selective_transitions_and_scratch_roles_are_frozen() {
    let cases = [
        (0.0, 0.0, KernelKind::RealReal, vec![Plane::Real]),
        (
            1.0,
            0.0,
            KernelKind::RideLeft,
            vec![Plane::Real, Plane::Imag],
        ),
        (
            0.0,
            1.0,
            KernelKind::RideRight,
            vec![Plane::Real, Plane::Imag],
        ),
        (
            1.0,
            1.0,
            KernelKind::Merge3M,
            vec![Plane::Real, Plane::Imag],
        ),
    ];

    for (left_imag, right_imag, expected_kind, expected_planes) in cases {
        let bundle =
            build_plan_bundle(&two_leaf_scalar_network(left_imag, right_imag), 1e-12).unwrap();
        let node = &bundle.realified_rank3.nodes[0];
        assert_eq!(node.kind, expected_kind);
        assert_eq!(
            bundle.realified_rank3.values[node.output.0].planes,
            expected_planes
        );
        let expected_scratch = if expected_kind == KernelKind::Merge3M {
            vec![
                ScratchRole::LeftSum,
                ScratchRole::RightSum,
                ScratchRole::Product1,
                ScratchRole::Product2,
                ScratchRole::Product3,
            ]
        } else {
            vec![]
        };
        assert_eq!(
            node.scratch
                .iter()
                .map(|scratch| scratch.role.clone())
                .collect::<Vec<_>>(),
            expected_scratch
        );
        assert_eq!(
            bundle.flat_4m.nodes[0]
                .scratch
                .iter()
                .map(|scratch| scratch.role.clone())
                .collect::<Vec<_>>(),
            vec![
                ScratchRole::Product1,
                ScratchRole::Product2,
                ScratchRole::Product3,
                ScratchRole::Product4,
            ]
        );
    }
}

#[test]
fn plan_accounting_invariants_and_validation_hold() {
    let mut network = two_leaf_scalar_network(1.0, 0.0);
    network
        .tensors
        .extend(two_leaf_scalar_network(0.0, 1.0).tensors);
    network.tree = BinaryContractionTree::Node {
        output_modes: vec![],
        left: Box::new(BinaryContractionTree::Node {
            output_modes: vec![0],
            left: Box::new(BinaryContractionTree::Leaf { tensor_index: 0 }),
            right: Box::new(BinaryContractionTree::Leaf { tensor_index: 1 }),
        }),
        right: Box::new(BinaryContractionTree::Node {
            output_modes: vec![0],
            left: Box::new(BinaryContractionTree::Leaf { tensor_index: 2 }),
            right: Box::new(BinaryContractionTree::Leaf { tensor_index: 3 }),
        }),
    };

    let bundle = build_plan_bundle(&network, 1e-12).unwrap();
    bundle.validate().unwrap();
    let stats = &bundle.realified_rank3.stats;
    let cost = stats.realification_cost.as_ref().unwrap();
    assert_eq!(
        stats.merge_3m_nodes,
        stats.complex_leaf_count.saturating_sub(1)
    );
    assert!(
        (cost.real_real_fraction + cost.ride_fraction + cost.merge_fraction - 1.0).abs() < 1e-15
    );
    assert_eq!(
        cost.predicted_arithmetic_overhead,
        1.0 + cost.ride_fraction + 2.0 * cost.merge_fraction
    );
    assert_eq!(
        stats.real_matmul_volume,
        cost.real_real_volume + 2 * cost.ride_volume + 3 * cost.merge_volume
    );
    assert_eq!(
        bundle.flat_4m.stats.real_matmul_volume,
        4 * bundle.flat_4m.stats.real_skeleton_volume
    );
    assert!(bundle.real_skeleton.stats.realification_cost.is_none());
    assert!(bundle.flat_4m.stats.realification_cost.is_none());

    let mut mutated = bundle;
    mutated.realified_rank3.nodes[0].kind = KernelKind::Flat4M;
    assert!(mutated.validate().is_err());

    let leaf_only = ComplexNetwork {
        tensors: vec![ComplexTensor {
            spec: TensorSpec {
                modes: vec![],
                shape: vec![],
            },
            real: vec![1.0],
            imag: vec![0.0],
        }],
        output_modes: vec![],
        size_dict: vec![],
        tree: BinaryContractionTree::Leaf { tensor_index: 0 },
    };
    assert!(build_plan_bundle(&leaf_only, 1e-12).is_err());
}

#[test]
fn cpu_plane_kernels_match_hand_calculated_complex_matrices() {
    for (left_imag, right_imag, expected_kind) in [
        (0.0, 0.0, KernelKind::RealReal),
        (0.75, 0.0, KernelKind::RideLeft),
        (0.0, -0.5, KernelKind::RideRight),
        (0.75, -0.5, KernelKind::Merge3M),
    ] {
        let network = matrix_scalar_network(left_imag, right_imag);
        let bundle = build_plan_bundle(&network, 1e-12).unwrap();
        assert_eq!(bundle.realified_rank3.nodes[0].kind, expected_kind);

        let expected = {
            let complex = |tensor: &ComplexTensor<f64>, index: usize| {
                num_complex::Complex64::new(tensor.real[index], tensor.imag[index])
            };
            let mut result = num_complex::Complex64::new(0.0, 0.0);
            for i in 0..2 {
                for j in 0..2 {
                    let mut product = num_complex::Complex64::new(0.0, 0.0);
                    for k in 0..2 {
                        product += complex(&network.tensors[0], i + 2 * k)
                            * complex(&network.tensors[1], k + 2 * j);
                    }
                    result += product * complex(&network.tensors[2], i + 2 * j);
                }
            }
            result
        };

        for plan in [&bundle.flat_4m, &bundle.realified_rank3] {
            let mut executable = prepare_cpu_f64(plan, &bundle.inputs).unwrap();
            executable.enqueue().unwrap();
            executable.synchronize().unwrap();
            let actual = executable.output().unwrap();
            assert!((actual.re - expected.re).abs() <= 1e-12);
            assert!((actual.im - expected.im).abs() <= 1e-12);
        }
    }

    let bundle = build_plan_bundle(&matrix_scalar_network(0.5, -0.25), 1e-12).unwrap();
    let mut oversized = bundle.inputs.clone();
    oversized.tensors[0].real[0] = f64::MAX;
    assert!(prepare_cpu_f32(&bundle.realified_rank3, &oversized).is_err());
}

#[test]
fn cpu_random_realification_properties_cover_seeded_tree_shapes() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(20260723);
    for case in 0..100usize {
        let leaves = 2 + case % 7;
        let dimension = 1 + (case / 7) % 4;
        let class_pattern = case % 4;
        let mut tensors = Vec::with_capacity(leaves);
        for leaf in 0..leaves {
            let is_complex = match class_pattern {
                0 => false,
                1 => leaf == 0,
                2 => leaf % 2 == 0,
                _ => true,
            };
            let mut real = Vec::with_capacity(dimension);
            let mut imag = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                let magnitude = rng.random_range(0.2_f64..1.2);
                let phase = if is_complex {
                    rng.random_range(-std::f64::consts::PI..std::f64::consts::PI)
                } else {
                    0.0
                };
                let value = num_complex::Complex64::from_polar(magnitude, phase);
                real.push(value.re);
                imag.push(if is_complex && leaf % 2 == 1 {
                    -value.im
                } else {
                    value.im
                });
            }
            tensors.push(ComplexTensor {
                spec: TensorSpec {
                    modes: vec![0],
                    shape: vec![dimension],
                },
                real,
                imag,
            });
        }
        let network = ComplexNetwork {
            tensors,
            output_modes: vec![],
            size_dict: vec![(0, dimension)],
            tree: vector_chain_tree(leaves),
        };
        let bundle = build_plan_bundle(&network, 1e-12).unwrap();
        let reference = (0..dimension)
            .map(|position| {
                network
                    .tensors
                    .iter()
                    .map(|tensor| {
                        num_complex::Complex64::new(tensor.real[position], tensor.imag[position])
                    })
                    .product::<num_complex::Complex64>()
            })
            .sum::<num_complex::Complex64>();

        for plan in [&bundle.flat_4m, &bundle.realified_rank3] {
            let mut executable = prepare_cpu_f64(plan, &bundle.inputs).unwrap();
            executable.enqueue().unwrap();
            executable.synchronize().unwrap();
            let actual = executable.output().unwrap();
            let error = (num_complex::Complex64::new(actual.re, actual.im) - reference).norm();
            assert!(
                error <= 1e-12 + 1e-9 * reference.norm(),
                "case {case}, {:?}: error {error}",
                plan.representation
            );

            let mut executable = prepare_cpu_f32(plan, &bundle.inputs).unwrap();
            executable.enqueue().unwrap();
            executable.synchronize().unwrap();
            let actual = executable.output().unwrap();
            assert!(actual.re.is_finite() && actual.im.is_finite());
            let scale = reference
                .norm()
                .max(2.0_f64.powf(-((case % 8 + 1) as f64) / 2.0));
            let scaled_error =
                (num_complex::Complex64::new(actual.re, actual.im) - reference).norm() / scale;
            assert!(
                scaled_error < 2e-5,
                "case {case}, {:?}: scaled F32 error {scaled_error}",
                plan.representation
            );
        }
    }
}

#[test]
fn cpu_frozen_geometry_honors_operand_and_output_permutations() {
    let network = ComplexNetwork {
        tensors: vec![
            ComplexTensor {
                spec: TensorSpec {
                    modes: vec![1, 0],
                    shape: vec![3, 2],
                },
                real: vec![1.0, 2.0, -1.0, 0.5, 3.0, 4.0],
                imag: vec![0.2, 0.0, -0.1, 0.3, 0.0, -0.4],
            },
            ComplexTensor {
                spec: TensorSpec {
                    modes: vec![2, 1],
                    shape: vec![4, 3],
                },
                real: (0..12).map(|value| value as f64 / 5.0 - 0.7).collect(),
                imag: (0..12).map(|value| (value as f64 - 4.0) / 17.0).collect(),
            },
            ComplexTensor {
                spec: TensorSpec {
                    modes: vec![2, 0],
                    shape: vec![4, 2],
                },
                real: vec![1.0, 0.5, -0.5, 2.0, 0.25, -1.0, 1.5, 0.75],
                imag: vec![0.0; 8],
            },
        ],
        output_modes: vec![],
        size_dict: vec![(0, 2), (1, 3), (2, 4)],
        tree: BinaryContractionTree::Node {
            output_modes: vec![],
            left: Box::new(BinaryContractionTree::Node {
                output_modes: vec![2, 0],
                left: Box::new(BinaryContractionTree::Leaf { tensor_index: 0 }),
                right: Box::new(BinaryContractionTree::Leaf { tensor_index: 1 }),
            }),
            right: Box::new(BinaryContractionTree::Leaf { tensor_index: 2 }),
        },
    };
    let bundle = build_plan_bundle(&network, 1e-12).unwrap();
    assert_eq!(
        bundle.realified_rank3.nodes[0].contraction.left_permutation,
        vec![1, 0]
    );
    assert_eq!(
        bundle.realified_rank3.nodes[0]
            .contraction
            .right_permutation,
        vec![1, 0]
    );
    assert_eq!(
        bundle.realified_rank3.nodes[0]
            .contraction
            .output_permutation,
        Some(vec![1, 0])
    );

    let complex = |tensor: usize, index: usize| {
        num_complex::Complex64::new(
            network.tensors[tensor].real[index],
            network.tensors[tensor].imag[index],
        )
    };
    let mut reference = num_complex::Complex64::new(0.0, 0.0);
    for i in 0..2 {
        for j in 0..4 {
            let mut intermediate = num_complex::Complex64::new(0.0, 0.0);
            for k in 0..3 {
                intermediate += complex(0, k + 3 * i) * complex(1, j + 4 * k);
            }
            reference += intermediate * complex(2, j + 4 * i);
        }
    }

    for plan in [&bundle.flat_4m, &bundle.realified_rank3] {
        let mut executable = prepare_cpu_f64(plan, &bundle.inputs).unwrap();
        executable.enqueue().unwrap();
        executable.synchronize().unwrap();
        let actual = executable.output().unwrap();
        let error = (num_complex::Complex64::new(actual.re, actual.im) - reference).norm();
        assert!(error <= 1e-12 + 1e-9 * reference.norm());
    }
}

#[test]
fn complex_reference_preserves_supplied_floating_point_postorder() {
    let tensor = |value| ComplexTensor {
        spec: TensorSpec {
            modes: vec![0],
            shape: vec![1],
        },
        real: vec![value],
        imag: vec![0.0],
    };
    let mut left_associative = ComplexNetwork {
        tensors: vec![tensor(1e308), tensor(1e-308), tensor(1e-308)],
        output_modes: vec![],
        size_dict: vec![(0, 1)],
        tree: vector_chain_tree(3),
    };
    let left_bundle = build_plan_bundle(&left_associative, 1e-12).unwrap();
    let left = contract_complex64(&left_bundle.realified_rank3, &left_bundle.inputs).unwrap();

    left_associative.tree = BinaryContractionTree::Node {
        output_modes: vec![],
        left: Box::new(BinaryContractionTree::Leaf { tensor_index: 0 }),
        right: Box::new(BinaryContractionTree::Node {
            output_modes: vec![0],
            left: Box::new(BinaryContractionTree::Leaf { tensor_index: 1 }),
            right: Box::new(BinaryContractionTree::Leaf { tensor_index: 2 }),
        }),
    };
    let right_bundle = build_plan_bundle(&left_associative, 1e-12).unwrap();
    let right = contract_complex64(&right_bundle.realified_rank3, &right_bundle.inputs).unwrap();

    assert!(left.re > 0.0);
    assert_eq!(right.re, 0.0);
    assert_ne!(left, right);
}

#[test]
fn complex_reference_matches_two_leaf_flat_and_selective_results() {
    let bundle = build_plan_bundle(&two_leaf_scalar_network(0.25, -0.5), 1e-12).unwrap();
    let reference = contract_complex64(&bundle.realified_rank3, &bundle.inputs).unwrap();
    for plan in [&bundle.flat_4m, &bundle.realified_rank3] {
        let mut executable = prepare_cpu_f64(plan, &bundle.inputs).unwrap();
        executable.enqueue().unwrap();
        executable.synchronize().unwrap();
        let actual = executable.output().unwrap();
        assert!((actual.re - reference.re).abs() <= 1e-12);
        assert!((actual.im - reference.im).abs() <= 1e-12);
    }
}

#[test]
fn benchmark_protocol_calibrates_interleaves_and_summarizes_samples() {
    let order = Arc::new(Mutex::new(Vec::new()));
    let mut fast = FakeExecutable::new(
        0,
        Duration::from_micros(100),
        ComplexValue { re: 1.0, im: 0.0 },
        Arc::clone(&order),
    );
    let mut slow = FakeExecutable::new(
        1,
        Duration::from_millis(2),
        ComplexValue { re: 2.0, im: 0.0 },
        Arc::clone(&order),
    );
    let config = BenchmarkConfig {
        warmups: 3,
        samples: 5,
        min_sample_ms: 1,
        measurement_order_seed: 20260723,
    };
    let timings = {
        let mut targets = [
            BenchmarkTarget {
                backend: "fast",
                dtype: "f64",
                execution_mode: "prepared",
                executable: &mut fast,
            },
            BenchmarkTarget {
                backend: "slow",
                dtype: "f64",
                execution_mode: "prepared",
                executable: &mut slow,
            },
        ];
        benchmark_prepared(&mut targets, &config).unwrap()
    };

    assert_eq!(timings.len(), 2);
    for timing in &timings {
        assert_eq!(timing.raw_seconds_per_contraction.len(), 5);
        let mut sorted = timing.raw_seconds_per_contraction.clone();
        sorted.sort_by(f64::total_cmp);
        assert_eq!(timing.best_seconds, sorted[0]);
        assert_eq!(timing.median_seconds, sorted[2]);
        assert_eq!(
            timing.iqr_seconds,
            (sorted[3] + sorted[4]) / 2.0 - (sorted[0] + sorted[1]) / 2.0
        );
    }
    let fast_timing = timings
        .iter()
        .find(|timing| timing.backend == "fast")
        .unwrap();
    let slow_timing = timings
        .iter()
        .find(|timing| timing.backend == "slow")
        .unwrap();
    assert_eq!(slow_timing.inner_iterations, 1);

    assert_eq!(&fast.batches[..3], &[1, 1, 1]);
    assert_eq!(&slow.batches[..3], &[1, 1, 1]);
    let fast_calibration = &fast.batches[3..fast.batches.len() - 5];
    assert_eq!(fast_calibration.first(), Some(&1));
    assert!(fast_calibration
        .windows(2)
        .all(|window| window[1] == 2 * window[0]));
    assert!(fast.batches[fast.batches.len() - 5..]
        .iter()
        .all(|iterations| *iterations == fast_timing.inner_iterations));
}

fn benchmark_event_order(seed: u64) -> Vec<usize> {
    let order = Arc::new(Mutex::new(Vec::new()));
    let mut first = FakeExecutable::new(
        0,
        Duration::from_millis(2),
        ComplexValue { re: 0.0, im: 0.0 },
        Arc::clone(&order),
    );
    let mut second = FakeExecutable::new(
        1,
        Duration::from_millis(2),
        ComplexValue { re: 0.0, im: 0.0 },
        Arc::clone(&order),
    );
    let mut third = FakeExecutable::new(
        2,
        Duration::from_millis(2),
        ComplexValue { re: 0.0, im: 0.0 },
        Arc::clone(&order),
    );
    {
        let mut targets = [
            BenchmarkTarget {
                backend: "first",
                dtype: "f64",
                execution_mode: "prepared",
                executable: &mut first,
            },
            BenchmarkTarget {
                backend: "second",
                dtype: "f64",
                execution_mode: "prepared",
                executable: &mut second,
            },
            BenchmarkTarget {
                backend: "third",
                dtype: "f64",
                execution_mode: "prepared",
                executable: &mut third,
            },
        ];
        benchmark_prepared(
            &mut targets,
            &BenchmarkConfig {
                warmups: 3,
                samples: 5,
                min_sample_ms: 1,
                measurement_order_seed: seed,
            },
        )
        .unwrap();
    }
    let events = order.lock().unwrap().clone();
    events
}

#[test]
fn benchmark_interleaving_is_seeded_and_nonfinite_outputs_fail() {
    assert_eq!(
        benchmark_event_order(20260723),
        benchmark_event_order(20260723)
    );
    assert_ne!(
        benchmark_event_order(20260723),
        benchmark_event_order(20260724)
    );

    let order = Arc::new(Mutex::new(Vec::new()));
    let mut invalid = FakeExecutable::new(
        0,
        Duration::from_millis(2),
        ComplexValue {
            re: f64::NAN,
            im: 0.0,
        },
        order,
    );
    let mut targets = [BenchmarkTarget {
        backend: "invalid",
        dtype: "f64",
        execution_mode: "prepared",
        executable: &mut invalid,
    }];
    let error = benchmark_prepared(
        &mut targets,
        &BenchmarkConfig {
            warmups: 1,
            samples: 1,
            min_sample_ms: 1,
            measurement_order_seed: 7,
        },
    )
    .unwrap_err();
    assert!(matches!(error, ExecutionError::NonFiniteOutput(_)));
}

fn sequential_benchmark_run(
    seed: u64,
) -> (
    Vec<omeinsum::static_plan::ModeTiming>,
    Vec<(usize, usize)>,
    usize,
) {
    let targets = [
        SequentialBenchmarkTarget {
            representation: Representation::RealSkeleton,
            backend: "cpu",
            dtype: "f64",
            execution_mode: "sliced-host-f64-accumulated",
        },
        SequentialBenchmarkTarget {
            representation: Representation::Flat4M,
            backend: "cpu",
            dtype: "f64",
            execution_mode: "sliced-host-f64-accumulated",
        },
        SequentialBenchmarkTarget {
            representation: Representation::RealifiedRank3,
            backend: "cpu",
            dtype: "f64",
            execution_mode: "sliced-host-f64-accumulated",
        },
    ];
    let seconds_per_contraction = [0.00025, 0.0005, 0.002];
    let mut calls = Vec::new();
    let mut live = 0usize;
    let mut max_live = 0usize;
    let timings = benchmark_sequential_resident(
        &targets,
        &BenchmarkConfig {
            warmups: 3,
            samples: 5,
            min_sample_ms: 1,
            measurement_order_seed: seed,
        },
        |target_index, inner_iterations| {
            live += 1;
            max_live = max_live.max(live);
            calls.push((target_index, inner_iterations));
            let elapsed = seconds_per_contraction[target_index] * inner_iterations as f64;
            let output = ComplexValue {
                re: target_index as f64 + 1.0,
                im: -(target_index as f64),
            };
            live -= 1;
            Ok((elapsed, output))
        },
    )
    .unwrap();
    (timings, calls, max_live)
}

#[test]
fn sequential_resident_benchmark_is_seeded_single_live_and_statistically_complete() {
    let (timings, calls, max_live) = sequential_benchmark_run(20260723);
    let (_, repeated_calls, repeated_max_live) = sequential_benchmark_run(20260723);
    let (_, other_calls, _) = sequential_benchmark_run(20260724);

    assert_eq!(max_live, 1);
    assert_eq!(repeated_max_live, 1);
    assert_eq!(calls, repeated_calls);
    assert_ne!(
        &calls[calls.len() - 15..],
        &other_calls[other_calls.len() - 15..]
    );
    assert_eq!(
        timings
            .iter()
            .map(|timing| timing.inner_iterations)
            .collect::<Vec<_>>(),
        vec![4, 2, 1]
    );
    assert_eq!(timings.len(), 3);
    for (index, timing) in timings.iter().enumerate() {
        assert_eq!(timing.raw_seconds_per_contraction.len(), 5);
        assert_eq!(
            timing.raw_seconds_per_contraction,
            vec![[0.00025, 0.0005, 0.002][index]; 5]
        );
        assert_eq!(timing.best_seconds, [0.00025, 0.0005, 0.002][index]);
        assert_eq!(timing.median_seconds, [0.00025, 0.0005, 0.002][index]);
        assert_eq!(timing.iqr_seconds, 0.0);
        assert_eq!(
            timing.output,
            ComplexValue {
                re: index as f64 + 1.0,
                im: -(index as f64),
            }
        );
    }
    for round in calls[calls.len() - 15..].chunks_exact(3) {
        let mut order = round
            .iter()
            .map(|(target_index, _)| *target_index)
            .collect::<Vec<_>>();
        order.sort_unstable();
        assert_eq!(order, vec![0, 1, 2]);
    }
}

#[test]
fn arena_forked_tree_keeps_overlapping_intermediates_disjoint() {
    let ranges = vec![
        LiveRange::new(ValueId(0), 0, 0, 64, 256),
        LiveRange::new(ValueId(1), 0, 0, 64, 256),
        LiveRange::new(ValueId(2), 0, 1, 64, 256),
        LiveRange::new(ValueId(3), 0, 1, 64, 256),
        LiveRange::new(ValueId(4), 0, 2, 64, 256),
        LiveRange::new(ValueId(5), 1, 2, 64, 256),
        LiveRange::new(ValueId(6), 2, 2, 64, 256),
    ];
    let plan = allocate_live_ranges(&ranges).unwrap();
    assert_eq!(
        plan.slots,
        vec![
            ArenaSlot::new(ValueId(0), 0, 64),
            ArenaSlot::new(ValueId(1), 256, 64),
            ArenaSlot::new(ValueId(2), 512, 64),
            ArenaSlot::new(ValueId(3), 768, 64),
            ArenaSlot::new(ValueId(4), 1024, 64),
            ArenaSlot::new(ValueId(5), 0, 64),
            ArenaSlot::new(ValueId(6), 256, 64),
        ]
    );
    assert_eq!(plan.arena_bytes, 1280);
    assert_eq!(plan.semantic_peak_bytes, 320);
}

#[test]
fn arena_chain_reuses_expired_storage_at_the_lowest_offset() {
    let ranges = vec![
        LiveRange::new(ValueId(0), 0, 0, 64, 256),
        LiveRange::new(ValueId(1), 0, 0, 64, 256),
        LiveRange::new(ValueId(2), 0, 1, 64, 256),
        LiveRange::new(ValueId(3), 0, 1, 64, 256),
        LiveRange::new(ValueId(4), 1, 2, 64, 256),
        LiveRange::new(ValueId(5), 2, 2, 64, 256),
    ];
    let first = allocate_live_ranges(&ranges).unwrap();
    let second = allocate_live_ranges(&ranges).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.slots,
        vec![
            ArenaSlot::new(ValueId(0), 0, 64),
            ArenaSlot::new(ValueId(1), 256, 64),
            ArenaSlot::new(ValueId(2), 512, 64),
            ArenaSlot::new(ValueId(3), 768, 64),
            ArenaSlot::new(ValueId(4), 0, 64),
            ArenaSlot::new(ValueId(5), 256, 64),
        ]
    );
    assert_eq!(first.arena_bytes, 1024);
}

#[test]
fn arena_random_live_ranges_are_aligned_in_bounds_and_non_aliasing() {
    for seed in 0..100u64 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let count = rng.random_range(1..32);
        let ranges = (0..count)
            .map(|value| {
                let start = rng.random_range(0..12);
                let end = rng.random_range(start..12);
                let alignment = [64, 128, 256, 512][rng.random_range(0..4)];
                LiveRange::new(
                    ValueId(value),
                    start,
                    end,
                    rng.random_range(1..1500),
                    alignment,
                )
            })
            .collect::<Vec<_>>();
        let first = allocate_live_ranges(&ranges).unwrap();
        let second = allocate_live_ranges(&ranges).unwrap();
        assert_eq!(first, second, "seed {seed}");
        for (range, slot) in ranges.iter().zip(&first.slots) {
            assert_eq!(range.value, slot.value);
            assert_eq!(slot.offset % range.alignment, 0, "seed {seed}");
            assert!(
                slot.offset.checked_add(slot.bytes).unwrap() <= first.arena_bytes,
                "seed {seed}"
            );
        }
        for left in 0..ranges.len() {
            for right in left + 1..ranges.len() {
                let time_overlaps = ranges[left].start <= ranges[right].end
                    && ranges[right].start <= ranges[left].end;
                if !time_overlaps {
                    continue;
                }
                let left_slot = &first.slots[left];
                let right_slot = &first.slots[right];
                let space_disjoint = left_slot.offset + left_slot.bytes <= right_slot.offset
                    || right_slot.offset + right_slot.bytes <= left_slot.offset;
                assert!(space_disjoint, "seed {seed}, values {left}/{right}");
            }
        }
    }
}

#[test]
fn arena_plans_every_static_f32_semantic_plane() {
    let bundle = build_plan_bundle(&two_leaf_scalar_network(0.25, 0.0), 1e-12).unwrap();
    for plan in [
        &bundle.real_skeleton,
        &bundle.flat_4m,
        &bundle.realified_rank3,
    ] {
        let arena = plan_f32_arena(plan).unwrap();
        assert_eq!(arena.slots.len(), plan.values.len());
        assert!(arena.slots.iter().all(|slot| slot.offset % 256 == 0));
        assert!(arena.semantic_peak_bytes <= arena.arena_bytes);
    }
}

#[test]
fn arena_keeps_uploaded_leaves_disjoint_for_repeatable_execution() {
    let bundle = build_plan_bundle(&matrix_scalar_network(0.25, -0.5), 1e-12).unwrap();
    for plan in [
        &bundle.real_skeleton,
        &bundle.flat_4m,
        &bundle.realified_rank3,
    ] {
        let arena = plan_f32_arena(plan).unwrap();
        for leaf in &arena.slots[..plan.leaf_values.len()] {
            for computed in &arena.slots[plan.leaf_values.len()..] {
                let disjoint = leaf.offset + leaf.bytes <= computed.offset
                    || computed.offset + computed.bytes <= leaf.offset;
                assert!(
                    disjoint,
                    "{:?} leaf {} aliases computed value {} across repeated enqueues",
                    plan.representation, leaf.value.0, computed.value.0
                );
            }
        }
    }
}

#[test]
fn ascend_lowering_real_and_rides_preserve_green_plane_batching() {
    let real = build_plan_bundle(&two_leaf_scalar_network(0.0, 0.0), 1e-12).unwrap();
    let real_trace = lower_plan_traces(&real.real_skeleton).unwrap();
    assert_eq!(real_trace[0].kind, KernelKind::RealReal);
    assert_eq!(real_trace[0].matmul_calls, 1);
    assert_eq!(real_trace[0].green_batch, 1);
    assert_eq!(
        real_trace[0].real_matmul_volume,
        real.real_skeleton.nodes[0].real_skeleton_volume
    );

    let ride_left = build_plan_bundle(&two_leaf_scalar_network(0.25, 0.0), 1e-12).unwrap();
    let left_trace = lower_plan_traces(&ride_left.realified_rank3).unwrap();
    assert_eq!(left_trace[0].kind, KernelKind::RideLeft);
    assert_eq!(left_trace[0].matmul_calls, 1);
    assert_eq!(left_trace[0].green_batch, 2);
    assert_eq!(
        left_trace[0].real_matmul_volume,
        2 * ride_left.realified_rank3.nodes[0].real_skeleton_volume
    );
    assert_eq!(green_operand_plane_batches(KernelKind::RideLeft), (2, 1));

    let ride_right = build_plan_bundle(&two_leaf_scalar_network(0.0, 0.25), 1e-12).unwrap();
    let right_trace = lower_plan_traces(&ride_right.realified_rank3).unwrap();
    assert_eq!(right_trace[0].kind, KernelKind::RideRight);
    assert_eq!(right_trace[0].matmul_calls, 1);
    assert_eq!(right_trace[0].green_batch, 2);
    assert_eq!(green_operand_plane_batches(KernelKind::RideRight), (1, 2));
}

#[test]
fn ascend_lowering_rank3_and_flat_have_exact_real_operator_counts() {
    let bundle = build_plan_bundle(&two_leaf_scalar_network(0.25, -0.5), 1e-12).unwrap();
    let merge = &bundle.realified_rank3.nodes[0];
    let trace = lower_plan_traces(&bundle.realified_rank3).unwrap();
    assert_eq!(trace[0].kind, KernelKind::Merge3M);
    assert_eq!(trace[0].matmul_calls, 3);
    assert_eq!(trace[0].elementwise_calls, 5);
    assert_eq!(trace[0].green_batch, 1);
    assert_eq!(trace[0].real_matmul_volume, 3 * merge.real_skeleton_volume);
    assert!(!merge
        .scratch
        .iter()
        .any(|scratch| scratch.role == ScratchRole::Product4));

    let flat = &bundle.flat_4m.nodes[0];
    let trace = lower_plan_traces(&bundle.flat_4m).unwrap();
    assert_eq!(trace[0].kind, KernelKind::Flat4M);
    assert_eq!(trace[0].matmul_calls, 4);
    assert_eq!(trace[0].elementwise_calls, 2);
    assert_eq!(trace[0].real_matmul_volume, 4 * flat.real_skeleton_volume);
    assert_eq!(
        flat.scratch
            .iter()
            .map(|scratch| scratch.role.clone())
            .collect::<Vec<_>>(),
        vec![
            ScratchRole::Product1,
            ScratchRole::Product2,
            ScratchRole::Product3,
            ScratchRole::Product4,
        ]
    );
}

#[test]
fn ascend_lowering_preserves_the_single_permute_fast_path() {
    let dimensions = (0..10).map(|mode| (mode, 2)).collect::<Vec<_>>();
    let current = (0..10).collect::<Vec<_>>();
    let desired = [6, 7, 8, 9, 2, 3, 4, 5, 0, 1];
    let collapsed = coalesce_permutation(&current, &desired, &dimensions, 8).unwrap();
    assert_eq!(collapsed.input_shape, vec![4, 16, 16]);
    assert_eq!(collapsed.axes, vec![2, 1, 0]);
    assert_eq!(collapsed.output_shape, vec![16, 16, 4]);

    let steps = decompose_permutation(&current, &desired, &dimensions, 8).unwrap();
    assert_eq!(steps, vec![collapsed]);
}

#[test]
fn ascend_lowering_decomposes_an_over_rank_permutation_into_bounded_block_moves() {
    let dimensions = (0..12).map(|mode| (mode, 2)).collect::<Vec<_>>();
    let current = (0..12).collect::<Vec<_>>();
    let desired = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11];

    let direct_error = coalesce_permutation(&current, &desired, &dimensions, 8).unwrap_err();
    assert!(matches!(direct_error, ExecutionError::Unsupported(_)));

    let steps = decompose_permutation(&current, &desired, &dimensions, 8).unwrap();
    assert!(steps.len() > 1);
    assert!(steps.iter().all(|step| {
        step.input_shape.len() <= 4
            && step.axes.len() == step.input_shape.len()
            && step.output_shape.len() == step.input_shape.len()
            && step.input_shape.iter().product::<usize>()
                == step.output_shape.iter().product::<usize>()
    }));
    assert_eq!(
        replay_permutation_modes(&current, &dimensions, &steps),
        desired
    );
}

#[test]
fn ascend_lowering_decomposition_replays_seeded_arbitrary_mode_orders() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x0A5C_E910);
    for rank in 2..=16 {
        let dimensions = (0..rank).map(|mode| (mode, 2)).collect::<Vec<_>>();
        let current = (0..rank).collect::<Vec<_>>();
        for _ in 0..32 {
            let mut desired = current.clone();
            for index in (1..desired.len()).rev() {
                desired.swap(index, rng.random_range(0..=index));
            }
            let steps = decompose_permutation(&current, &desired, &dimensions, 7).unwrap();
            assert!(steps.iter().all(|step| step.input_shape.len() <= 7));
            assert_eq!(
                replay_permutation_modes(&current, &dimensions, &steps),
                desired
            );
        }
    }
}

#[test]
fn ascend_lowering_omits_identity_and_singleton_axes_from_decomposition() {
    let dimensions = vec![(0, 2), (1, 1), (2, 3), (3, 1), (4, 5)];
    let identity =
        decompose_permutation(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4], &dimensions, 8).unwrap();
    assert!(identity.is_empty());

    let steps = decompose_permutation(&[0, 1, 2, 3, 4], &[4, 3, 2, 1, 0], &dimensions, 8).unwrap();
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].input_shape, vec![2, 3, 5]);
    assert_eq!(steps[0].axes, vec![2, 1, 0]);
    assert_eq!(steps[0].output_shape, vec![5, 3, 2]);
}

fn replay_permutation_modes(
    initial_modes: &[i32],
    dimensions: &[(i32, usize)],
    steps: &[omeinsum::static_plan::CoalescedPermutation],
) -> Vec<i32> {
    let sizes = dimensions
        .iter()
        .copied()
        .collect::<std::collections::HashMap<_, _>>();
    let mut modes = initial_modes
        .iter()
        .copied()
        .filter(|mode| sizes[mode] != 1)
        .collect::<Vec<_>>();
    for step in steps {
        let mut groups = Vec::with_capacity(step.input_shape.len());
        let mut cursor = 0usize;
        for expected_size in &step.input_shape {
            let mut product = 1usize;
            let start = cursor;
            while product < *expected_size {
                product *= sizes[&modes[cursor]];
                cursor += 1;
            }
            assert_eq!(product, *expected_size);
            groups.push(modes[start..cursor].to_vec());
        }
        assert_eq!(cursor, modes.len());
        modes = step
            .axes
            .iter()
            .flat_map(|axis| groups[*axis as usize].iter().copied())
            .collect();
    }
    modes
}

#[test]
fn ascend_lowering_rejects_mismatched_permutation_modes() {
    let dimensions = (0..10).map(|mode| (mode, 2)).collect::<Vec<_>>();
    let rank_error =
        decompose_permutation(&[0, 1], &[0], &dimensions, 8).expect_err("rank change must fail");
    assert!(matches!(rank_error, ExecutionError::Unsupported(_)));

    let error = decompose_permutation(
        &(0..10).collect::<Vec<_>>(),
        &[0, 2, 4, 6, 8, 1, 3, 5, 7, 99],
        &dimensions,
        8,
    )
    .unwrap_err();
    assert!(matches!(error, ExecutionError::Unsupported(_)));
}
