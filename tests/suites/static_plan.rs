use omeinsum::static_plan::{
    allocate_live_ranges, benchmark_prepared, build_geometry_plan, build_plan_bundle,
    contract_complex64, plan_f32_arena, prepare_cpu_f32, prepare_cpu_f64, ArenaSlot,
    BenchmarkConfig, BenchmarkTarget, BinaryContractionTree, ComplexNetwork, ComplexTensor,
    ComplexValue, ExecutionError, InputSet, InputTensor, KernelKind, LeafClass, LiveRange,
    PlanBundle, PlanStats, Plane, PreparedExecutable, Representation, ScratchRole, StaticPlan,
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

#[test]
fn plan_bundle_round_trips_json() {
    let bundle = PlanBundle {
        format: "omeinsum-static-plan-v1".to_string(),
        realness_tol: 1e-12,
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
