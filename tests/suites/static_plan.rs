use omeinsum::static_plan::{
    build_geometry_plan, build_plan_bundle, BinaryContractionTree, ComplexNetwork, ComplexTensor,
    InputSet, InputTensor, KernelKind, LeafClass, PlanBundle, PlanStats, Plane, Representation,
    ScratchRole, StaticPlan, TensorSpec, ValueId, ValueSpec,
};

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
