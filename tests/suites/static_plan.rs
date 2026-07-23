use omeinsum::static_plan::{
    build_geometry_plan, BinaryContractionTree, ComplexNetwork, ComplexTensor, InputSet,
    InputTensor, LeafClass, PlanBundle, PlanStats, Plane, Representation, StaticPlan, TensorSpec,
    ValueId, ValueSpec,
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
