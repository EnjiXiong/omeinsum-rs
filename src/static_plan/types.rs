use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Representation {
    #[serde(rename = "real-skeleton")]
    RealSkeleton,
    #[serde(rename = "flat-4m")]
    Flat4M,
    #[serde(rename = "realified-rank3")]
    RealifiedRank3,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum KernelKind {
    #[serde(rename = "real-real")]
    RealReal,
    #[serde(rename = "ride-left")]
    RideLeft,
    #[serde(rename = "ride-right")]
    RideRight,
    #[serde(rename = "merge-3m")]
    Merge3M,
    #[serde(rename = "flat-4m")]
    Flat4M,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Plane {
    Real,
    Imag,
    Sum,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LeafClass {
    Real,
    Complex,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SliceAssignmentOrder {
    BinaryReflectedGray,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SliceSpec {
    pub modes: Vec<i32>,
    pub dimensions: Vec<usize>,
    pub assignment_order: SliceAssignmentOrder,
    pub slice_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlicedVolume {
    pub representation: Representation,
    pub source_real_matmul_volume: u128,
    pub sliced_real_matmul_volume: u128,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlicedPlanBundle {
    pub format: String,
    pub source_tree_hash: String,
    pub source_plan_hashes: Vec<(Representation, String)>,
    pub source_inputs: InputSet<f64>,
    pub slice: SliceSpec,
    pub reduced: PlanBundle,
    pub aggregate_volumes: Vec<SlicedVolume>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LeafPreprocessing {
    #[default]
    Raw,
    PhaseCanonicalized,
}

impl LeafPreprocessing {
    pub fn is_raw(&self) -> bool {
        *self == Self::Raw
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ValueId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ScratchId(pub usize);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorSpec {
    pub modes: Vec<i32>,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputTensor<T> {
    pub spec: TensorSpec,
    pub real: Vec<T>,
    pub imag: Vec<T>,
    pub class: LeafClass,
    pub imag_max: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classification_imag_max: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputSet<T> {
    pub tensors: Vec<InputTensor<T>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InputUpdate<T> {
    pub index: usize,
    pub tensor: InputTensor<T>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComplexTensor<T> {
    pub spec: TensorSpec,
    pub real: Vec<T>,
    pub imag: Vec<T>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueSpec {
    pub id: ValueId,
    pub tensor: TensorSpec,
    pub planes: Vec<Plane>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScratchRole {
    LeftSum,
    RightSum,
    Product1,
    Product2,
    Product3,
    Product4,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScratchSpec {
    pub id: ScratchId,
    pub role: ScratchRole,
    pub elements: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContractionSpec {
    pub batch_modes: Vec<i32>,
    pub left_modes: Vec<i32>,
    pub right_modes: Vec<i32>,
    pub contracted_modes: Vec<i32>,
    pub left_trace_modes: Vec<i32>,
    pub right_trace_modes: Vec<i32>,
    pub batch: usize,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub left_permutation: Vec<usize>,
    pub right_permutation: Vec<usize>,
    pub output_permutation: Option<Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanNode {
    pub id: usize,
    pub left: ValueId,
    pub right: ValueId,
    pub output: ValueId,
    pub kind: KernelKind,
    pub contraction: ContractionSpec,
    pub scratch: Vec<ScratchSpec>,
    pub real_skeleton_volume: u128,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealificationCost {
    pub real_real_volume: u128,
    pub ride_volume: u128,
    pub merge_volume: u128,
    pub real_real_fraction: f64,
    pub ride_fraction: f64,
    pub merge_fraction: f64,
    pub predicted_arithmetic_overhead: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanStats {
    pub real_leaf_count: usize,
    pub complex_leaf_count: usize,
    pub real_real_nodes: usize,
    pub ride_left_nodes: usize,
    pub ride_right_nodes: usize,
    pub merge_3m_nodes: usize,
    pub flat_4m_nodes: usize,
    pub real_skeleton_volume: u128,
    pub real_matmul_volume: u128,
    pub realification_cost: Option<RealificationCost>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StaticPlan {
    pub representation: Representation,
    pub tree_hash: String,
    pub plan_hash: String,
    pub leaf_values: Vec<ValueId>,
    pub values: Vec<ValueSpec>,
    pub nodes: Vec<PlanNode>,
    pub output: ValueId,
    pub stats: PlanStats,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseCanonicalizationReport {
    pub source_real_leaf_count: usize,
    pub source_complex_leaf_count: usize,
    pub canonicalized_leaf_count: usize,
    pub phase_anchor: Option<usize>,
    pub accumulated_phase: ComplexValue,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanBundle {
    pub format: String,
    pub realness_tol: f64,
    #[serde(default, skip_serializing_if = "LeafPreprocessing::is_raw")]
    pub leaf_preprocessing: LeafPreprocessing,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_canonicalization: Option<PhaseCanonicalizationReport>,
    pub tree_hash: String,
    pub inputs: InputSet<f64>,
    pub real_skeleton: StaticPlan,
    pub flat_4m: StaticPlan,
    pub realified_rank3: StaticPlan,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ComplexValue {
    pub re: f64,
    pub im: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BinaryContractionTree {
    Leaf {
        tensor_index: usize,
    },
    Node {
        output_modes: Vec<i32>,
        left: Box<BinaryContractionTree>,
        right: Box<BinaryContractionTree>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComplexNetwork<T> {
    pub tensors: Vec<ComplexTensor<T>>,
    pub output_modes: Vec<i32>,
    pub size_dict: Vec<(i32, usize)>,
    pub tree: BinaryContractionTree,
}
