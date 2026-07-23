use serde::Serialize;
use sha2::{Digest, Sha256};

use super::{BinaryContractionTree, PlanError, StaticPlan};

pub(crate) fn tree_hash(tree: &BinaryContractionTree) -> Result<String, PlanError> {
    hash_serializable(tree)
}

pub(crate) fn plan_hash(plan: &StaticPlan) -> Result<String, PlanError> {
    #[derive(Serialize)]
    struct PlanHashView<'a> {
        representation: &'a super::Representation,
        tree_hash: &'a str,
        leaf_values: &'a [super::ValueId],
        values: &'a [super::ValueSpec],
        nodes: &'a [super::PlanNode],
        output: super::ValueId,
        stats: &'a super::PlanStats,
    }

    hash_serializable(&PlanHashView {
        representation: &plan.representation,
        tree_hash: &plan.tree_hash,
        leaf_values: &plan.leaf_values,
        values: &plan.values,
        nodes: &plan.nodes,
        output: plan.output,
        stats: &plan.stats,
    })
}

fn hash_serializable<T: Serialize + ?Sized>(value: &T) -> Result<String, PlanError> {
    let encoded = serde_json::to_vec(value)
        .map_err(|error| PlanError::Hash(format!("canonical serialization failed: {error}")))?;
    Ok(hex::encode(Sha256::digest(encoded)))
}

impl StaticPlan {
    pub fn recompute_plan_hash(&self) -> Result<String, PlanError> {
        plan_hash(self)
    }
}
