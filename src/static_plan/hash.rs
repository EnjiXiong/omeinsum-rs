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

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    use super::hash_serializable;

    #[derive(Debug, Serialize, Deserialize)]
    struct FloatingCost {
        predicted_arithmetic_overhead: f64,
    }

    #[test]
    fn hash_floats_survive_json_round_trip_bit_exactly() {
        let original = FloatingCost {
            predicted_arithmetic_overhead: 1026.0 / 454.0,
        };
        let json = serde_json::to_string(&original).unwrap();
        let round_trip: FloatingCost = serde_json::from_str(&json).unwrap();

        assert_eq!(
            original.predicted_arithmetic_overhead.to_bits(),
            round_trip.predicted_arithmetic_overhead.to_bits(),
            "JSON {json} changed the f64 used by static-plan hashing"
        );
        assert_eq!(
            hash_serializable(&original).unwrap(),
            hash_serializable(&round_trip).unwrap()
        );
    }
}
