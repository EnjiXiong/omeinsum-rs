mod arena;
mod ascend_lowering;
mod benchmark;
mod builder;
mod cpu;
mod error;
mod hash;
mod leaf_preprocessing;
mod slicing;
mod types;

pub use arena::{allocate_live_ranges, plan_f32_arena, ArenaPlan, ArenaSlot, LiveRange};
pub use ascend_lowering::{
    coalesce_permutation, decompose_permutation, green_operand_plane_batches, lower_plan_traces,
    CoalescedPermutation, LoweredNodeTrace,
};
pub use benchmark::*;
pub use builder::{build_geometry_plan, build_plan_bundle, build_plan_bundle_with_preprocessing};
pub use cpu::{contract_complex64, prepare_cpu_f32, prepare_cpu_f64};
pub use error::PlanError;
pub use slicing::{build_sliced_plan_bundle, gray_assignments, slice_inputs, SlicedExecutable};
pub use types::*;
