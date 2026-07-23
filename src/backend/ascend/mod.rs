//! Native AscendCL/ACLNN execution for frozen static plans.
//!
//! This module is compiled only with the `ascend` Cargo feature.

#![allow(clippy::result_large_err)] // Frozen backend diagnostics retain full context.

pub mod context;
mod error;
pub(crate) mod executable;
pub(crate) mod ffi;
pub(crate) mod operator;
pub mod storage;

use serde::{Deserialize, Serialize};

use crate::static_plan::{ExecutionError, InputSet, Representation, StaticPlan};

use context::Context;
pub use operator::{run_matmul_smoke, AscendMatmulSmokeReport};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AscendPrecisionMode {
    KeepDtype,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AscendExecutionMode {
    RepeatableAclnn,
    CapturedModel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AscendSessionConfig {
    pub device_id: i32,
    pub precision_mode: AscendPrecisionMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AscendExecutableConfig {
    pub execution_mode: AscendExecutionMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AscendMemoryStats {
    pub semantic_bytes: u64,
    pub device_arena_bytes: u64,
    pub scratch_bytes: u64,
    pub workspace_bytes: u64,
    pub peak_device_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AscendDeviceInfo {
    pub device_id: i32,
    pub soc_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CaptureStatus {
    NotRequested,
    Ready,
    SkippedUnsupported { reason: String },
}

pub struct AscendSession {
    pub(crate) context: Context,
    pub(crate) device_info: AscendDeviceInfo,
}

pub struct AscendExecutable<'session> {
    pub(crate) session: &'session AscendSession,
    pub(crate) representation: Representation,
    pub(crate) state: executable::ExecutableState<'session>,
    pub(crate) memory_stats: AscendMemoryStats,
    pub(crate) capture_status: CaptureStatus,
}

impl AscendSession {
    pub fn new(config: &AscendSessionConfig) -> Result<Self, ExecutionError> {
        if config.precision_mode != AscendPrecisionMode::KeepDtype {
            return Err(ExecutionError::Unsupported(
                "Ascend supports only keep-dtype precision".to_string(),
            ));
        }
        let context = Context::new(config.device_id)?;
        let device_info = AscendDeviceInfo {
            device_id: context.device_id(),
            soc_name: context.soc_name().to_string(),
        };
        Ok(Self {
            context,
            device_info,
        })
    }

    pub fn device_info(&self) -> &AscendDeviceInfo {
        &self.device_info
    }

    #[doc(hidden)]
    pub fn round_trip_f32(&self, values: &[f32]) -> Result<Vec<f32>, ExecutionError> {
        let bytes = u64::try_from(std::mem::size_of_val(values))
            .map_err(|_| ExecutionError::Unsupported("host input is too large".to_string()))?;
        let buffer = self.context.allocate(bytes)?;
        buffer.copy_h2d(0, values)?;
        self.context.synchronize()?;
        let mut output = vec![0.0f32; values.len()];
        buffer.copy_d2h(0, &mut output)?;
        self.context.synchronize()?;
        Ok(output)
    }
}

impl<'session> AscendExecutable<'session> {
    pub fn prepare(
        session: &'session AscendSession,
        plan: &StaticPlan,
        inputs: &InputSet<f64>,
        config: &AscendExecutableConfig,
    ) -> Result<Self, ExecutionError> {
        if config.execution_mode != AscendExecutionMode::RepeatableAclnn {
            return Err(ExecutionError::Unsupported(
                "captured Ascend execution is not enabled yet".to_string(),
            ));
        }
        let (state, memory_stats) = executable::ExecutableState::prepare(session, plan, inputs)?;
        Ok(Self {
            session,
            representation: plan.representation.clone(),
            state,
            memory_stats,
            capture_status: CaptureStatus::NotRequested,
        })
    }

    pub fn memory_stats(&self) -> &AscendMemoryStats {
        &self.memory_stats
    }

    pub fn capture_status(&self) -> &CaptureStatus {
        &self.capture_status
    }

    pub fn device_info(&self) -> &AscendDeviceInfo {
        self.session.device_info()
    }
}
