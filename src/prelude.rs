//! Common re-exports for the crate.

#[cfg(feature = "blocking")]
pub use crate::blocking::GpuCompute;

pub use crate::GpuComputeAsync;

pub use crate::StageDesc;
/// This re-exports is needed for giving the scratchpad size.
pub use std::num::NonZeroUsize;
