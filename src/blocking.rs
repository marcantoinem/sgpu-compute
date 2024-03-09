use crate::*;

pub struct GpuCompute(GpuComputeAsync);

impl GpuCompute {
    pub fn new() -> Self {
        Self(pollster::block_on(GpuComputeAsync::new()))
    }

    pub fn gen_pipeline<
        Input: bytemuck::Pod,
        Uniform: bytemuck::Pod,
        Output: bytemuck::Pod,
        const N: usize,
    >(
        &self,
        scratchpad_size: Option<NonZeroUsize>,
        stages: [StageDesc; N],
    ) -> Pipeline<Input, Uniform, Output, N> {
        pollster::block_on(self.0.gen_pipeline(scratchpad_size, stages))
    }
}
