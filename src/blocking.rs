use crate::*;
use std::ops::{Deref, DerefMut};

/// This is a blocking version of `GpuComputeAsync`. It is enabled by the `blocking` feature. This feature is enabled by default.
pub struct GpuCompute(GpuComputeAsync);

impl GpuCompute {
    #[inline]
    pub fn new() -> Self {
        Self(pollster::block_on(GpuComputeAsync::new()))
    }

    #[inline]
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
        Pipeline(pollster::block_on(
            self.0.gen_pipeline(scratchpad_size, stages),
        ))
    }
}

pub struct Pipeline<
    'a,
    Input: bytemuck::Pod,
    Uniform: bytemuck::Pod,
    Output: bytemuck::Pod,
    const N: usize,
>(PipelineAsync<'a, Input, Uniform, Output, N>);

impl<'a, Input: bytemuck::Pod, Uniform: bytemuck::Pod, Output: bytemuck::Pod, const N: usize>
    Pipeline<'a, Input, Uniform, Output, N>
{
    #[inline]
    pub fn run<T: Send + 'static>(
        &mut self,
        input: &Input,
        workgroups: [(u32, u32, u32); N],
        callback: impl FnOnce(&Output) -> T + Send,
    ) -> T {
        pollster::block_on(self.0.run(input, workgroups, callback))
    }
}

impl<'a, Input: bytemuck::Pod, Uniform: bytemuck::Pod, Output: bytemuck::Pod, const N: usize> Deref
    for Pipeline<'a, Input, Uniform, Output, N>
{
    type Target = PipelineAsync<'a, Input, Uniform, Output, N>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, Input: bytemuck::Pod, Uniform: bytemuck::Pod, Output: bytemuck::Pod, const N: usize>
    DerefMut for Pipeline<'a, Input, Uniform, Output, N>
{
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
