use crate::{primitive_number::PrimitiveNumber, GpuComputeAsync};

pub struct GpuCompute(GpuComputeAsync);

impl GpuCompute {
    pub fn new(shader: &str) -> Self {
        Self(pollster::block_on(GpuComputeAsync::new(shader)))
    }

    pub fn change_compute_shader(&mut self, shader: &str) {
        self.0.change_compute_shader(shader);
    }

    pub fn compute<A: PrimitiveNumber>(&self, inputs: &[A]) -> Vec<A> {
        pollster::block_on(self.0.compute(inputs))
    }
}
