use std::array;

use crate::normal_distribution::numerical_integration_cpu;
use sgpu_compute::prelude::*;

pub mod normal_distribution;

#[test]
fn normal_distribution_compare() {
    const N: u32 = 1000;
    const WORKGROUP_SIZE: u32 = 10;
    const N_WORKGROUP: u32 = N / WORKGROUP_SIZE;
    let gpu_compute = GpuCompute::new();
    let mut pipeline = gpu_compute.gen_pipeline::<[f32; N as usize], u32, [f32; N as usize], 1>(
        None,
        [StageDesc {
            name: Some("norm"),
            shader: include_str!("../examples/normal_distribution.wgsl"),
            entrypoint: "main",
        }],
    );
    pipeline.write_uniform(&32768);
    let input: [f32; N as usize] = array::from_fn(|i| i as f32 / 300.0);
    let cpu = numerical_integration_cpu(&input);
    pipeline.run(&input, [(N_WORKGROUP, 1, 1)], |outputs| {
        assert_eq!(cpu.len(), outputs.len());
        for (a, b) in outputs.iter().zip(cpu) {
            assert!((a - b).abs() < 0.00001, "{} != {}", a, b);
        }
    })
}
