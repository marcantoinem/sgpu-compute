use std::f32::consts::PI;

use sgpu_compute::{blocking::GpuCompute, StageDesc};

fn normal_distribution(x: f32) -> f32 {
    (-0.5 * (x.powi(2))).exp() / (2.0 * PI).sqrt()
}

fn numerical_integration_cpu_single(born: f32) -> f32 {
    let mut area = 0.0;
    let width = born / 32768.0;
    for i in 0..32768 {
        let x = i as f32 * width + 0.5 * width;
        area += width * normal_distribution(x);
    }
    0.5 + area
}

pub fn numerical_integration_cpu(to_integrate: &[f32]) -> Vec<f32> {
    to_integrate
        .iter()
        .map(|&x| numerical_integration_cpu_single(x))
        .collect()
}

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
    let input: [f32; N as usize] = (0..N)
        .into_iter()
        .map(|i| i as f32 / 300.0)
        .collect::<Vec<_>>()
        .try_into()
        .expect("Could not convert vec");
    let cpu = numerical_integration_cpu(&input);
    pipeline.run_blocking(&input, [(N_WORKGROUP, 1, 1)], |outputs| {
        assert_eq!(cpu.len(), outputs.len());
        for (a, b) in outputs.iter().zip(cpu) {
            assert!((a - b).abs() < 0.00001, "{} != {}", a, b);
        }
    })
}
