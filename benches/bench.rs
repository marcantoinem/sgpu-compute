use std::f32::consts::PI;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sgpu_compute::blocking::GpuCompute;

fn normal_distribution(x: f32) -> f32 {
    (-0.5 * (x.powi(2))).exp() / (2.0 * PI).sqrt()
}

fn numerical_integration_cpu_single(born: f32) -> f32 {
    let mut area = 0.0;
    let width = born / 16384.0;
    for i in 0..16384 {
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

fn criterion_benchmark(c: &mut Criterion) {
    let gpu_compute = GpuCompute::new(include_str!("normal_distribution.wgsl"));
    let input: Vec<_> = (0..1000).into_iter().map(|i| i as f32 / 300.0).collect();
    c.bench_function("test normal distribution GPU", |b| {
        b.iter(|| gpu_compute.compute(black_box(&input)))
    });
    c.bench_function("test normal distribution CPU", |b| {
        b.iter(|| numerical_integration_cpu(black_box(&input)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
