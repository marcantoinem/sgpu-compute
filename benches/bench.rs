use crate::normal_distribution::numerical_integration_cpu;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sgpu_compute::prelude::*;

#[path = "../tests/normal_distribution.rs"] // Forbidden hack to avoid code duplication
mod normal_distribution;

fn normal_distribution_benchmark(c: &mut Criterion) {
    let gpu_compute = GpuCompute::new();
    let mut pipeline = gpu_compute.gen_pipeline::<[f32; 1000], u32, [f32; 1000], 1>(
        None,
        [StageDesc {
            name: Some("norm"),
            shader: include_str!("../examples/normal_distribution.wgsl"),
            entrypoint: "main",
        }],
    );
    const N: u32 = 1000;
    const WORKGROUP_SIZE: u32 = 10;
    const N_WORKGROUP: u32 = N / WORKGROUP_SIZE;
    pipeline.write_uniform(&32768);
    let input = std::array::from_fn(|i| i as f32 / 300.0);
    c.bench_function("test normal distribution GPU", |b| {
        b.iter(|| pipeline.run(black_box(&input), [(N_WORKGROUP, 1, 1)], |vals| *vals))
    });
    c.bench_function("test normal distribution CPU", |b| {
        b.iter(|| numerical_integration_cpu(black_box(&input)))
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    normal_distribution_benchmark(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
