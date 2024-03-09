use crate::normal_distribution::numerical_integration_cpu;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sgpu_compute::prelude::*;

#[path = "../tests/normal_distribution.rs"] // Forbidden hack to avoid code duplication
mod normal_distribution;

fn criterion_benchmark(c: &mut Criterion) {
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
    let input: [f32; N as usize] = (0..N)
        .into_iter()
        .map(|i| i as f32 / 300.0)
        .collect::<Vec<_>>()
        .try_into()
        .expect("Could not convert vec");
    c.bench_function("test normal distribution GPU", |b| {
        b.iter(|| {
            let mut output = [0.0; 1000];
            pipeline.run(black_box(&input), [(N_WORKGROUP, 1, 1)], |vals| {
                output.copy_from_slice(vals)
            })
        })
    });
    c.bench_function("test normal distribution CPU", |b| {
        b.iter(|| numerical_integration_cpu(black_box(&input)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
