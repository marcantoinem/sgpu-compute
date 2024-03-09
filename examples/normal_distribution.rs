use sgpu_compute::{blocking::GpuCompute, StageDesc};

fn main() {
    let gpu = GpuCompute::new();
    let mut pipeline = gpu.gen_pipeline::<[f32; 10], u32, [f32; 10], 1>(
        None,
        [StageDesc {
            name: Some("norm"),
            shader: include_str!("normal_distribution.wgsl"),
            entrypoint: "main",
        }],
    );
    pipeline.write_uniform(&32768);
    let result = pipeline.run_blocking(
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [(10, 1, 1)],
        |vals| *vals,
    );
    println!("{:?}", result);
}
