use sgpu_compute::prelude::*;

fn main() {
    let gpu = GpuCompute::new();
    let mut pipeline = gpu.gen_pipeline(
        None,
        [StageDesc {
            name: Some("norm"),
            shader: include_str!("normal_distribution.wgsl"),
            entrypoint: "main",
        }],
    );
    let input: [f32; 100] = std::array::from_fn(|i| i as f32 / 100.0);
    pipeline.write_uniform(&32768);
    let result: [f32; 100] = pipeline.run(&input, [(10, 1, 1)], |vals| *vals);
    println!("{:?}", result);
}
