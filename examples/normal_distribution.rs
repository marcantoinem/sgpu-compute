use sgpu_compute::blocking::GpuCompute;

fn main() {
    let gpu = GpuCompute::new(include_str!("normal_distribution.wgsl"));
    let result: Vec<f32> = gpu.compute(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]);
    println!("{:?}", result);
}
