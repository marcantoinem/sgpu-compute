# SGPU-Compute
**S**imple **GPU**-Compute using WebGPU

## Quickstart
```Rust
use sgpu_compute::blocking::GpuCompute;

fn main() {
    let gpu = GpuCompute::new(include_str!("your_shader.wgsl"));
    let your_input = ...;
    let result: Vec<f32> = gpu.compute(&your_input);
    println!("{:?}", result);
}
```

```Rust
@group(0)
@binding(0)
var<storage, read_write> v_indices: array<f32>; // this is used as both input and output for convenience

fn your_function(x: f32) -> f32 {
    return x * 2.0;
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    v_indices[global_id.x] = your_function(v_indices[global_id.x]);
}
```

## Features
- Quick setup for using WGPU for computing
- Blocking and async API are available

## TODO
- Create more examples
- Add more flexibility for buffer utilization
- Add option for custom workgroup size
- Documentate the entire crate
- Publish this crate

## Examples
Example are provided inside the examples directory
