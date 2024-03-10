# SGPU-Compute
**S**imple **GPU**-Compute using WebGPU

This crate aims to provide a simple and easy-to-use interface to run compute shaders with WGPU and WGSL. It is designed to be as simple as possible to use, while still providing a lot of flexibility for performance reason.

## Quickstart
```Rust
use sgpu_compute::prelude::*;

let my_shader = "
   @group(0) @binding(0) var<uniform> coefficient: u32;
   @group(0) @binding(1) var<storage, read> in: array<u32>;
   @group(0) @binding(2) var<storage, read_write> out: array<u32>;
   @compute
   @workgroup_size(8, 1, 1)
   fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
       out[global_id.x] = coefficient * in[global_id.x];
   }
";

const N_ELEMENT: usize = 64;
const WORKGROUP_SIZE: usize = 8;
const N_WORKGROUP: u32 = (N_ELEMENT / WORKGROUP_SIZE) as u32;

let gpu = GpuCompute::new();
let mut pipeline = gpu.gen_pipeline(
       None,
       [StageDesc {
           name: Some("norm"),
           shader: my_shader,
           entrypoint: "main",
       }],
   );

const COEFFICIENT: u32 = 42;
let input: [u32; N_ELEMENT] = std::array::from_fn(|i| i as u32);
pipeline.write_uniform(&COEFFICIENT);
let result_gpu = pipeline.run(&input, [(N_WORKGROUP, 1, 1)], |vals: &[u32; N_ELEMENT]| *vals);
let result_cpu = input.map(|v| v * COEFFICIENT);
assert_eq!(result_gpu, result_cpu);
```

## Features
- Quick setup for using WGPU for computing
- Blocking and async API are available
- Multi-stage shader are possible

## Examples
Example are provided inside the examples directory
