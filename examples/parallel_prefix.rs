use std::num::NonZeroUsize;

use rand::Rng;
use sgpu_compute::{blocking::GpuCompute, StageDesc};

#[derive(Debug, Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Uniform {
    width: u32,
}

fn gen<const N: usize>() -> [f32; N] {
    let mut out = [0.0; N];
    rand::thread_rng().fill(&mut out[..]);
    out
}

fn parallel_prefix<const N: usize>(input: &[f32; N]) -> [f32; N] {
    let mut out = [0.0; N];
    let mut tot = 0.0;
    for i in 0..N {
        tot += input[i];
        out[i] = tot;
    }
    return out;
}

fn main() {
    const PER_WORKER: u32 = 4;
    const N: usize = 3000;
    const N_WORKER: u32 = (N as u32 - 1) / PER_WORKER + 1;
    const N_WG: u32 = (N_WORKER - 1) / 16 + 1;
    const N_PADDED: usize = (N_WG * 16 * PER_WORKER) as _;

    let input: [f32; N] = gen();

    let gpu = GpuCompute::new();
    let mut pipeline = gpu.gen_pipeline::<_, Uniform, _, 3>(
        NonZeroUsize::new(std::mem::size_of::<f32>() * N_PADDED as usize / PER_WORKER as usize),
        [
            StageDesc {
                name: Some("first_pass"),
                shader: include_str!("parallel_prefix.wgsl"),
                entrypoint: "pass1",
            },
            StageDesc {
                name: Some("second_pass"),
                shader: include_str!("parallel_prefix.wgsl"),
                entrypoint: "pass2",
            },
            StageDesc {
                name: Some("last_pass"),
                shader: include_str!("parallel_prefix.wgsl"),
                entrypoint: "pass3",
            },
        ],
    );
    pipeline.write_uniform(&Uniform { width: PER_WORKER });
    let mut input_padded = [0.0; N_PADDED];
    input_padded[..N].copy_from_slice(&input[..]);
    let result: [f32; N] = pipeline.run_blocking(
        &input_padded,
        [(N_WG as _, 1, 1), (1, 1, 1), (N_WG, 1, 1)],
        |vals: &[f32; N_PADDED]| {
            let mut res = [0.0f32; N];
            res.copy_from_slice(&vals[..N]);
            res
        },
    );
    let expected = parallel_prefix(&input);
    for (i, (v, exp)) in result.iter().zip(expected.iter()).enumerate() {
        if (v / exp - 1.0).abs() > i as f32 * f32::EPSILON {
            pipeline.dbg_print_scratchpad::<[f32; N_PADDED / PER_WORKER as usize]>();
            println!("Error at idx {}: {} ≠ {}", i, v, exp);
        }
    }
}
