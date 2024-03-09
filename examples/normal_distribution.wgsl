struct Uniform {
    n_block: u32,
}

@group(0) @binding(0) var<uniform> cfg: Uniform;
@group(0) @binding(1) var<storage, read> in: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

const pi: f32 = 3.1415926535897932384626433;

/// Compute the standard normal distribution with mean 0 and variance 1.0
fn normal_distribution(x: f32) -> f32 {
    let exponent: f32 = -0.5 * x * x;
    return exp(exponent) / sqrt(2.0 * pi);
}

/// Integrate between 0 and x the normal distribution with numerical integration and n_blocks rectangles.
fn integrate(born: f32) -> f32 {
    let width: f32 = born / f32(cfg.n_block);
    var area: f32 = 0.0;

    for (var i: u32 = 0; i < cfg.n_block; i++) {
        let x: f32 = width * (f32(i) + 0.5);
        area += normal_distribution(x) * width;
    }

    return area + 0.5;
}

@compute
@workgroup_size(10, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    out[global_id.x] = integrate(in[global_id.x]);
}