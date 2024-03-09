struct Uniform {
    width: u32,
}

@group(0) @binding(0) var<uniform> settings: Uniform;
@group(0) @binding(1) var<storage, read_write> scratchpad: array<f32>;
@group(0) @binding(2) var<storage, read> in: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute
@workgroup_size(16, 1, 1)
fn pass1(@builtin(global_invocation_id) id: vec3<u32>) {
    let start = settings.width * id.x;
    let end = settings.width * (id.x + 1);

    var total = 0.0;
    for (var i = start; i < end; i++) {
        total += in[i];
    }
    scratchpad[id.x] = total;
}

@compute
@workgroup_size(1, 1, 1)
fn pass2() {
    var total = 0.0;
    for (var i = 0u; i < arrayLength(&in) / settings.width; i++) {
        let v = scratchpad[i];
        scratchpad[i] = total;
        total += v;
    }
}

@compute
@workgroup_size(16, 1, 1)
fn pass3(@builtin(global_invocation_id) id: vec3<u32>) {
    let start = settings.width * id.x;
    let end = settings.width * (id.x + 1);
    var total = scratchpad[id.x];
    for (var i = start; i < end; i++) {
        total += in[i];
        out[i] = total;
    }
}