use rayon::prelude::*;
use std::f32::consts::PI;

fn normal_distribution(x: f32) -> f32 {
    (-0.5 * (x.powi(2))).exp() / (2.0 * PI).sqrt()
}

fn numerical_integration_cpu_single(born: f32) -> f32 {
    let mut area = 0.0;
    let width = born / 32768.0;
    for i in 0..32768 {
        let x = i as f32 * width + 0.5 * width;
        area += width * normal_distribution(x);
    }
    0.5 + area
}

pub fn numerical_integration_cpu(to_integrate: &[f32]) -> Vec<f32> {
    to_integrate
        .par_iter()
        .map(|&x| numerical_integration_cpu_single(x))
        .collect()
}
