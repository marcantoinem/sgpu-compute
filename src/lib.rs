//! # SGPU-Compute - **S**imple **GPU**-Compute using WebGPU
//! This crate aims to provide a simple and easy-to-use interface to run compute shaders with WGPU and WGSL. It is designed to be as simple as possible to use, while still providing a lot of flexibility for performance reason.
//! ## Example
//! ```
//! use sgpu_compute::prelude::*;
//!
//! let my_shader = "
//!    @group(0) @binding(0) var<uniform> coefficient: u32;
//!    @group(0) @binding(1) var<storage, read> in: array<u32>;
//!    @group(0) @binding(2) var<storage, read_write> out: array<u32>;
//!
//!    @compute
//!    @workgroup_size(8, 1, 1)
//!    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
//!        out[global_id.x] = coefficient * in[global_id.x];
//!    }
//! ";
//!
//! const N_ELEMENT: usize = 64;
//! const WORKGROUP_SIZE: usize = 8;
//! const N_WORKGROUP: u32 = (N_ELEMENT / WORKGROUP_SIZE) as u32;
//!
//! let gpu = GpuCompute::new();
//! let mut pipeline = gpu.gen_pipeline(
//!        None,
//!        [StageDesc {
//!            name: Some("norm"),
//!            shader: my_shader,
//!            entrypoint: "main",
//!        }],
//!    );
//!
//! const COEFFICIENT: u32 = 42;
//!
//! let input: [u32; N_ELEMENT] = std::array::from_fn(|i| i as u32);
//! pipeline.write_uniform(&COEFFICIENT);
//! let result_gpu = pipeline.run(&input, [(N_WORKGROUP, 1, 1)], |vals: &[u32; N_ELEMENT]| *vals);
//! let result_cpu = input.map(|v| v * COEFFICIENT);
//! assert_eq!(result_gpu, result_cpu);
//! ```
use std::{borrow::Cow, marker::PhantomData, num::NonZeroUsize};
use wgpu::{util::DownloadBuffer, Device, Queue};

#[cfg(feature = "blocking")]
pub mod blocking;

pub mod prelude;

/// This struct represents a pipeline. It is used to run compute shaders.
pub struct PipelineAsync<
    'a,
    Input: bytemuck::Pod,
    Uniform: bytemuck::Pod,
    Output: bytemuck::Pod,
    const N: usize,
> {
    uniform: Option<wgpu::Buffer>,
    input: wgpu::Buffer,
    scratchpad: Option<wgpu::Buffer>,
    staging: wgpu::Buffer,
    output: wgpu::Buffer,
    bindgroup: wgpu::BindGroup,
    stages: [wgpu::ComputePipeline; N],
    device: &'a GpuComputeAsync,
    stages_desc: [StageDesc; N],
    _phantom: PhantomData<(Input, Uniform, Output)>,
}

pub struct StageDesc {
    pub name: Option<&'static str>,
    pub shader: &'static str,
    pub entrypoint: &'static str,
}

/// This is the main struct of the library. It is used to create pipelines and run them. It requires an async runtime to work. If you want a blocking version, you can use the `GpuCompute` struct. If you don't use the blocking version disable default features.
pub struct GpuComputeAsync {
    device: Device,
    queue: Queue,
}

impl GpuComputeAsync {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("GPU not available.");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::PIPELINE_STATISTICS_QUERY
                        | wgpu::Features::TIMESTAMP_QUERY,
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        Self { device, queue }
    }

    /// The input, the uniform and the output must be `bytemuck::Pod` like shown in this small example. The `N` const parameter is the number of stages in the pipeline.
    /// ```rust
    /// use sgpu_compute::prelude::*;
    ///
    /// #[derive(Debug, Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
    /// #[repr(C)]
    /// struct Uniform {
    ///     width: u32,
    ///     height: u32,
    /// }
    /// #[pollster::main]
    /// async fn main() {
    ///     let gpu = GpuComputeAsync::new().await;
    ///     let pipeline = gpu.gen_pipeline::<[f32; 100], Uniform, [f32; 100], 1>( // This is the manual way to specify generics, but it can be inferred most of the times
    ///         None, // No scratchpad
    ///         [StageDesc {
    ///             name: Some("norm"),
    ///             shader: "@compute @workgroup_size(1) fn main() {}", // See other examples for shader content  
    ///             entrypoint: "main",
    ///         }]
    ///     ).await;
    /// }
    /// ```
    pub async fn gen_pipeline<
        Input: bytemuck::Pod,
        Uniform: bytemuck::Pod,
        Output: bytemuck::Pod,
        const N: usize,
    >(
        &self,
        scratchpad_size: Option<NonZeroUsize>,
        stages: [StageDesc; N],
    ) -> PipelineAsync<Input, Uniform, Output, N> {
        let uniform = if std::mem::size_of::<Uniform>() > 0 {
            Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: "Uniform buffer".into(),
                size: std::mem::size_of::<Uniform>() as _,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: false,
            }))
        } else {
            None
        };
        let scratchpad = scratchpad_size.map(|size| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: "Scratchpad buffer".into(),
                size: size.get() as _,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        });
        let input = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Input buffer"),
            size: std::mem::size_of::<Input>() as _,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging buffer"),
            size: std::mem::size_of::<Output>() as _,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging buffer"),
            size: std::mem::size_of::<Output>() as _,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut bindgroup_layout_items = uniform
            .as_ref()
            .map(|_| wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .into_iter()
            .chain(scratchpad.as_ref().map(|_| wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }))
            .chain(Some(wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }))
            .chain(Some(wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }))
            .collect::<Vec<_>>();
        bindgroup_layout_items
            .iter_mut()
            .enumerate()
            .for_each(|(i, item)| item.binding = i as _);

        let mut bindgroup_items = uniform
            .as_ref()
            .map(|buf| wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(buf.as_entire_buffer_binding()),
            })
            .into_iter()
            .chain(scratchpad.as_ref().map(|buf| wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(buf.as_entire_buffer_binding()),
            }))
            .chain(Some(wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(input.as_entire_buffer_binding()),
            }))
            .chain(Some(wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(staging.as_entire_buffer_binding()),
            }))
            .collect::<Vec<_>>();
        bindgroup_items
            .iter_mut()
            .enumerate()
            .for_each(|(i, item)| item.binding = i as _);

        let bindgroup_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &bindgroup_layout_items,
                    label: Some("Global bind group layout"),
                });
        let bindgroup = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bindgroup_layout,
            entries: &bindgroup_items,
            label: Some("Global bind group"),
        });

        let stages_pipeline: [_; N] = stages
            .iter()
            .map(|desc| {
                let shader = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: desc
                            .name
                            .map(|n| format!("Shader for stage {}", n))
                            .as_ref()
                            .map(AsRef::as_ref),
                        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(desc.shader)),
                    });

                let pipeline_layout =
                    self.device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: desc
                                .name
                                .map(|n| format!("Compute pipeline layout for stage {}", n))
                                .as_ref()
                                .map(AsRef::as_ref),
                            bind_group_layouts: &[&bindgroup_layout],
                            push_constant_ranges: &[],
                        });

                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: desc
                            .name
                            .map(|n| format!("Compute pipeline for stage {}", n))
                            .as_ref()
                            .map(AsRef::as_ref),
                        layout: Some(&pipeline_layout),
                        module: &shader,
                        entry_point: desc.entrypoint,
                    })
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("Wrong length?");

        PipelineAsync {
            uniform,
            input,
            scratchpad,
            staging,
            output,
            bindgroup,
            stages: stages_pipeline,
            stages_desc: stages,
            device: &self,
            _phantom: PhantomData,
        }
    }
}

impl<'a, Input: bytemuck::Pod, Uniform: bytemuck::Pod, Output: bytemuck::Pod, const N: usize>
    PipelineAsync<'a, Input, Uniform, Output, N>
{
    #[inline]
    pub fn write_uniform(&mut self, uniform: &Uniform) {
        self.device.queue.write_buffer(
            self.uniform.as_ref().expect("No uniforms"),
            0,
            bytemuck::bytes_of(uniform),
        )
    }

    pub fn dbg_print_scratchpad<T: bytemuck::Pod + bytemuck::AnyBitPattern + std::fmt::Debug>(
        &mut self,
    ) {
        DownloadBuffer::read_buffer(
            &self.device.device,
            &self.device.queue,
            &self.scratchpad.as_ref().expect("No scratchpad").slice(..),
            |res| {
                println!(
                    "Contents: {:?}",
                    bytemuck::from_bytes::<T>(
                        res.expect("Could not read scratchpad content").as_ref()
                    )
                );
            },
        )
    }

    pub async fn run<T: Send + 'static>(
        &mut self,
        input: &Input,
        workgroups: [(u32, u32, u32); N],
        callback: impl FnOnce(&Output) -> T + Send,
    ) -> T {
        self.device
            .queue
            .write_buffer(&self.input, 0, bytemuck::bytes_of(input));
        let mut encoder = self
            .device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for i in 0..N {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: self.stages_desc[i]
                    .name
                    .map(|n| format!("Compute pass for stage {}", n))
                    .as_ref()
                    .map(AsRef::as_ref),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.stages[i]);
            cpass.set_bind_group(0, &self.bindgroup, &[]);
            cpass.insert_debug_marker(
                &self.stages_desc[i]
                    .name
                    .map_or_else(|| format!("sgpu-{}", i), |n| format!("sgpu-{}", n)),
            );
            cpass.dispatch_workgroups(workgroups[i].0, workgroups[i].1, workgroups[i].2);
        }
        encoder.copy_buffer_to_buffer(
            &self.staging,
            0,
            &self.output,
            0,
            std::mem::size_of::<Output>() as _,
        );
        self.device.queue.submit(Some(encoder.finish()));
        let (sender, receiver) = flume::bounded(1);
        self.output
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |e| {
                sender.send(e.expect("Could not map buffer")).unwrap()
            });
        self.device.device.poll(wgpu::Maintain::Wait);
        receiver.recv_async().await.expect("Error with channel");
        let res = callback(bytemuck::from_bytes(
            self.output.slice(..).get_mapped_range().as_ref(),
        ));
        self.output.unmap();
        res
    }
}
