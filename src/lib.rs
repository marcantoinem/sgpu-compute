use std::{borrow::Cow, marker::PhantomData, num::NonZeroUsize};
use wgpu::{util::DownloadBuffer, Device, Queue};

#[cfg(feature = "blocking")]
pub mod blocking;

pub struct Pipeline<
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

    pub async fn gen_pipeline<
        Input: bytemuck::Pod,
        Uniform: bytemuck::Pod,
        Output: bytemuck::Pod,
        const N: usize,
    >(
        &self,
        scratchpad_size: Option<NonZeroUsize>,
        stages: [StageDesc; N],
    ) -> Pipeline<Input, Uniform, Output, N> {
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
            .chain(
                scratchpad
                    .as_ref()
                    .map(|_| wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }),
            )
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

        Pipeline {
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
    Pipeline<'a, Input, Uniform, Output, N>
{
    pub fn write_uniform(&mut self, uniform: &Uniform) {
        self.device.queue.write_buffer(
            self.uniform.as_ref().expect("No uniforms"),
            0,
            bytemuck::bytes_of(uniform),
        )
    }

    pub fn dbg_print_scratchpad<T: bytemuck::Pod + bytemuck::AnyBitPattern + std::fmt::Debug>(&mut self) {
        DownloadBuffer::read_buffer(&self.device.device, &self.device.queue, &self.scratchpad.as_ref().expect("No scratchpad").slice(..), |res| {
            println!("Contents: {:?}", bytemuck::from_bytes::<T>(res.expect("Could not read scratchpad content").as_ref()));
        })
    }

    #[cfg(feature = "blocking")]
    pub fn run_blocking<T: Send + 'static>(
        &mut self,
        input: &Input,
        workgroups: [(u32, u32, u32); N],
        callback: impl FnOnce(&Output) -> T + Send + 'static,
    ) -> T {
        pollster::block_on(self.run(input, workgroups, callback))
    }

    pub async fn run<T: Send + 'static>(
        &mut self,
        input: &Input,
        workgroups: [(u32, u32, u32); N],
        callback: impl FnOnce(&Output) -> T + Send + 'static,
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
