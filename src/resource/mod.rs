use std::sync::Arc;

use crate::pipeline::single_image_descriptor_set_layout::SingleImageDescriptorValue;
use tyleri_gpu_utils::image::format::FormatSize;
use tyleri_gpu_utils::memory::memory_updater::MemoryUpdater;
use tyleri_gpu_utils::memory::{try_memory_type, IMemBakBuf, IMemBakImg, MemoryObjectBuilder};
use yarvk::descriptor_set::descriptor_set::DescriptorSet;
use yarvk::device::Device;
use yarvk::device_memory::IMemoryRequirements;
use yarvk::image_subresource_range::ImageSubresourceRange;
use yarvk::image_view::{ImageView, ImageViewType};
use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::physical_device::SharingMode;
use yarvk::pipeline::pipeline_stage_flags::PipelineStageFlag;
use yarvk::{
    AccessFlags, BufferUsageFlags, ComponentMapping, ComponentSwizzle, ContinuousBuffer,
    ContinuousBufferBuilder, ContinuousImage, ContinuousImageBuilder, Extent2D, Extent3D, Format,
    ImageAspectFlags, ImageLayout, ImageSubresourceLayers, ImageTiling, ImageType, ImageUsageFlags,
    Offset3D, SampleCountFlags,
};

use crate::rendering_function::RenderingFunction;
use crate::Renderer;

pub mod resource_allocator;

pub struct ResCreateInfo<T: MemoryObjectBuilder> {
    builder: T,
    memory_type: MemoryType,
}

pub struct ResourcesInfo {
    vertices_info: ResCreateInfo<ContinuousBufferBuilder>,
    indices_info: ResCreateInfo<ContinuousBufferBuilder>,
    texture_info: ResCreateInfo<ContinuousImageBuilder>,
}

impl ResourcesInfo {
    pub fn new(device: &Arc<Device>) -> Self {
        Self {
            vertices_info: Self::create_vertices_info(device),
            indices_info: Self::create_indices_info(device),
            texture_info: Self::create_texture_info(device),
        }
    }
    fn create_indices_info(device: &Arc<Device>) -> ResCreateInfo<ContinuousBufferBuilder> {
        let device_memory_properties = device.physical_device.memory_properties();
        let mut buffer_builder = ContinuousBuffer::builder(&device);
        buffer_builder.sharing_mode(SharingMode::EXCLUSIVE);
        buffer_builder.size(1);
        buffer_builder.usage(BufferUsageFlags::INDEX_BUFFER | BufferUsageFlags::TRANSFER_DST);
        let index_buffer = buffer_builder.build().unwrap();
        let index_buffer_memory_req = index_buffer.get_memory_requirements();
        let memory_type = try_memory_type(
            index_buffer_memory_req,
            device_memory_properties,
            None,
            1024 * 1024 * 1024,
            |memory_type| Some(memory_type.clone()),
        )
        .unwrap();
        ResCreateInfo {
            builder: buffer_builder,
            memory_type,
        }
    }

    fn create_vertices_info(device: &Arc<Device>) -> ResCreateInfo<ContinuousBufferBuilder> {
        let device_memory_properties = device.physical_device.memory_properties();
        let mut buffer_builder = ContinuousBuffer::builder(&device);
        buffer_builder.sharing_mode(SharingMode::EXCLUSIVE);
        buffer_builder.size(1);
        buffer_builder.usage(BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::TRANSFER_DST);
        let vertices = buffer_builder.build().unwrap();
        let vertices_buffer_memory_req = vertices.get_memory_requirements();
        let memory_type = try_memory_type(
            vertices_buffer_memory_req,
            device_memory_properties,
            None,
            1024 * 1024 * 1024,
            |memory_type| Some(memory_type.clone()),
        )
        .unwrap();
        ResCreateInfo {
            builder: buffer_builder,
            memory_type,
        }
    }

    fn create_texture_info(device: &Arc<Device>) -> ResCreateInfo<ContinuousImageBuilder> {
        let device_memory_properties = device.physical_device.memory_properties();
        let mut image_builder = ContinuousImage::builder(&device);
        image_builder.image_type(ImageType::TYPE_2D);
        image_builder.format(Format::R8G8B8A8_UNORM);
        image_builder.extent(Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        });
        image_builder.mip_levels(1);
        image_builder.array_layers(1);
        image_builder.samples(SampleCountFlags::TYPE_1);
        image_builder.tiling(ImageTiling::OPTIMAL);
        image_builder.usage(ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST);
        image_builder.sharing_mode(SharingMode::EXCLUSIVE);
        let texture_image = image_builder.build().unwrap();
        let texture_image_memory_req = texture_image.get_memory_requirements();
        let memory_type = try_memory_type(
            texture_image_memory_req,
            device_memory_properties,
            None,
            1024 * 1024 * 1024,
            |memory_type| Some(memory_type.clone()),
        )
        .unwrap();
        ResCreateInfo {
            builder: image_builder,
            memory_type,
        }
    }
}

impl<T: RenderingFunction> Renderer<T> {
    pub fn create_vertices(
        &self,
        data: &[(u64 /*size*/, Arc<dyn Fn(&mut [u8]) + Send + Sync>)],
    ) -> Vec<Arc<IMemBakBuf>> {
        let builder = self
            .render_device
            .memory_allocator
            .resource_infos
            .vertices_info
            .builder
            .clone();
        let memory_type = &self
            .render_device
            .memory_allocator
            .resource_infos
            .vertices_info
            .memory_type;
        self.create_buffer(builder, memory_type, data)
    }
    pub fn create_indices(
        &self,
        data: &[(u64 /*size*/, Arc<dyn Fn(&mut [u8]) + Send + Sync>)],
    ) -> Vec<Arc<IMemBakBuf>> {
        let builder = self
            .render_device
            .memory_allocator
            .resource_infos
            .indices_info
            .builder
            .clone();
        let memory_type = &self
            .render_device
            .memory_allocator
            .resource_infos
            .indices_info
            .memory_type;
        self.create_buffer(builder, memory_type, data)
    }
    pub fn create_textures(
        &self,
        data: &[(Extent2D /*size*/, Arc<dyn Fn(&mut [u8]) + Send + Sync>)],
    ) -> Vec<Arc<DescriptorSet<SingleImageDescriptorValue>>> {
        let builder = self
            .render_device
            .memory_allocator
            .resource_infos
            .texture_info
            .builder
            .clone();
        let memory_type = &self
            .render_device
            .memory_allocator
            .resource_infos
            .texture_info
            .memory_type;
        let image_views: Vec<_> = self
            .create_image(builder, memory_type, data)
            .into_iter()
            .map(|texture_image| {
                ImageView::builder(texture_image.clone())
                    .view_type(ImageViewType::Type2d)
                    .format(Format::R8G8B8A8_UNORM)
                    .components(ComponentMapping {
                        r: ComponentSwizzle::R,
                        g: ComponentSwizzle::G,
                        b: ComponentSwizzle::B,
                        a: ComponentSwizzle::A,
                    })
                    .subresource_range(
                        ImageSubresourceRange::builder()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1)
                            .build(),
                    )
                    .build()
                    .unwrap()
            })
            .collect();
        let mut descriptor_sets = Vec::with_capacity(image_views.len());
        self.render_device
            .single_image_descriptor_set_layout
            .descriptor_pool_list
            .allocate(image_views.len() as _, &mut descriptor_sets)
            .unwrap();
        let mut updatable = self.render_device.device.update_descriptor_sets();
        descriptor_sets
            .iter_mut()
            .enumerate()
            .for_each(|(index, descriptor_set)| {
                updatable.add(descriptor_set, |_| {
                    let image_view = &image_views[index];
                    SingleImageDescriptorValue {
                        t0: [(image_view.clone(), ImageLayout::SHADER_READ_ONLY_OPTIMAL)],
                    }
                })
            });
        updatable.update();
        descriptor_sets
            .into_iter()
            .map(|descriptor_set| Arc::new(descriptor_set))
            .collect()
    }
    fn create_buffer(
        &self,
        mut builder: ContinuousBufferBuilder,
        memory_type: &MemoryType,
        data: &[(u64 /*size*/, Arc<dyn Fn(&mut [u8]) + Send + Sync>)],
    ) -> Vec<Arc<IMemBakBuf>> {
        let mut total_size = 0;
        for (size, _) in data {
            total_size += size;
        }
        let it = data.iter().map(|(size, _)| {
            builder.size(*size);
            builder.build().unwrap()
        });
        let allocator = self
            .render_device
            .memory_allocator
            .get_block_based_allocator(memory_type);
        let mut buffers = allocator.par_allocate(it, Some(total_size)).unwrap();
        let updater = MemoryUpdater::default();
        buffers.iter_mut().enumerate().for_each(|(index, buffer)| {
            updater.add_buffer(
                buffer,
                0,
                buffer.size(),
                AccessFlags::SHADER_READ,
                PipelineStageFlag::VertexShader.into(),
                data[index].1.clone(),
            )
        });
        updater.update(&mut self.render_device.memory_allocator.queue.lock());
        buffers
    }
    fn create_image(
        &self,
        mut builder: ContinuousImageBuilder,
        memory_type: &MemoryType,
        data: &[(Extent2D, Arc<dyn Fn(&mut [u8]) + Send + Sync>)],
    ) -> Vec<Arc<IMemBakImg>> {
        let mut total_size = 0;
        for (extent, _) in data {
            total_size +=
                extent.width as u64 * extent.height as u64 * builder.get_format().format_size();
        }
        let it = data.iter().map(|(extent, _)| {
            builder.extent(extent.clone().into());
            builder.build().unwrap()
        });
        let allocator = self
            .render_device
            .memory_allocator
            .get_block_based_allocator(memory_type);
        let mut images = allocator.par_allocate(it, Some(total_size)).unwrap();
        let updater = MemoryUpdater::default();
        images.iter_mut().enumerate().for_each(|(index, image)| {
            updater.add_image(
                image,
                builder.get_format().format_size(),
                ImageSubresourceLayers::builder()
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .build(),
                Offset3D::default(),
                data[index].0.into(),
                AccessFlags::SHADER_READ,
                ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                PipelineStageFlag::FragmentShader.into(),
                data[index].1.clone(),
            )
        });
        updater.update(&mut self.render_device.memory_allocator.queue.lock());
        images
    }
}
