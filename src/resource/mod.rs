use std::sync::Arc;

use tyleri_api::data_structure::vertices::Vertex;
use tyleri_gpu_utils::image::format::FormatSize;
use tyleri_gpu_utils::memory::block_based_memory::bindless_buffer::BindlessBuffer;
use tyleri_gpu_utils::memory::memory_updater::MemoryUpdater;
use tyleri_gpu_utils::memory::IMemBakImg;
use yarvk::descriptor_set::descriptor_set::DescriptorSet;
use yarvk::image_subresource_range::ImageSubresourceRange;
use yarvk::image_view::{ImageView, ImageViewType};
use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::physical_device::SharingMode;
use yarvk::pipeline::pipeline_stage_flags::PipelineStageFlag;
use yarvk::{
    AccessFlags, ComponentMapping, ComponentSwizzle, ContinuousImage, ContinuousImageBuilder,
    Extent2D, Extent3D, Format, ImageAspectFlags, ImageLayout, ImageSubresourceLayers, ImageTiling,
    ImageType, ImageUsageFlags, Offset3D, SampleCountFlags,
};

use crate::pipeline::single_image_descriptor_set_layout::SingleImageDescriptorValue;
use crate::render_device::RenderDevice;

pub mod resource_allocator;
mod resource_info;

pub type StaticVertices = BindlessBuffer<Vertex>;
pub type StaticIndices = BindlessBuffer<u32>;
pub type StaticTexture = DescriptorSet<SingleImageDescriptorValue>;

impl RenderDevice {
    pub fn create_vertices(
        &self,
        data: Vec<(
            usize, /*len*/
            Box<dyn FnOnce(&mut [Vertex]) + Send + Sync>,
        )>,
    ) -> Vec<Arc<StaticVertices>> {
        if data.is_empty() {
            return Vec::new();
        }
        self.memory_allocator
            .static_vertices_buffer
            .allocate(data, &mut self.memory_allocator.queue.lock())
    }
    pub fn create_indices(
        &self,
        data: Vec<(
            usize, /*len*/
            Box<dyn FnOnce(&mut [u32]) + Send + Sync>,
        )>,
    ) -> Vec<Arc<StaticIndices>> {
        if data.is_empty() {
            return Vec::new();
        }
        self.memory_allocator
            .static_indices_buffer
            .allocate(data, &mut self.memory_allocator.queue.lock())
    }
    pub fn create_textures(
        &self,
        data: Vec<(
            Extent2D, /*size*/
            Box<dyn FnOnce(&mut [u8]) + Send + Sync>,
        )>,
    ) -> Vec<Arc<StaticTexture>> {
        if data.is_empty() {
            return Vec::new();
        }
        let device = &self.device;
        // duplicated code
        let mut builder = ContinuousImage::builder(device);
        builder.image_type(ImageType::TYPE_2D);
        builder.format(Format::R8G8B8A8_UNORM);
        builder.extent(Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        });
        builder.mip_levels(1);
        builder.array_layers(1);
        builder.samples(SampleCountFlags::TYPE_1);
        builder.tiling(ImageTiling::OPTIMAL);
        builder.usage(ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST);
        builder.sharing_mode(SharingMode::EXCLUSIVE);
        let memory_type = &self
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
        self.single_image_descriptor_set_layout
            .descriptor_pool_list
            .allocate(image_views.len() as _, &mut descriptor_sets)
            .unwrap();
        let mut updatable = self.device.update_descriptor_sets();
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
    fn create_image(
        &self,
        mut builder: ContinuousImageBuilder,
        memory_type: &MemoryType,
        data: Vec<(Extent2D, Box<dyn FnOnce(&mut [u8]) + Send + Sync>)>,
    ) -> Vec<Arc<IMemBakImg>> {
        let mut total_size = 0;
        for (extent, _) in data.as_slice() {
            total_size +=
                extent.width as u64 * extent.height as u64 * builder.get_format().format_size();
        }
        let it = data.iter().map(|(extent, _)| {
            builder.extent(extent.clone().into());
            builder.build().unwrap()
        });
        let allocator = self.memory_allocator.get_block_based_allocator(memory_type);
        let images = allocator.par_allocate(it, Some(total_size)).unwrap();
        let updater = MemoryUpdater::default();
        images
            .iter()
            .cloned()
            .zip(data)
            .for_each(|(image, (extent, f))| {
                updater.add_image(
                    &image as _,
                    builder.get_format().format_size(),
                    ImageSubresourceLayers::builder()
                        .aspect_mask(ImageAspectFlags::COLOR)
                        .layer_count(1)
                        .build(),
                    Offset3D::default(),
                    extent.into(),
                    AccessFlags::SHADER_READ,
                    ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    PipelineStageFlag::FragmentShader.into(),
                    f,
                )
            });
        updater.update(&mut self.memory_allocator.queue.lock());
        images
    }
}
