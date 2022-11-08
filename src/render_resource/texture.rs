use crate::render_resource::ResourceBinding;
use crate::unlimited_descriptor_pool::{DescriptorUpdateInfo, UnlimitedDescriptorPool};

use crate::memory_allocator::{MemoryAllocator, MemoryBindingBuilder};
use crate::queue_manager::recordable_queue::RecordableQueue;
use crate::Renderer;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::sync::Arc;
use yarvk::barrier::ImageMemoryBarrier;
use yarvk::descriptor::descriptor_set::DescriptorSet;
use yarvk::descriptor::descriptor_set_layout::{DescriptorSetLayout, DescriptorSetLayoutBinding};
use yarvk::descriptor::write_descriptor_sets::{DescriptorImageInfo, WriteDescriptorSets};
use yarvk::descriptor::DescriptorType;
use yarvk::device::Device;
use yarvk::device_features::PhysicalDeviceFeatures::SamplerAnisotropy;
use yarvk::image_subresource_range::ImageSubresourceRange;
use yarvk::image_view::{ImageView, ImageViewType};
use yarvk::physical_device::SharingMode;
use yarvk::pipeline::pipeline_stage_flags::PipelineStageFlags;
use yarvk::pipeline::shader_stage::ShaderStage;
use yarvk::queue::Queue;
use yarvk::sampler::Sampler;
use yarvk::{AccessFlags, BorderColor, BufferImageCopy, BufferUsageFlags, CompareOp, ComponentMapping, ComponentSwizzle, ContinuousBuffer, ContinuousImage, DependencyFlags, Extent2D, Extent3D, Filter, Format, Handle, ImageAspectFlags, ImageLayout, ImageSubresourceLayers, ImageTiling, ImageType, ImageUsageFlags, MemoryPropertyFlags, SampleCountFlags, SamplerAddressMode, SamplerMipmapMode};

pub struct TextureImageInfo {
    pub data: Vec<u8>,
    pub format: Format,
    pub extent: Extent2D,
    pub index: usize,
}

pub struct TextureAllocator {
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    descriptor_pool: UnlimitedDescriptorPool<TextureSamplerUpdateInfo>,
    memory_allocator: Arc<MemoryAllocator>,
}

impl TextureAllocator {
    pub fn new(
        default_sampler: Arc<Sampler>,
        memory_allocator: &Arc<MemoryAllocator>,
    ) -> Result<Self, yarvk::Result> {
        let device = default_sampler.device.clone();
        let descriptor_set_layout = DescriptorSetLayout::builder(device.clone())
            .add_binding(
                DescriptorSetLayoutBinding::builder()
                    .binding(ResourceBinding::TextureSampler as _)
                    .descriptor_type(DescriptorType::CombinedImageSamplerImmutable(vec![
                        default_sampler.clone(),
                    ]))
                    .descriptor_count(1)
                    .add_stage_flag(ShaderStage::Fragment)
                    .build(),
            )
            .build()?;
        let descriptor_pool =
            UnlimitedDescriptorPool::new(device.clone(), descriptor_set_layout.clone());
        Ok(Self {
            descriptor_set_layout,
            descriptor_pool,
            memory_allocator: memory_allocator.clone(),
        })
    }
    pub fn descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.descriptor_set_layout.clone()
    }
    pub fn allocate_textures<It: IntoParallelIterator<Item = TextureImageInfo>>(
        &mut self,
        it: It,
        transfer_queue: Option<&mut RecordableQueue>,
        present_queue: &mut RecordableQueue,
    ) {
        let updates = it
            .into_par_iter()
            .map(|image_info: TextureImageInfo| {
                // buffer to hold image data
                let image_data = image_info.data.as_slice();
                let device = self.descriptor_set_layout.device.clone();
                let mut image_buffer = ContinuousBuffer::builder(device.clone())
                    .size(image_data.len() as _)
                    .usage(BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(SharingMode::EXCLUSIVE)
                    .build_and_bind(
                        &self.memory_allocator,
                        MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
                        false,
                    )
                    .unwrap();
                Arc::get_mut(&mut image_buffer)
                    .unwrap()
                    .map_host_local_memory(&|mut_slice| {
                        mut_slice[0..image_data.len()].copy_from_slice(image_data);
                    })
                    .unwrap();
                // texture image
                let texture_image = ContinuousImage::builder(device.clone())
                    .image_type(ImageType::TYPE_2D)
                    .format(image_info.format)
                    .extent(image_info.extent.into())
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(SampleCountFlags::TYPE_1)
                    .tiling(ImageTiling::OPTIMAL)
                    .usage(ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED)
                    .sharing_mode(SharingMode::EXCLUSIVE)
                    .build_and_bind(
                        &self.memory_allocator,
                        MemoryPropertyFlags::DEVICE_LOCAL,
                        false,
                    )
                    .unwrap();
                let tex_image_view = ImageView::builder(texture_image.clone())
                    .view_type(ImageViewType::Type2d)
                    .format(texture_image.image_create_info.format)
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
                    .unwrap();
                let texture_barrier = ImageMemoryBarrier::builder(texture_image.clone())
                    .dst_access_mask(AccessFlags::TRANSFER_WRITE)
                    .new_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
                    .subresource_range(
                        ImageSubresourceRange::builder()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1)
                            .build(),
                    )
                    .build();
                let mut command_buffer = (transfer_queue.as_ref().unwrap_or(&present_queue))
                    .get_thread_local_secondary_buffer()
                    .unwrap();
                let command_buffer = command_buffer.value_mut();
                command_buffer.cmd_pipeline_barrier(
                    [PipelineStageFlags::BottomOfPipe],
                    [PipelineStageFlags::Transfer],
                    DependencyFlags::empty(),
                    [],
                    [],
                    [texture_barrier],
                );
                let buffer_copy_regions = BufferImageCopy::builder()
                    .image_subresource(
                        ImageSubresourceLayers::builder()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .build(),
                    )
                    .image_extent(Extent3D {
                        width: image_info.extent.width,
                        height: image_info.extent.height,
                        depth: 1,
                    });
                command_buffer.cmd_copy_buffer_to_image(
                    image_buffer.clone(),
                    texture_image.clone(),
                    ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[buffer_copy_regions.build()],
                );
                (image_info.index, tex_image_view)
            })
            .collect::<Vec<_>>();
        let copy_end_barriers = updates.iter().map(|(_, view)| {
            ImageMemoryBarrier::builder(view.image.clone())
                .src_access_mask(AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(AccessFlags::SHADER_READ)
                .old_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .subresource_range(
                    ImageSubresourceRange::builder()
                        .aspect_mask(ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1)
                        .build(),
                )
                .build()
        });
        self.descriptor_pool.allocate(
            updates
                .iter()
                .map(|(index, view)| (*index, TextureSamplerUpdateInfo::new(view.clone()))),
        );
        // Copy images
        transfer_queue
            .unwrap_or(present_queue)
            .simple_secondary_record()
            .unwrap();
        // TODO submit together
        // set image layout
        present_queue
            .simple_record(|command_buffer| {
                command_buffer.cmd_pipeline_barrier(
                    [PipelineStageFlags::Transfer],
                    [PipelineStageFlags::FragmentShader],
                    DependencyFlags::empty(),
                    [],
                    [],
                    copy_end_barriers,
                );
                Ok(())
            })
            .unwrap();
    }
    pub fn get_descriptor_set(&self, index: &usize) -> Option<&DescriptorSet> {
        self.descriptor_pool.get_descriptor_set(index)
    }
}

#[derive(Clone)]
pub struct TextureSamplerUpdateInfo {
    update_infos: [DescriptorImageInfo; 1],
}

impl TextureSamplerUpdateInfo {
    pub fn new(image_view: Arc<ImageView>) -> Self {
        Self {
            update_infos: [DescriptorImageInfo::builder()
                .image_view(image_view)
                .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .build()],
        }
    }
}

impl DescriptorUpdateInfo for TextureSamplerUpdateInfo {
    fn add_to_write_descriptor_set<'a>(
        &'a self,
        descriptor_set_index: usize,
        updatable: &mut WriteDescriptorSets<'a>,
    ) {
        updatable.update_image(
            descriptor_set_index,
            ResourceBinding::TextureSampler as _,
            0,
            &self.update_infos,
        )
    }
}
