use crate::render_resource::ResourceBinding;
use crate::unlimited_descriptor_pool::{DescriptorUpdateInfo, UnlimitedDescriptorPool};

use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::sync::Arc;
use yarvk::descriptor::descriptor_set_layout::{DescriptorSetLayout, DescriptorSetLayoutBinding};
use yarvk::descriptor::write_descriptor_sets::{DescriptorImageInfo, WriteDescriptorSets};
use yarvk::descriptor::DescriptorType;
use yarvk::device::Device;
use yarvk::device_features::PhysicalDeviceFeatures::SamplerAnisotropy;
use yarvk::pipeline::shader_stage::ShaderStage;
use yarvk::sampler::Sampler;
use yarvk::{
    BorderColor, CompareOp, Extent2D, Filter, Format, SamplerAddressMode, SamplerMipmapMode,
};

pub struct TextureImageInfo {
    data: Vec<u8>,
    format: Format,
    extent: Extent2D,
}

pub struct TextureAllocator {
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    pub descriptor_pool: UnlimitedDescriptorPool<TextureSamplerUpdateInfo>,
}

impl TextureAllocator {
    pub fn new(default_sampler: Arc<Sampler>) -> Result<Self, yarvk::Result> {
        let device = default_sampler.device.clone();
        let descriptor_set_layout = DescriptorSetLayout::builder(device.clone())
            .add_binding(
                DescriptorSetLayoutBinding::builder()
                    .binding(ResourceBinding::TextureSampler as _)
                    .descriptor_type(DescriptorType::CombinedImageSamplerImmutable(vec![
                        default_sampler,
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
        })
    }
}

pub struct TextureSamplerUpdateInfo {
    pub update_infos: [DescriptorImageInfo; 1],
}

impl TextureSamplerUpdateInfo {
    pub fn descriptor_layout(
        device: Arc<Device>,
    ) -> Result<Arc<DescriptorSetLayout>, yarvk::Result> {
        DescriptorSetLayout::builder(device.clone())
            .add_binding(
                DescriptorSetLayoutBinding::builder()
                    .binding(ResourceBinding::TextureSampler as _)
                    .descriptor_type(DescriptorType::CombinedImageSampler)
                    .descriptor_count(1)
                    .add_stage_flag(ShaderStage::Fragment)
                    .build(),
            )
            .build()
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
