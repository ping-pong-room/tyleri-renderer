use crate::render_resource::ResourceBinding;
use crate::unlimited_descriptor_pool::DescriptorUpdateInfo;

use std::sync::Arc;
use yarvk::descriptor::descriptor_set_layout::{DescriptorSetLayout, DescriptorSetLayoutBinding};
use yarvk::descriptor::write_descriptor_sets::{DescriptorImageInfo, WriteDescriptorSets};
use yarvk::descriptor::DescriptorType;
use yarvk::device::Device;

use yarvk::pipeline::shader_stage::ShaderStage;



// TODO immutable sampler
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
