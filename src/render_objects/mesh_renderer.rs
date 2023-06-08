use std::mem::size_of;
use std::slice::from_raw_parts;
use std::sync::Arc;

use glam::Mat4;
use tyleri_api::data_structure::vertices::Vertex;
use tyleri_gpu_utils::descriptor::single_image_descriptor_set_layout::SingleImageDescriptorValue;
use tyleri_gpu_utils::memory::block_based_memory::bindless_buffer::BindlessBuffer;
use yarvk::command::command_buffer::CommandBuffer;
use yarvk::command::command_buffer::Level::SECONDARY;
use yarvk::command::command_buffer::RenderPassScope::INSIDE;
use yarvk::command::command_buffer::State::RECORDING;
use yarvk::descriptor_set::descriptor_set::DescriptorSet;
use yarvk::pipeline::shader_stage::ShaderStage;
use yarvk::pipeline::Pipeline;
use yarvk::PipelineBindPoint;

#[repr(C)]
struct MVP {
    view_x_model: Mat4,
    projection: Mat4,
}

pub struct MeshRenderer {
    // TODO maybe split vertices to three buffers?
    pub vertices: Arc<BindlessBuffer<Vertex>>,
    pub indices: Arc<BindlessBuffer<u32>>,
    pub descriptor_set: Arc<DescriptorSet<SingleImageDescriptorValue>>,
    pub model: Mat4,
}

impl MeshRenderer {
    pub fn new(
        vertices: Arc<BindlessBuffer<Vertex>>,
        indices: Arc<BindlessBuffer<u32>>,
        descriptor_set: Arc<DescriptorSet<SingleImageDescriptorValue>>,
    ) -> Self {
        Self {
            vertices,
            indices,
            descriptor_set,
            model: Default::default(),
        }
    }
    pub fn renderer_mesh(
        &self,
        pipeline: &Arc<Pipeline>,
        view: &Mat4,
        projection: &Mat4,
        command_buffer: &mut CommandBuffer<{ SECONDARY }, { RECORDING }, { INSIDE }>,
    ) {
        let view_x_model = *view * self.model;
        let mvp = MVP {
            view_x_model,
            projection: projection.clone(),
        };
        let push_constant =
            unsafe { from_raw_parts(&mvp as *const MVP as *const u8, size_of::<MVP>()) };
        command_buffer.cmd_push_constants(
            &pipeline.pipeline_layout,
            &ShaderStage::Vertex,
            0,
            push_constant,
        );
        command_buffer.cmd_bind_descriptor_sets(
            PipelineBindPoint::GRAPHICS,
            pipeline.pipeline_layout.clone(),
            0,
            [self.descriptor_set.clone() as _],
            &[],
        );
        command_buffer.cmd_draw_indexed(
            self.indices.len as u32,
            1,
            self.indices.offset as _,
            self.vertices.offset as _,
            1,
        );
    }
}
