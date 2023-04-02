use std::mem::size_of;
use std::slice::from_raw_parts;
use std::sync::Arc;

use glam::Mat4;
use yarvk::command::command_buffer::CommandBuffer;
use yarvk::command::command_buffer::Level::SECONDARY;
use yarvk::command::command_buffer::RenderPassScope::INSIDE;
use yarvk::command::command_buffer::State::RECORDING;
use yarvk::descriptor_set::descriptor_set::DescriptorSet;
use yarvk::device_memory::IMemoryRequirements;
use yarvk::image_view::ImageView;
use yarvk::pipeline::shader_stage::ShaderStage;
use yarvk::pipeline::Pipeline;
use yarvk::{IBuffer, ImageLayout, IndexType, PipelineBindPoint};

use crate::pipeline::single_image_descriptor_set_layout::SingleImageDescriptorValue;

#[repr(C)]
struct MVP {
    view_x_model: Mat4,
    projection: Mat4,
}

pub struct MeshRenderer {
    // TODO maybe split vertices to three buffers?
    vertices: Arc<IBuffer>,
    indices: Arc<IBuffer>,
    descriptor_set: Arc<DescriptorSet<SingleImageDescriptorValue>>,
    model: Mat4,
}

impl MeshRenderer {
    pub fn new(
        vertices: Arc<IBuffer>,
        indices: Arc<IBuffer>,
        descriptor_set: Arc<DescriptorSet<SingleImageDescriptorValue>>,
    ) -> Self {
        Self {
            vertices,
            indices,
            descriptor_set,
            model: Default::default(),
        }
    }
    pub fn set_vertices(&mut self, vertices: Arc<IBuffer>) {
        self.vertices = vertices
    }
    pub fn set_indices(&mut self, indices: Arc<IBuffer>) {
        self.indices = indices
    }
    pub fn set_texture(&mut self, texture: Arc<ImageView>) {
        let mut updatable = self.vertices.device.update_descriptor_sets();
        // Tried to batch update descriptors, but it's way to complicated, and need locks,
        // don't know if worth it
        let descriptor_set =
            Arc::get_mut(&mut self.descriptor_set).expect("descriptor set is holding by others");
        updatable.add(descriptor_set, |_| SingleImageDescriptorValue {
            t0: [(texture.clone(), ImageLayout::SHADER_READ_ONLY_OPTIMAL)],
        })
    }
    pub fn set_model_matrix(&mut self, model: Mat4) {
        self.model = model
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
        command_buffer.cmd_bind_vertex_buffers(0, [self.vertices.clone() as _], &[0]);
        command_buffer.cmd_bind_index_buffer(self.indices.clone(), 0, IndexType::UINT32);
        command_buffer.cmd_draw_indexed(
            (self.indices.get_memory_requirements().size / 4) as u32,
            1,
            0,
            0,
            1,
        );
    }
}
