use std::sync::Arc;
use tyleri_api::data_structure::vertices::UIVertex;
use tyleri_gpu_utils::memory::variable_length_buffer::VariableLengthBuffer;

use yarvk::command::command_buffer::CommandBuffer;
use yarvk::command::command_buffer::Level::SECONDARY;
use yarvk::command::command_buffer::RenderPassScope::INSIDE;
use yarvk::command::command_buffer::State::RECORDING;
use yarvk::descriptor_set::descriptor_set::DescriptorSet;
use yarvk::pipeline::Pipeline;
use yarvk::PipelineBindPoint;

use crate::pipeline::single_image_descriptor_set_layout::SingleImageDescriptorValue;
use crate::render_scene::RenderScene;

pub(crate) struct UIElement {
    vertex_offset: usize,
    index_offset: usize,
    index_len: usize,
    descriptor_set: Arc<DescriptorSet<SingleImageDescriptorValue>>,
}

impl UIElement {
    pub fn renderer_ui(
        &self,
        pipeline: &Arc<Pipeline>,
        command_buffer: &mut CommandBuffer<{ SECONDARY }, { RECORDING }, { INSIDE }>,
    ) {
        command_buffer.cmd_bind_descriptor_sets(
            PipelineBindPoint::GRAPHICS,
            pipeline.pipeline_layout.clone(),
            0,
            [self.descriptor_set.clone() as _],
            &[],
        );
        command_buffer.cmd_draw_indexed(
            self.index_len as u32,
            1,
            self.index_offset as _,
            self.vertex_offset as _,
            1,
        );
    }
}

pub type RawUIData = Vec<(
    Vec<UIVertex>,
    Vec<u32>,
    Arc<DescriptorSet<SingleImageDescriptorValue>>,
)>;

pub struct UI {
    ui_elements: Vec<UIElement>,
}

impl UI {
    pub(crate) fn to_ui_elements(
        raw_data: RawUIData,
        vertex_buffer: &mut VariableLengthBuffer<UIVertex>,
        index_buffer: &mut VariableLengthBuffer<u32>,
    ) -> Vec<UIElement> {
        if raw_data.is_empty() {
            return vec![];
        }
        let mut ui_elements = Vec::with_capacity(raw_data.len());
        let mut total_vertices_len: usize = 0;
        let mut total_indices_len: usize = 0;
        raw_data.iter().for_each(|(vertices, indices, _)| {
            total_vertices_len += vertices.len();
            total_indices_len += indices.len();
        });

        vertex_buffer.expand_to(total_vertices_len);

        index_buffer.expand_to(total_indices_len);

        raw_data.iter().for_each(|(vertices, indices, texture)| {
            let vertex_offset = vertex_buffer.write(vertices);
            let index_offset = index_buffer.write(indices);
            ui_elements.push(UIElement {
                vertex_offset,
                index_offset,
                index_len: indices.len(),
                descriptor_set: texture.clone(),
            })
        });
        ui_elements
    }
    pub(crate) fn get_ui_elements(&self) -> &[UIElement] {
        self.ui_elements.as_slice()
    }
}

impl RenderScene {
    pub fn add_ui(&mut self, raw_data: RawUIData) {
        let ui_vertex_buffer =
            Arc::get_mut(&mut self.render_resources.ui_vertices).expect("internal error");
        let ui_index_buffer =
            Arc::get_mut(&mut self.render_resources.ui_indices).expect("internal error");
        if raw_data.is_empty() {
            self.render_resources.ui = vec![]
        }
        let mut ui_elements = Vec::with_capacity(raw_data.len());
        let mut total_vertices_len: usize = 0;
        let mut total_indices_len: usize = 0;
        raw_data.iter().for_each(|(vertices, indices, _)| {
            total_vertices_len += vertices.len();
            total_indices_len += indices.len();
        });

        ui_vertex_buffer.expand_to(total_vertices_len);

        ui_index_buffer.expand_to(total_indices_len);

        raw_data.iter().for_each(|(vertices, indices, texture)| {
            let vertex_offset = ui_vertex_buffer.write(vertices);
            let index_offset = ui_index_buffer.write(indices);
            ui_elements.push(UIElement {
                vertex_offset,
                index_offset,
                index_len: indices.len(),
                descriptor_set: texture.clone(),
            })
        });
        self.render_resources.ui = ui_elements;
    }
}
