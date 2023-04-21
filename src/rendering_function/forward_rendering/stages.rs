use std::slice::from_raw_parts;
use std::sync::Arc;

use glam::Vec2;
use yarvk::command::command_buffer::CommandBuffer;
use yarvk::command::command_buffer::Level::SECONDARY;
use yarvk::command::command_buffer::RenderPassScope::INSIDE;
use yarvk::command::command_buffer::State::RECORDING;
use yarvk::pipeline::shader_stage::ShaderStage;
use yarvk::{Extent2D, IndexType, PipelineBindPoint, Rect2D, Viewport};

use crate::render_device::RenderDevice;
use crate::render_objects::camera::Camera;
use crate::render_objects::mesh_renderer::MeshRenderer;
use crate::render_objects::ParallelGroup;
use crate::render_scene::RenderResources;
use crate::rendering_function::forward_rendering::ForwardRenderingFunction;

impl ForwardRenderingFunction {
    pub(crate) fn on_start(
        &self,
        camera: &Camera,
        command_buffer: &mut CommandBuffer<{ SECONDARY }, { RECORDING }, { INSIDE }>,
    ) {
        let viewport = camera.get_viewport();
        let scissor = camera.get_scissor();
        // set viewport and scissors
        command_buffer.cmd_set_viewport(viewport);
        command_buffer.cmd_set_scissor(scissor);
    }
    pub(super) fn on_render_ui(
        &self,
        window_size: Extent2D,
        scale_factor: f64,
        render_details: &RenderResources,
        command_buffer: &mut CommandBuffer<{ SECONDARY }, { RECORDING }, { INSIDE }>,
    ) {
        let ui_elements = render_details.ui.as_slice();
        if ui_elements.is_empty() {
            return;
        }
        let ui_indices = &render_details.ui_indices;
        if ui_indices.len() == 0 {
            return;
        }

        let ui_vertices = &render_details.ui_vertices;
        // bind pipeline
        command_buffer.cmd_bind_pipeline(
            PipelineBindPoint::GRAPHICS,
            self.ui_pipeline.pipeline.clone(),
        );
        // set viewport and scissors
        command_buffer.cmd_set_viewport(&Viewport {
            x: 0.0,
            y: 0.0,
            width: window_size.width as _,
            height: window_size.height as _,
            min_depth: 0.0,
            max_depth: 1.0,
        });
        command_buffer.cmd_set_scissor(&Rect2D {
            offset: Default::default(),
            extent: window_size,
        });
        let width_points = window_size.width as f32 / scale_factor as f32;
        let height_points = window_size.height as f32 / scale_factor as f32;
        let vec2 = Vec2::new(width_points, height_points);
        let push_constant = unsafe {
            from_raw_parts(
                &vec2 as *const Vec2 as *const u8,
                std::mem::size_of_val(&vec2),
            )
        };
        command_buffer.cmd_push_constants(
            &&self.common_pipeline.pipeline.pipeline_layout.clone(),
            &ShaderStage::Vertex,
            0,
            push_constant,
        );
        command_buffer.cmd_bind_vertex_buffers(0, [ui_vertices.clone() as _], &[0]);
        command_buffer.cmd_bind_index_buffer(ui_indices.clone() as _, 0, IndexType::UINT32);
        ui_elements.iter().for_each(|ui_element| {
            ui_element.renderer_ui(&self.common_pipeline.pipeline, command_buffer)
        })
    }
    pub(super) fn on_render_meshes(
        &self,
        render_device: &RenderDevice,
        camera: &Camera,
        parallel_meshes: &ParallelGroup<Arc<MeshRenderer>>,
        thread_index: usize,
        command_buffer: &mut CommandBuffer<{ SECONDARY }, { RECORDING }, { INSIDE }>,
    ) {
        let view_matrix = camera.get_view_matrix();
        let projection_matrix = camera.get_projection_matrix();
        let meshes = parallel_meshes
            .get_group_by_thread(thread_index)
            .expect("internal error: no group in thread index");
        command_buffer.cmd_bind_pipeline(
            PipelineBindPoint::GRAPHICS,
            self.common_pipeline.pipeline.clone(),
        );
        command_buffer.cmd_bind_vertex_buffers(
            0,
            [render_device
                .memory_allocator
                .static_vertices_buffer
                .get_buffer() as _],
            &[0],
        );
        command_buffer.cmd_bind_index_buffer(
            render_device
                .memory_allocator
                .static_indices_buffer
                .get_buffer() as _,
            0,
            IndexType::UINT32,
        );
        meshes.iter().for_each(|mesh_renderer| {
            mesh_renderer.renderer_mesh(
                &self.common_pipeline.pipeline,
                view_matrix,
                projection_matrix,
                command_buffer,
            );
        })
    }
}
