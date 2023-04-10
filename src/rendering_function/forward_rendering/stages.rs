use crate::render_device::RenderDevice;
use rayon::iter::ParallelIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator};
use yarvk::command::command_buffer::CommandBuffer;
use yarvk::command::command_buffer::Level::SECONDARY;
use yarvk::command::command_buffer::RenderPassScope::INSIDE;
use yarvk::command::command_buffer::State::RECORDING;
use yarvk::{IndexType, PipelineBindPoint};

use crate::render_objects::camera::Camera;
use crate::render_objects::RenderScene;
use crate::rendering_function::forward_rendering::ForwardRenderingFunction;

impl ForwardRenderingFunction {
    pub(super) fn on_start(
        &self,
        camera: &Camera,
        secondary_command_buffers: &mut [CommandBuffer<{ SECONDARY }, { RECORDING }, { INSIDE }>],
    ) {
        let viewport = camera.get_viewport();
        let scissor = camera.get_scissor();
        secondary_command_buffers
            .par_iter_mut()
            .for_each(|command_buffer| {
                // set viewport and scissors
                command_buffer.cmd_set_viewport(viewport);
                command_buffer.cmd_set_scissor(scissor);
                // bind pipeline
                command_buffer.cmd_bind_pipeline(
                    PipelineBindPoint::GRAPHICS,
                    self.common_pipeline.pipeline.clone(),
                );
            });
    }
    pub(super) fn on_render_meshes(
        &self,
        render_device: &RenderDevice,
        camera: &Camera,
        render_scene: &RenderScene,
        secondary_command_buffers: &mut [CommandBuffer<{ SECONDARY }, { RECORDING }, { INSIDE }>],
    ) {
        let mesh_group = render_scene.get_mesh_group();
        let view_matrix = camera.get_view_matrix();
        let projection_matrix = camera.get_projection_matrix();
        secondary_command_buffers
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, command_buffer)| {
                command_buffer.cmd_bind_vertex_buffers(
                    0,
                    [render_device.memory_allocator.vertices_buffer.get_buffer() as _],
                    &[0],
                );
                command_buffer.cmd_bind_index_buffer(
                    render_device.memory_allocator.indices_buffer.get_buffer() as _,
                    0,
                    IndexType::UINT32,
                );
                mesh_group[index].iter().for_each(|mesh_renderer| {
                    mesh_renderer.renderer_mesh(
                        &self.common_pipeline.pipeline,
                        view_matrix,
                        projection_matrix,
                        command_buffer,
                    );
                })
            })
    }
}
