use std::sync::Arc;
use yarvk::command::command_buffer::CommandBuffer;
use yarvk::command::command_buffer::Level::SECONDARY;
use yarvk::command::command_buffer::RenderPassScope::INSIDE;
use yarvk::command::command_buffer::State::RECORDING;
use yarvk::pipeline::pipeline_cache::PipelineCacheImpl;
use yarvk::pipeline::{PipelineBuilder, PipelineLayout};
use yarvk::queue::Queue;
use yarvk::swapchain::Swapchain;

pub mod forward_rendering_function;
pub mod frame_store;

pub(crate) trait RenderingFunction {
    fn record_next_frame<
        F: FnOnce(
            &mut [CommandBuffer<{ SECONDARY }, { RECORDING }, { INSIDE }>],
        ) -> Result<(), yarvk::Result>,
    >(
        &mut self,
        swapchain: &mut Swapchain,
        present_queue: &mut Queue,
        f: F,
    ) -> Result<(), yarvk::Result>;

    fn pipeline_builder<'a>(
        &'a self,
        layout: Arc<PipelineLayout>,
        pipeline_cache: &'a PipelineCacheImpl<false>,
        subpass: u32,
    ) -> PipelineBuilder<'a>;
}
