#![feature(trait_upcasting)]
use crate::allocator::Allocator;
use crate::queue_manager::QueueManager;
use crate::renderer_builder::RendererBuilder;
use std::sync::Arc;

use yarvk::swapchain::Swapchain;

use yarvk::command::command_buffer::CommandBuffer;
use yarvk::command::command_buffer::Level::SECONDARY;
use yarvk::command::command_buffer::RenderPassScope::INSIDE;
use yarvk::command::command_buffer::State::RECORDING;

use yarvk::pipeline::{PipelineBuilder, PipelineLayout};
use yarvk::Extent2D;

pub mod allocator;
pub mod queue_manager;
pub mod renderer_builder;
pub mod rendering_function;

use crate::rendering_function::forward_rendering_function::ForwardRenderingFunction;
use crate::rendering_function::RenderingFunction;
pub use renderer_builder::*;

pub enum RenderingFunctionType {
    ForwardRendering,
}

pub struct Renderer {
    pub queue_manager: QueueManager,
    pub swapchain: Swapchain,
    pub allocator: Allocator,
    forward_rendering_function: ForwardRenderingFunction,
}

impl Renderer {
    pub fn builder() -> RendererBuilder {
        RendererBuilder::new()
    }
    pub fn render_next_frame<
        F: FnMut(
            &mut [CommandBuffer<{ SECONDARY }, { RECORDING }, { INSIDE }, true>],
        ) -> Result<(), yarvk::Result>,
    >(
        &mut self,
        f: F,
    ) {
        let mut queue = self
            .queue_manager
            .take_present_queue_priority_high()
            .unwrap();
        self.forward_rendering_function
            .record_next_frame(&mut self.swapchain, &mut queue, f)
            .unwrap();
        self.queue_manager.push_queue(queue);
    }
    pub fn on_resolution_changed(&mut self, resolution: Extent2D) -> Result<(), yarvk::Result> {
        self.swapchain = create_swapchain(
            self.swapchain.device.clone(),
            self.swapchain.surface.clone(),
            resolution,
        )?;
        self.forward_rendering_function = ForwardRenderingFunction::new(
            &self.swapchain,
            &mut self.queue_manager,
            &mut self.allocator,
        )?;
        Ok(())
    }
    pub fn forward_rendering_pipeline_builder(
        &self,
        layout: Arc<PipelineLayout>,
    ) -> PipelineBuilder {
        self.forward_rendering_function.pipeline_builder(layout, 0)
    }
}
