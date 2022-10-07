use crate::allocator::Allocator;
use crate::builder::RendererBuilder;
use crate::queue_manager::QueueManager;
use crate::renderpass_set::RenderPassSet;

use yarvk::swapchain::Swapchain;

use yarvk::command::command_buffer::CommandBuffer;
use yarvk::command::command_buffer::Level::SECONDARY;
use yarvk::command::command_buffer::RenderPassScope::INSIDE;
use yarvk::command::command_buffer::State::RECORDING;

pub mod allocator;
pub mod builder;
pub mod queue_manager;
pub mod renderpass_set;

pub use builder::*;

pub struct Renderer {
    pub queue_manager: QueueManager,
    pub swapchain: Swapchain,
    pub allocator: Allocator,
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
        renderpass_set: &mut RenderPassSet,
        f: F,
    ) {
        let mut queue = self
            .queue_manager
            .take_present_queue_priority_high()
            .unwrap();
        renderpass_set
            .render_next_frame(&mut self.swapchain, &mut queue, f)
            .unwrap();
        self.queue_manager.push_queue(queue);
    }
}
