use yarvk::command::command_buffer::CommandBuffer;
use yarvk::command::command_buffer::Level::{PRIMARY, SECONDARY};
use yarvk::command::command_buffer::RenderPassScope::OUTSIDE;
use yarvk::command::command_buffer::State::{EXECUTABLE, INITIAL};
use yarvk::Extent2D;

use crate::render_device::RenderDevice;
use crate::render_scene::RenderResources;
use crate::render_window::swapchain::ImageViewSwapchain;
use crate::render_window::ImageHandle;

pub mod forward_rendering;

pub trait RenderingFunction {
    fn new(render_device: &RenderDevice, swapchain: &ImageViewSwapchain) -> Self;
    fn record(
        &mut self,
        render_device: &RenderDevice,
        image_handle: &ImageHandle,
        primary_command_buffer: CommandBuffer<{ PRIMARY }, { INITIAL }, { OUTSIDE }>,
        secondary_command_buffer: Vec<CommandBuffer<{ SECONDARY }, { INITIAL }, { OUTSIDE }>>,
        render_details: &RenderResources,
        scale_factor: f64,
        window_size: Extent2D,
    ) -> CommandBuffer<{ PRIMARY }, { EXECUTABLE }, { OUTSIDE }>;
}
