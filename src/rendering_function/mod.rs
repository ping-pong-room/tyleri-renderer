use raw_window_handle::RawWindowHandle;
use std::sync::Arc;

use yarvk::command::command_buffer::CommandBuffer;
use yarvk::command::command_buffer::Level::PRIMARY;
use yarvk::command::command_buffer::RenderPassScope::OUTSIDE;
use yarvk::command::command_buffer::State::{EXECUTABLE, INVALID};
use yarvk::BoundContinuousImage;

use crate::display::swapchain::ImageViewSwapchain;
use crate::render_device::RenderDevice;
use crate::render_objects::RenderScene;

pub mod forward_rendering;

pub trait RenderingFunction {
    fn new(render_device: &RenderDevice, swapchain: &ImageViewSwapchain) -> Self;
    fn record(
        &mut self,
        render_device: &RenderDevice,
        image: &Arc<BoundContinuousImage>,
        command_buffer: CommandBuffer<{ PRIMARY }, { INVALID }, { OUTSIDE }>,
        render_scene: &RenderScene,
        display_handle: &RawWindowHandle,
    ) -> CommandBuffer<{ PRIMARY }, { EXECUTABLE }, { OUTSIDE }>;
}
