use std::sync::Arc;

use raw_window_handle::RawWindowHandle;
use rustc_hash::FxHashMap;
use yarvk::pipeline::pipeline_stage_flags::PipelineStageFlag;
use yarvk::queue::submit_info::{SubmitInfo, Submittable};
use yarvk::semaphore::Semaphore;
use yarvk::surface::Surface;
use yarvk::swapchain::PresentInfo;
use yarvk::{Extent2D, Handle};

use crate::display::swapchain::{ImageViewSwapchain, PresentImageView};
use crate::render_device::RenderDevice;
use crate::render_objects::RenderScene;
use crate::rendering_function::RenderingFunction;

pub mod swapchain;

pub struct Display<T: RenderingFunction> {
    display_handle: RawWindowHandle,
    swapchain: ImageViewSwapchain,
    render_complete_semaphores: FxHashMap<u64 /*image handle*/, Semaphore>,
    rendering_function: T,
}

impl<T: RenderingFunction> Display<T> {
    pub fn handle(&self) -> RawWindowHandle {
        self.display_handle
    }
    pub fn resolution(&self) -> Extent2D {
        self.swapchain.swapchain.image_extent
    }
    pub fn new(
        display_handle: RawWindowHandle,
        render_device: &RenderDevice,
        surface: &Arc<Surface>,
        resolution: &Extent2D,
    ) -> Self {
        let swapchain = ImageViewSwapchain::new(render_device, surface, resolution);
        let render_complete_semaphores = swapchain
            .swapchain
            .get_swapchain_images()
            .iter()
            .map(|image| {
                (
                    image.handle(),
                    Semaphore::new(&render_device.device).unwrap(),
                )
            })
            .collect();
        let rendering_function = T::new(render_device, &swapchain);
        Self {
            display_handle,
            swapchain,
            render_complete_semaphores,
            rendering_function,
        }
    }
    pub fn present(&mut self, render_device: &mut RenderDevice, render_scene: &RenderScene) {
        let present_image_view = self.swapchain.take_view();
        let image = present_image_view.image;
        let command_buffer_handle = present_image_view.command_buffer_handle;
        let mut present_complete_semaphore = present_image_view.present_complete_semaphore;
        let rendering_complete_semaphore = self
            .render_complete_semaphores
            .get_mut(&image.handle())
            .expect("internal error: no semaphore available");
        let (fence, mut command_buffer) = present_image_view.signaling_fence.wait().unwrap();
        let fence = fence.reset().unwrap();
        let command_buffer = command_buffer
            .take_invalid_primary_buffer(&command_buffer_handle)
            .expect("internal error: no command buffer found in result");
        let command_buffer = self.rendering_function.record(
            render_device,
            &image,
            command_buffer,
            render_scene,
            &self.display_handle,
        );
        let submit_info = SubmitInfo::builder()
            .add_wait_semaphore(
                &mut present_complete_semaphore,
                PipelineStageFlag::BottomOfPipe.into(),
            )
            .add_one_time_submit_command_buffer(command_buffer)
            .add_signal_semaphore(rendering_complete_semaphore)
            .build();
        let present_queue = &mut render_device.present_queue;
        let signaling_fence = Submittable::new()
            .add_submit_info(submit_info)
            .submit(present_queue, fence)
            .unwrap();
        let mut present_info = PresentInfo::builder()
            .add_swapchain_and_image(&mut self.swapchain.swapchain, &image)
            .add_wait_semaphore(rendering_complete_semaphore)
            .build();
        present_queue.queue_present(&mut present_info).unwrap();
        self.swapchain.return_view(PresentImageView {
            present_complete_semaphore,
            command_buffer_handle,
            image,
            signaling_fence,
        });
    }
}
