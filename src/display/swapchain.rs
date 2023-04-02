use std::sync::Arc;

use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rustc_hash::FxHashMap;
use yarvk::command::command_buffer::Level::{PRIMARY, SECONDARY};
use yarvk::command::command_buffer::TransientCommandBuffer;
use yarvk::extensions::PhysicalDeviceExtensionType;
use yarvk::fence::{Fence, SignalingFence};
use yarvk::physical_device::SharingMode;
use yarvk::queue::submit_info::SubmitResult;
use yarvk::semaphore::Semaphore;
use yarvk::surface::Surface;
use yarvk::swapchain::Swapchain;
use yarvk::{
    BoundContinuousImage, CompositeAlphaFlagsKHR, Extent2D, Handle, PresentModeKHR,
    SurfaceTransformFlagsKHR,
};

use crate::render_device::RenderDevice;

pub struct PresentImageView {
    pub present_complete_semaphore: Semaphore,
    pub command_buffer_handle: u64,
    pub image: Arc<BoundContinuousImage>,
    pub signaling_fence: SignalingFence<SubmitResult>,
}

pub struct ImageViewSwapchain {
    pub swapchain: Swapchain,
    semaphores: Vec<Semaphore>,
    command_buffers: FxHashMap<
        u64, /*image view handle*/
        (
            SignalingFence<SubmitResult>,
            u64, /*command buffer handle*/
        ),
    >,
}

impl Drop for ImageViewSwapchain {
    fn drop(&mut self) {
        let command_buffers = std::mem::take(&mut self.command_buffers);
        command_buffers.into_iter().for_each(|(_, (fence, _))| {
            fence.wait().unwrap();
        })
    }
}

impl ImageViewSwapchain {
    pub fn new(
        render_device: &RenderDevice,
        surface: &Arc<Surface>,
        resolution: &Extent2D,
    ) -> Self {
        let device = &render_device.device;
        let swapchian_extension = device
            .get_extension::<{ PhysicalDeviceExtensionType::KhrSwapchain }>()
            .unwrap();
        // TODO notice pipeline about the format
        let surface_format = surface.get_physical_device_surface_formats()[0];
        let surface_capabilities = surface.get_physical_device_surface_capabilities();
        let mut desired_image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0
            && desired_image_count > surface_capabilities.max_image_count
        {
            desired_image_count = surface_capabilities.max_image_count;
        }
        let surface_resolution = match surface_capabilities.current_extent.width {
            u32::MAX => *resolution,
            _ => surface_capabilities.current_extent,
        };
        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(SurfaceTransformFlagsKHR::IDENTITY)
        {
            SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };
        let present_modes = surface.get_physical_device_surface_present_modes();
        let present_mode = present_modes
            .iter()
            .cloned()
            .find(|&mode| mode == PresentModeKHR::MAILBOX)
            .unwrap_or(PresentModeKHR::FIFO);
        let swapchain = Swapchain::builder(surface.clone(), swapchian_extension.clone())
            .min_image_count(desired_image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(surface_resolution)
            .image_sharing_mode(SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped()
            .image_array_layers(1)
            .build()
            .unwrap();
        let present_images = swapchain.get_swapchain_images();
        let command_buffers = present_images
            .iter()
            .map(|image| {
                let mut submit_result = SubmitResult::default();
                let present_queue_family = &render_device.present_queue.queue_family_property;
                let primary_command_buffer = TransientCommandBuffer::<{ PRIMARY }>::new(
                    &device,
                    present_queue_family.clone().clone(),
                )
                .unwrap();
                // TODO configurable command buffer counts
                let secondary_command_buffers = (0..rayon::current_num_threads())
                    .into_par_iter()
                    .map(|_| {
                        TransientCommandBuffer::<{ SECONDARY }>::new(
                            &device,
                            present_queue_family.clone().clone(),
                        )
                        .unwrap()
                    })
                    .collect();
                let primary_command_buffer_handle = primary_command_buffer.handle();
                submit_result.add_primary_buffer(primary_command_buffer, secondary_command_buffers);
                let fence = Fence::new_signaling(device, submit_result).unwrap();
                (image.handle(), (fence, primary_command_buffer_handle))
            })
            .collect();

        let semaphores = present_images
            .iter()
            .map(|_| Semaphore::new(&device).unwrap())
            .collect();

        Self {
            swapchain,
            semaphores,
            command_buffers,
        }
    }
    pub fn take_view(&mut self) -> PresentImageView {
        let present_complete_semaphore =
            self.semaphores.pop().expect("internal error: no semaphore");
        let image = self
            .swapchain
            .acquire_next_image_semaphore_only(u64::MAX, &present_complete_semaphore)
            .unwrap();
        let (signaling_fence, command_buffer_handle) = self
            .command_buffers
            .remove(&image.handle())
            .expect("internal error: no command buffer");
        PresentImageView {
            present_complete_semaphore,
            command_buffer_handle,
            image,
            signaling_fence,
        }
    }
    pub fn return_view(&mut self, present_image_view: PresentImageView) {
        self.semaphores
            .push(present_image_view.present_complete_semaphore);
        self.command_buffers.insert(
            present_image_view.image.handle(),
            (
                present_image_view.signaling_fence,
                present_image_view.command_buffer_handle,
            ),
        );
    }
}
