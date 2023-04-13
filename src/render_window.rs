use std::mem::MaybeUninit;
use std::sync::Arc;

use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rustc_hash::FxHashMap;
use yarvk::command::command_buffer::Level::{PRIMARY, SECONDARY};
use yarvk::command::command_buffer::TransientCommandBuffer;
use yarvk::extensions::PhysicalInstanceExtensionType;
use yarvk::fence::{Fence, SignalingFence};
use yarvk::pipeline::pipeline_stage_flags::PipelineStageFlag;
use yarvk::queue::submit_info::{SubmitInfo, SubmitResult, Submittable};
use yarvk::surface::Surface;
use yarvk::swapchain::PresentInfo;
use yarvk::{BoundContinuousImage, Extent2D, Handle};

use crate::render_device::RenderDevice;
use crate::render_scene::{PresentResources, RecordResources};
use crate::render_scene::{RenderResources, RenderScene};
use crate::render_window::swapchain::ImageViewSwapchain;
use crate::rendering_function::RenderingFunction;
use crate::WindowHandle;

pub mod present_image_view;
pub mod swapchain;

pub type ImageHandle = u64;

struct UsingResources {
    present_resources: PresentResources,
    primary_command_buffer_handle: u64,
    record_resources: SignalingFence<SubmitResult>,
    render_resources: RenderResources,
}

pub struct RenderWindow<T: RenderingFunction> {
    window_handle: WindowHandle,
    scale_factor: f64,
    swapchain: ImageViewSwapchain,
    available_render_scene: RenderScene,
    using_resources: FxHashMap<ImageHandle /*image handle*/, UsingResources>,
    rendering_function: T,
}

impl<T: RenderingFunction> RenderWindow<T> {
    pub fn window_handle(&self) -> &WindowHandle {
        &self.window_handle
    }
    pub fn get_swapchain_images(&self) -> &[Arc<BoundContinuousImage>] {
        self.swapchain.swapchain.get_swapchain_images()
    }
    pub fn resolution(&self) -> Extent2D {
        self.swapchain.swapchain.image_extent
    }
    pub fn new(
        window_handle: WindowHandle,
        scale_factor: f64,
        render_device: &RenderDevice,
        resolution: &Extent2D,
    ) -> Self {
        let device = &render_device.device;
        let khr_surface_ext = render_device
            .device
            .physical_device
            .instance
            .get_extension::<{ PhysicalInstanceExtensionType::KhrSurface }>()
            .unwrap();
        let surface = Surface::get_physical_device_surface_support(
            khr_surface_ext.clone(),
            window_handle.display_handle,
            window_handle.window_handle,
            &render_device.present_queue_family,
        )
        .unwrap()
        .expect("cannot find surface for a give device");
        let swapchain = ImageViewSwapchain::new(render_device, &surface, resolution);
        let rendering_function = T::new(render_device, &swapchain);
        let available_render_scene = RenderScene::new(render_device);
        let using_resources = swapchain
            .swapchain
            .get_swapchain_images()
            .iter()
            .map(|image| {
                let mut submit_result = SubmitResult::default();
                let present_queue_family = &render_device.present_queue_family;
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
                (
                    image.handle(),
                    UsingResources {
                        present_resources: PresentResources::new(device),
                        primary_command_buffer_handle,
                        record_resources: fence,
                        render_resources: RenderResources::new(render_device),
                    },
                )
            })
            .collect();

        Self {
            window_handle,
            scale_factor,
            swapchain,
            available_render_scene,
            using_resources,
            rendering_function,
        }
    }
    pub fn render(&mut self, render_device: &RenderDevice) {
        let mut tmp: MaybeUninit<RenderScene> = MaybeUninit::uninit();
        std::mem::swap(&mut self.available_render_scene, unsafe {
            &mut *tmp.as_mut_ptr()
        });
        let RenderScene {
            mut present_resources,
            record_resources,
            render_resources,
        } = unsafe { tmp.assume_init() };
        let image = self
            .swapchain
            .swapchain
            .acquire_next_image_semaphore_only(
                u64::MAX,
                &present_resources.present_complete_semaphore,
            )
            .unwrap();
        let fence = record_resources.fence;
        let primary_command_buffer = record_resources.primary_command_buffer;
        let secondary_command_buffers = record_resources.secondary_command_buffers;
        let primary_command_buffer_handle = primary_command_buffer.handle();
        let command_buffer = self.rendering_function.record(
            &render_device,
            &image.handle(),
            primary_command_buffer,
            secondary_command_buffers,
            &render_resources,
            self.scale_factor,
            self.swapchain.swapchain.image_extent.clone(),
        );
        let submit_info = SubmitInfo::builder()
            .add_wait_semaphore(
                &present_resources.present_complete_semaphore,
                PipelineStageFlag::BottomOfPipe.into(),
            )
            .add_one_time_submit_command_buffer(command_buffer)
            .add_signal_semaphore(&present_resources.rendering_complete_semaphore)
            .build();
        let mut present_queue = render_device
            .present_queues
            .pop()
            .expect("internal error: no queue is available");
        let signaling_fence = Submittable::new()
            .add_submit_info(submit_info)
            .submit(&mut present_queue, fence)
            .unwrap();
        let mut present_info = PresentInfo::builder()
            .add_swapchain_and_image(&mut self.swapchain.swapchain, &image)
            .add_wait_semaphore(&mut present_resources.rendering_complete_semaphore)
            .build();
        present_queue.queue_present(&mut present_info).unwrap();
        render_device.present_queues.push(present_queue);

        // wait previous frame finished
        let mut old_resources = self
            .using_resources
            .insert(
                image.handle(),
                UsingResources {
                    present_resources,
                    primary_command_buffer_handle,
                    record_resources: signaling_fence,
                    render_resources,
                },
            )
            .expect("internal error: not pending resources in last frame");
        let (fence, mut submit_result) = old_resources.record_resources.wait().unwrap();
        let fence = fence.reset().unwrap();
        let mut primary_command_buffer = submit_result
            .take_invalid_primary_buffer(&old_resources.primary_command_buffer_handle)
            .expect("internal error: no command buffer in result");
        let mut secondary_command_buffers =
            Vec::with_capacity(primary_command_buffer.secondary_buffers().len());
        while let Some(secondary_buffer) = primary_command_buffer.secondary_buffers().pop() {
            let secondary_buffer = secondary_buffer.reset().unwrap();
            secondary_command_buffers.push(secondary_buffer);
        }
        let primary_command_buffer = primary_command_buffer.reset().unwrap();

        old_resources.render_resources.clear();
        let mut new_presenting_scene = RenderScene {
            present_resources: old_resources.present_resources,
            record_resources: RecordResources {
                fence,
                primary_command_buffer,
                secondary_command_buffers,
            },
            render_resources: old_resources.render_resources,
        };
        std::mem::swap(&mut self.available_render_scene, &mut new_presenting_scene);
        std::mem::forget(new_presenting_scene);
    }
    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }
    pub fn get_render_scene(&mut self) -> &mut RenderScene {
        &mut self.available_render_scene
    }
}
impl<T: RenderingFunction> Drop for RenderWindow<T> {
    fn drop(&mut self) {
        let resources = std::mem::take(&mut self.using_resources);
        resources.into_iter().for_each(|(_, resources)| {
            resources.record_resources.wait().unwrap();
        })
    }
}
