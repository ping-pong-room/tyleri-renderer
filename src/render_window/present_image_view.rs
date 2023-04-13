// use crate::render_device::RenderDevice;
// use crate::render_scene::RenderScene;
// use crate::render_window::RenderWindow;
// use crate::rendering_function::RenderingFunction;
// use std::sync::Arc;
// use yarvk::fence::SignalingFence;
// use yarvk::pipeline::pipeline_stage_flags::PipelineStageFlag;
// use yarvk::queue::submit_info::{SubmitInfo, SubmitResult, Submittable};
// use yarvk::semaphore::Semaphore;
// use yarvk::swapchain::PresentInfo;
// use yarvk::{BoundContinuousImage, Handle};
//
// pub struct Presentable {
//     pub(crate) present_complete_semaphore: Semaphore,
//     pub(crate) command_buffer_handle: u64,
//     pub(crate) image: Arc<BoundContinuousImage>,
//     pub(crate) signaling_fence: SignalingFence<SubmitResult>,
// }
//
// impl Presentable {
//     pub fn present<T: RenderingFunction>(
//         self,
//         render_device: &RenderDevice,
//         mut render_scene: RenderScene,
//         window: &mut RenderWindow<T>,
//     ) {
//         let image = self.image;
//         let command_buffer_handle = self.command_buffer_handle;
//         let mut present_complete_semaphore = self.present_complete_semaphore;
//         let rendering_complete_semaphore = window
//             .render_complete_semaphores
//             .get_mut(&image.handle())
//             .expect("internal error: no semaphore available");
//         let (fence, mut command_buffer) = self.signaling_fence.wait().unwrap();
//         let fence = fence.reset().unwrap();
//         let command_buffer = command_buffer
//             .take_invalid_primary_buffer(&command_buffer_handle)
//             .expect("internal error: no command buffer found in result");
//         let command_buffer = window.rendering_function.record(
//             render_device,
//             &image,
//             command_buffer,
//             &mut render_scene,
//             window.scale_factor,
//         );
//         let submit_info = SubmitInfo::builder()
//             .add_wait_semaphore(
//                 &mut present_complete_semaphore,
//                 PipelineStageFlag::BottomOfPipe.into(),
//             )
//             .add_one_time_submit_command_buffer(command_buffer)
//             .add_signal_semaphore(rendering_complete_semaphore)
//             .build();
//         let mut present_queue = render_device.present_queues.pop().unwrap();
//         let signaling_fence = Submittable::new()
//             .add_submit_info(submit_info)
//             .submit(&mut present_queue, fence)
//             .unwrap();
//         let mut present_info = PresentInfo::builder()
//             .add_swapchain_and_image(&mut window.swapchain.swapchain, &image)
//             .add_wait_semaphore(rendering_complete_semaphore)
//             .build();
//         present_queue.queue_present(&mut present_info).unwrap();
//         window.swapchain.return_view(Presentable {
//             present_complete_semaphore,
//             command_buffer_handle,
//             image,
//             signaling_fence,
//         });
//         render_device.present_queues.push(present_queue);
//         render_device.render_scene_cache.push(render_scene);
//     }
// }
