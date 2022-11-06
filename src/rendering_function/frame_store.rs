use crate::render_resource::texture::TextureSamplerUpdateInfo;
use crate::unlimited_descriptor_pool::UnlimitedDescriptorPool;
use std::pin::Pin;
use std::sync::Arc;
use yarvk::command::command_buffer::Level::SECONDARY;
use yarvk::command::command_buffer::RenderPassScope::{INSIDE, OUTSIDE};
use yarvk::command::command_buffer::State::{INITIAL, RECORDING};
use yarvk::command::command_buffer::{CommandBuffer, CommandBufferInheritanceInfo};
use yarvk::fence::SignalingFence;
use yarvk::pipeline::pipeline_stage_flags::PipelineStageFlags;
use yarvk::queue::submit_info::{SubmitInfo, SubmitResult, Submittable};
use yarvk::queue::Queue;
use yarvk::render_pass::render_pass_begin_info::RenderPassBeginInfo;
use yarvk::semaphore::Semaphore;
use yarvk::swapchain::{PresentInfo, Swapchain};
use yarvk::{ContinuousImage, Rect2D, SubpassContents, Viewport};

pub(crate) struct FrameStore {
    pub(crate) renderpass_begin_info: RenderPassBeginInfo,
    pub(crate) inheritance_info: Pin<Arc<CommandBufferInheritanceInfo>>,
    pub(crate) present_complete_semaphore: Semaphore,
    pub(crate) rendering_complete_semaphore: Semaphore,
    pub(crate) fence: Option<SignalingFence<SubmitResult>>,
    pub(crate) primary_command_buffer_handle: u64,
    pub(crate) secondary_command_buffers:
        Vec<CommandBuffer<{ SECONDARY }, { INITIAL }, { OUTSIDE }, true>>,
}

impl FrameStore {
    pub fn record<
        F: FnOnce(
            &UnlimitedDescriptorPool<TextureSamplerUpdateInfo>,
            &mut [CommandBuffer<{ SECONDARY }, { RECORDING }, { INSIDE }, true>],
        ) -> Result<(), yarvk::Result>,
    >(
        &mut self,
        swapchain: &mut Swapchain,
        present_queue: &mut Queue,
        image: &ContinuousImage,
        texture_sampler_descriptor_pool: &UnlimitedDescriptorPool<TextureSamplerUpdateInfo>,
        f: F,
    ) -> Result<(), yarvk::Result> {
        let (signaled_fence, mut submit_result) = self.fence.take().unwrap().wait()?;
        let fence = signaled_fence.reset()?;
        let mut primary_command_buffer = submit_result
            .take_invalid_primary_buffer(&self.primary_command_buffer_handle)
            .unwrap();
        let secondary_buffers = primary_command_buffer.secondary_buffers();
        while let Some(secondary_buffer) = secondary_buffers.pop() {
            let secondary_buffer = secondary_buffer.reset().unwrap();
            self.secondary_command_buffers.push(secondary_buffer);
        }

        let primary_command_buffer = primary_command_buffer.reset()?;
        let primary_command_buffer = primary_command_buffer
            .record(|primary_command_buffer| {
                primary_command_buffer.cmd_begin_render_pass(
                    &self.renderpass_begin_info,
                    SubpassContents::SECONDARY_COMMAND_BUFFERS,
                    |primary_command_buffer| {
                        let mut secondary_buffers = CommandBuffer::<
                            { SECONDARY },
                            { INITIAL },
                            { OUTSIDE },
                            true,
                        >::record_render_pass_continue_buffers(
                            std::mem::take(&mut self.secondary_command_buffers),
                            self.inheritance_info.clone(),
                            |secondary_buffers| {
                                // set viewport and scissors
                                secondary_buffers.iter_mut().for_each(|secondary_buffer| {
                                    secondary_buffer.cmd_set_viewport(&Viewport {
                                        x: 0.0,
                                        y: 0.0,
                                        width: self.renderpass_begin_info.render_area.extent.width
                                            as f32,
                                        height: self.renderpass_begin_info.render_area.extent.height
                                            as f32,
                                        min_depth: 0.0,
                                        max_depth: 1.0,
                                    });
                                    secondary_buffer.cmd_set_scissor(&Rect2D {
                                        extent: self.renderpass_begin_info.render_area.extent,
                                        ..Default::default()
                                    });
                                });
                                f(texture_sampler_descriptor_pool, secondary_buffers)
                            },
                        )?;
                        primary_command_buffer.cmd_execute_commands(&mut secondary_buffers);
                        let _ = std::mem::replace(&mut self.secondary_command_buffers, unsafe {
                            std::mem::transmute(secondary_buffers)
                        });
                        Ok(())
                    },
                )?;
                Ok(())
            })
            .unwrap();
        let submit_info = SubmitInfo::builder()
            .add_wait_semaphore(
                &mut self.present_complete_semaphore,
                PipelineStageFlags::BottomOfPipe,
            )
            .add_one_time_submit_command_buffer(primary_command_buffer)
            .add_signal_semaphore(&mut self.rendering_complete_semaphore)
            .build();
        let fence = Submittable::new()
            .add_submit_info(submit_info)
            .submit(present_queue, fence)?;
        self.fence = Some(fence);
        let mut present_info = PresentInfo::builder()
            .add_swapchain_and_image(swapchain, &image)
            .add_wait_semaphore(&mut self.rendering_complete_semaphore)
            .build();
        present_queue.queue_present(&mut present_info).unwrap();
        Ok(())
    }
}

impl Drop for FrameStore {
    fn drop(&mut self) {
        self.fence.take().unwrap().wait().unwrap();
    }
}
