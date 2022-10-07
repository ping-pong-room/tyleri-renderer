use crate::allocator::dedicated_resource::DedicatedImage;
use crate::Renderer;
use rustc_hash::FxHashMap;
use std::pin::Pin;
use std::sync::Arc;
use yarvk::barrier::ImageMemoryBarrier;
use yarvk::command::command_buffer::Level::{PRIMARY, SECONDARY};
use yarvk::command::command_buffer::RenderPassScope::{INSIDE, OUTSIDE};
use yarvk::command::command_buffer::State::{INITIAL, RECORDING};
use yarvk::command::command_buffer::{CommandBuffer, CommandBufferInheritanceInfo};
use yarvk::command::command_pool::CommandPool;

use yarvk::fence::{Fence, SignalingFence};
use yarvk::frame_buffer::Framebuffer;
use yarvk::image_subresource_range::ImageSubresourceRange;
use yarvk::image_view::{ImageView, ImageViewType};
use yarvk::physical_device::SharingMode;
use yarvk::pipeline::pipeline_stage_flags::PipelineStageFlags;
use yarvk::queue::submit_info::{SubmitInfo, SubmitResult, Submittable};
use yarvk::queue::Queue;
use yarvk::render_pass::attachment::{AttachmentDescription, AttachmentReference};
use yarvk::render_pass::render_pass_begin_info::RenderPassBeginInfo;
use yarvk::render_pass::subpass::{SubpassDependency, SubpassDescription};
use yarvk::render_pass::RenderPass;
use yarvk::semaphore::Semaphore;
use yarvk::swapchain::{PresentInfo, Swapchain};
use yarvk::{
    AccessFlags, AttachmentLoadOp, AttachmentStoreOp, ClearColorValue, ClearDepthStencilValue,
    ClearValue, ComponentMapping, ComponentSwizzle, ContinuousImage, DependencyFlags, Format,
    Handle, ImageAspectFlags, ImageLayout, ImageTiling, ImageType, ImageUsageFlags,
    MemoryPropertyFlags, SampleCountFlags, SubpassContents, SUBPASS_EXTERNAL,
};

struct FrameStore {
    renderpass_begin_info: RenderPassBeginInfo,
    inheritance_info: Pin<Arc<CommandBufferInheritanceInfo>>,
    present_complete_semaphore: Semaphore,
    rendering_complete_semaphore: Semaphore,
    fence: Option<SignalingFence<SubmitResult>>,
    primary_command_buffer_handle: u64,
    secondary_command_buffers: Vec<CommandBuffer<{ SECONDARY }, { INITIAL }, { OUTSIDE }, true>>,
}

impl FrameStore {
    pub fn record<
        F: FnOnce(
            &mut [CommandBuffer<{ SECONDARY }, { RECORDING }, { INSIDE }, true>],
        ) -> Result<(), yarvk::Result>,
    >(
        &mut self,
        swapchain: &mut Swapchain,
        present_queue: &mut Queue,
        image: &ContinuousImage,
        f: F,
    ) -> Result<(), yarvk::Result> {
        let (signaled_fence, mut submit_result) = self.fence.take().unwrap().wait()?;
        let fence = signaled_fence.reset()?;
        let mut primary_command_buffer = submit_result
            .take_invalid_primary_buffer(&self.primary_command_buffer_handle)
            .unwrap();
        primary_command_buffer
            .secondary_buffers()
            .iter()
            .map(|(index, _)| *index)
            .collect::<Vec<_>>()
            .iter()
            .for_each(|index| {
                let secondary_buffer = primary_command_buffer
                    .secondary_buffers()
                    .remove(index)
                    .unwrap();
                let secondary_buffer = secondary_buffer.reset().unwrap();
                self.secondary_command_buffers.push(secondary_buffer);
            });

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
                            f,
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

pub struct RenderPassSet {
    pub render_pass: Arc<RenderPass>,
    frame_stores: FxHashMap<u64 /*image handler*/, FrameStore>,
    // This is used for acquire_next_image which require a semaphore before getting the image index
    present_complete_semaphore: Semaphore,
}

impl RenderPassSet {
    fn acquire_next_image(
        &mut self,
        swapchian: &mut Swapchain,
    ) -> Result<(&mut FrameStore, Arc<ContinuousImage>), yarvk::Result> {
        let next_image = swapchian
            .acquire_next_image_semaphore_only(u64::MAX, &self.present_complete_semaphore)?;
        let frame_store = self.frame_stores.get_mut(&next_image.handle()).unwrap();
        std::mem::swap(
            &mut self.present_complete_semaphore,
            &mut frame_store.present_complete_semaphore,
        );
        Ok((frame_store, next_image))
    }
    pub fn render_next_frame<
        F: FnMut(
            &mut [CommandBuffer<{ SECONDARY }, { RECORDING }, { INSIDE }, true>],
        ) -> Result<(), yarvk::Result>,
    >(
        &mut self,
        swapchain: &mut Swapchain,
        present_queue: &mut Queue,
        f: F,
    ) -> Result<(), yarvk::Result> {
        let (frame_store, image) = self.acquire_next_image(swapchain)?;
        frame_store.record(swapchain, present_queue, &image, f)?;
        Ok(())
    }
    pub fn new_render_set(renderer: &mut Renderer) -> Result<RenderPassSet, yarvk::Result> {
        let swapchain = &renderer.swapchain;
        let device = &swapchain.device;
        let present_images = swapchain.get_swapchain_images();
        let surface_format = swapchain.surface.get_physical_device_surface_formats()[0];

        let surface_resolution = swapchain.image_extent;
        let depth_image = ContinuousImage::builder(device.clone())
            .image_type(ImageType::TYPE_2D)
            .format(Format::D16_UNORM)
            .extent(surface_resolution.into())
            .mip_levels(1)
            .array_layers(1)
            .samples(SampleCountFlags::TYPE_1)
            .tiling(ImageTiling::OPTIMAL)
            .usage(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(SharingMode::EXCLUSIVE)
            .build()?;

        let memory_type = renderer
            .allocator
            .get_memory_type(&depth_image, MemoryPropertyFlags::DEVICE_LOCAL)
            .unwrap();
        let depth_image = Arc::new(DedicatedImage::new(depth_image, memory_type)?);
        let depth_image_view = ImageView::builder(depth_image.clone())
            .subresource_range(
                ImageSubresourceRange::builder()
                    .aspect_mask(ImageAspectFlags::DEPTH)
                    .level_count(1)
                    .layer_count(1)
                    .build(),
            )
            .format(depth_image.image_create_info.format)
            .view_type(ImageViewType::Type2d)
            .build()?;
        let renderpass = RenderPass::builder(device.clone())
            .add_attachment(
            AttachmentDescription::builder()
                .format(surface_format.format)
                .samples(SampleCountFlags::TYPE_1)
                .load_op(AttachmentLoadOp::CLEAR)
                .store_op(AttachmentStoreOp::STORE)
                .final_layout(ImageLayout::PRESENT_SRC_KHR)
                .build(),
        ).add_attachment(
            AttachmentDescription::builder()
                .format(Format::D16_UNORM)
                .samples(SampleCountFlags::TYPE_1)
                .load_op(AttachmentLoadOp::CLEAR)
                .initial_layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .final_layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .build(),
        ).add_subpass(SubpassDescription::builder()
            .add_color_attachment(
                AttachmentReference::builder()
                    .attachment_index(0)
                    .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .build(),
            )
            .depth_stencil_attachment(
                AttachmentReference::builder()
                    .attachment_index(1)
                    .layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .build(),
            )
            .build()).add_dependency(
            SubpassDependency::builder()
                .src_subpass(SUBPASS_EXTERNAL)
                .add_src_stage_mask(PipelineStageFlags::ColorAttachmentOutput)
                .add_dst_stage_mask(PipelineStageFlags::ColorAttachmentOutput)
                .dst_access_mask(
                    AccessFlags::COLOR_ATTACHMENT_READ | AccessFlags::COLOR_ATTACHMENT_WRITE,
                )
                .build(),
        ).build()?;
        let frame_stores = present_images
            .iter()
            .map(|image| {
                let image_view = ImageView::builder(image.clone())
                    .view_type(ImageViewType::Type2d)
                    .format(surface_format.format)
                    .components(ComponentMapping {
                        r: ComponentSwizzle::R,
                        g: ComponentSwizzle::G,
                        b: ComponentSwizzle::B,
                        a: ComponentSwizzle::A,
                    })
                    .subresource_range(
                        ImageSubresourceRange::builder()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    )
                    .build()?;
                let framebuffer = Framebuffer::builder(renderpass.clone())
                    .add_attachment(0, image_view.clone())
                    .add_attachment(1, depth_image_view.clone())
                    .width(surface_resolution.width)
                    .height(surface_resolution.height)
                    .layers(1)
                    .build(device.clone())?;
                let mut submit_result = SubmitResult::default();
                let present_queue_family = renderer.queue_manager.get_present_queue_family();
                let primary_command_buffer =
                    CommandPool::builder(present_queue_family.clone(), device.clone())
                        .build()
                        .unwrap()
                        .allocate_command_buffer::<{ PRIMARY }>()?;
                let primary_command_buffer_handle = primary_command_buffer.handle();
                let secondary_command_buffers = [0..num_cpus::get()]
                    .iter()
                    .map(|_| {
                        CommandPool::builder(present_queue_family.clone(), device.clone())
                            .build()
                            .unwrap()
                            .allocate_command_buffer::<{ SECONDARY }>()
                            .unwrap()
                    })
                    .collect();
                submit_result.add_primary_buffer(primary_command_buffer);
                let fence = Fence::new_signaling(device.clone(), submit_result)?;
                let renderpass_begin_info =
                    RenderPassBeginInfo::builder(renderpass.clone(), framebuffer.clone())
                        .render_area(surface_resolution.into())
                        .add_clear_value(ClearValue {
                            color: ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 0.0],
                            },
                        })
                        .add_clear_value(ClearValue {
                            depth_stencil: ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        })
                        .build();
                let inheritance_info = CommandBufferInheritanceInfo::builder()
                    .render_pass(renderpass.clone())
                    .subpass(0)
                    .build();
                let frame_store = FrameStore {
                    renderpass_begin_info,
                    inheritance_info,
                    present_complete_semaphore: Semaphore::new(device.clone())?,
                    rendering_complete_semaphore: Semaphore::new(device.clone())?,
                    fence: Some(fence),
                    primary_command_buffer_handle,
                    secondary_command_buffers,
                };
                Ok((image.handle(), frame_store))
            })
            .collect::<Result<FxHashMap<u64, FrameStore>, yarvk::Result>>()?;
        // Barrier cmd
        let mut queue = renderer
            .queue_manager
            .take_present_queue_priority_low()
            .unwrap();
        queue.simple_record(|command_buffer| {
            command_buffer.cmd_pipeline_barrier(
                &[PipelineStageFlags::BottomOfPipe],
                &[PipelineStageFlags::LateFragmentTests],
                DependencyFlags::empty(),
                &[],
                &[],
                &[ImageMemoryBarrier::builder(depth_image.clone())
                    .dst_access_mask(
                        AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    )
                    .new_layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .old_layout(ImageLayout::UNDEFINED)
                    .subresource_range(
                        ImageSubresourceRange::builder()
                            .aspect_mask(ImageAspectFlags::DEPTH)
                            .layer_count(1)
                            .level_count(1)
                            .build(),
                    )
                    .build()],
            );
            Ok(())
        })?;
        renderer.queue_manager.push_queue(queue);
        Ok(RenderPassSet {
            render_pass: renderpass,
            frame_stores,
            present_complete_semaphore: Semaphore::new(device.clone())?,
        })
    }
}
