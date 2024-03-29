use std::sync::Arc;

use rayon::iter::ParallelIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator};
use rustc_hash::FxHashMap;
use tyleri_gpu_utils::memory::array_device_memory::ArrayDeviceMemory;
use tyleri_gpu_utils::memory::{try_memory_type, IMemBakImg};
use yarvk::command::command_buffer::Level::{PRIMARY, SECONDARY};
use yarvk::command::command_buffer::RenderPassScope::OUTSIDE;
use yarvk::command::command_buffer::State::{EXECUTABLE, INITIAL};
use yarvk::command::command_buffer::{CommandBuffer, CommandBufferInheritanceInfo};
use yarvk::device_memory::IMemoryRequirements;
use yarvk::frame_buffer::Framebuffer;
use yarvk::image_subresource_range::ImageSubresourceRange;
use yarvk::image_view::{ImageView, ImageViewType};
use yarvk::physical_device::SharingMode;
use yarvk::pipeline::pipeline_stage_flags::PipelineStageFlag;
use yarvk::pipeline::PipelineCacheType;
use yarvk::render_pass::attachment::{AttachmentDescription, AttachmentReference};
use yarvk::render_pass::render_pass_begin_info::RenderPassBeginInfo;
use yarvk::render_pass::subpass::{SubpassDependency, SubpassDescription};
use yarvk::render_pass::RenderPass;
use yarvk::{
    AccessFlags, AttachmentLoadOp, AttachmentStoreOp, ClearColorValue, ClearDepthStencilValue,
    ClearValue, ComponentMapping, ComponentSwizzle, ContinuousImage, Extent2D, Format, Handle,
    ImageAspectFlags, ImageLayout, ImageTiling, ImageType, ImageUsageFlags, MemoryPropertyFlags,
    SampleCountFlags, SubpassContents, SUBPASS_EXTERNAL,
};

use crate::pipeline::common_pipeline::CommonPipeline;
use crate::pipeline::ui_pipeline::UIPipeline;
use crate::render_device::RenderDevice;
use crate::render_scene::RenderResources;
use crate::render_window::swapchain::ImageViewSwapchain;
use crate::render_window::ImageHandle;
use crate::rendering_function::RenderingFunction;

mod stages;

pub(crate) struct FrameStore {
    pub(crate) render_pass_begin_info: Arc<RenderPassBeginInfo>,
    pub(crate) inheritance_info: Arc<CommandBufferInheritanceInfo>,
}

pub struct ForwardRenderingFunction {
    frame_stores: FxHashMap<u64 /*command buffer handler*/, FrameStore>,
    common_pipeline: CommonPipeline,
    ui_pipeline: UIPipeline,
}

impl ForwardRenderingFunction {
    fn create_depth_images(
        render_device: &RenderDevice,
        surface_resolution: Extent2D,
        counts: usize,
    ) -> Option<Vec<Arc<IMemBakImg>>> {
        let device = &render_device.device;
        let depth_image_format = render_device.depth_image_format;
        let mut depth_image_builder = ContinuousImage::builder(device);
        depth_image_builder.image_type(ImageType::TYPE_2D);
        depth_image_builder.format(depth_image_format);
        depth_image_builder.extent(surface_resolution.into());
        depth_image_builder.mip_levels(1);
        depth_image_builder.array_layers(1);
        depth_image_builder.samples(SampleCountFlags::TYPE_1);
        depth_image_builder.tiling(ImageTiling::OPTIMAL);
        depth_image_builder.usage(
            ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | ImageUsageFlags::TRANSIENT_ATTACHMENT,
        );
        depth_image_builder.sharing_mode(SharingMode::EXCLUSIVE);
        let depth_image = depth_image_builder.build().ok().unwrap();
        let memory_requirement = depth_image.get_memory_requirements();
        let result = try_memory_type(
            memory_requirement,
            device.physical_device.memory_properties(),
            Some(MemoryPropertyFlags::LAZILY_ALLOCATED),
            memory_requirement.size * counts as u64,
            |memory_type| {
                return ArrayDeviceMemory::new_resources(
                    &device,
                    &depth_image_builder,
                    counts,
                    &memory_type,
                )
                .ok();
            },
        );
        if let Some(images) = result {
            return Some(images);
        } else {
            try_memory_type(
                memory_requirement,
                device.physical_device.memory_properties(),
                None,
                memory_requirement.size * counts as u64,
                |memory_type| {
                    return ArrayDeviceMemory::new_resources(
                        &device,
                        &depth_image_builder,
                        counts,
                        &memory_type,
                    )
                    .ok();
                },
            )
        }
    }
}

impl RenderingFunction for ForwardRenderingFunction {
    fn new(render_device: &RenderDevice, swapchain: &ImageViewSwapchain) -> Self {
        let device = &render_device.device;
        let present_images = swapchain.swapchain.get_swapchain_images();
        let surface_format = swapchain
            .swapchain
            .surface
            .get_physical_device_surface_formats()[0];
        let surface_resolution = swapchain.swapchain.image_extent;
        let render_pass = RenderPass::builder(&device)
            .add_attachment(
                AttachmentDescription::builder()
                    .format(surface_format.format)
                    .samples(SampleCountFlags::TYPE_1)
                    .load_op(AttachmentLoadOp::CLEAR)
                    .store_op(AttachmentStoreOp::STORE)
                    .final_layout(ImageLayout::PRESENT_SRC_KHR)
                    .build(),
            )
            .add_attachment(
                AttachmentDescription::builder()
                    .format(Format::D16_UNORM)
                    .samples(SampleCountFlags::TYPE_1)
                    .load_op(AttachmentLoadOp::CLEAR)
                    .initial_layout(ImageLayout::UNDEFINED)
                    .final_layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .build(),
            )
            .add_subpass(
                SubpassDescription::builder()
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
                    .build(),
            )
            .add_dependency(
                SubpassDependency::builder()
                    .src_subpass(SUBPASS_EXTERNAL)
                    .add_src_stage_mask(PipelineStageFlag::ColorAttachmentOutput.into())
                    .add_dst_stage_mask(PipelineStageFlag::ColorAttachmentOutput.into())
                    .dst_access_mask(
                        AccessFlags::COLOR_ATTACHMENT_READ | AccessFlags::COLOR_ATTACHMENT_WRITE,
                    )
                    .build(),
            )
            .build()
            .unwrap();
        let depth_images =
            Self::create_depth_images(&render_device, surface_resolution, present_images.len())
                .expect("no available memories for creating depth image");
        let frame_stores = present_images
            .par_iter()
            .enumerate()
            .map(|(index, image)| {
                // depth image
                let depth_image_view = ImageView::builder(depth_images[index].clone())
                    .subresource_range(
                        ImageSubresourceRange::builder()
                            .aspect_mask(ImageAspectFlags::DEPTH)
                            .level_count(1)
                            .layer_count(1)
                            .build(),
                    )
                    .format(render_device.depth_image_format)
                    .view_type(ImageViewType::Type2d)
                    .build()
                    .unwrap();
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
                    .build()
                    .unwrap();
                let framebuffer = Framebuffer::builder(render_pass.clone())
                    .add_attachment(0, image_view.clone())
                    .add_attachment(1, depth_image_view.clone())
                    .width(surface_resolution.width)
                    .height(surface_resolution.height)
                    .layers(1)
                    .build(device)
                    .unwrap();
                let render_pass_begin_info = Arc::new(
                    RenderPassBeginInfo::builder(render_pass.clone(), framebuffer.clone())
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
                        .build(),
                );
                let inheritance_info = CommandBufferInheritanceInfo::builder()
                    .render_pass(render_pass.clone())
                    .subpass(0)
                    .build();
                let frame_store = FrameStore {
                    render_pass_begin_info,
                    inheritance_info,
                };
                Ok((image.handle(), frame_store))
            })
            .collect::<Result<FxHashMap<ImageHandle, FrameStore>, yarvk::Result>>()
            .unwrap();
        let common_pipeline = CommonPipeline::new(
            &render_device.single_image_descriptor_set_layout,
            PipelineCacheType::InternallySynchronized(&render_device.pipeline_cache),
            &render_pass,
            0,
        );
        let ui_pipeline = UIPipeline::new(
            &render_device.single_image_descriptor_set_layout,
            PipelineCacheType::InternallySynchronized(&render_device.pipeline_cache),
            &render_pass,
            0,
        );
        Self {
            frame_stores,
            common_pipeline,
            ui_pipeline,
        }
    }

    fn record(
        &mut self,
        render_device: &RenderDevice,
        image_handle: &ImageHandle,
        primary_command_buffer: CommandBuffer<{ PRIMARY }, { INITIAL }, { OUTSIDE }>,
        secondary_command_buffer: Vec<CommandBuffer<{ SECONDARY }, { INITIAL }, { OUTSIDE }>>,
        render_details: &RenderResources,
        scale_factor: f64,
        window_size: Extent2D,
    ) -> CommandBuffer<{ PRIMARY }, { EXECUTABLE }, { OUTSIDE }> {
        let frame_store = self
            .frame_stores
            .get(image_handle)
            .expect("internal error: frame store not exist");
        let primary_command_buffer = primary_command_buffer.begin().unwrap();
        let mut primary_command_buffer = primary_command_buffer.cmd_begin_render_pass(
            frame_store.render_pass_begin_info.clone(),
            SubpassContents::SECONDARY_COMMAND_BUFFERS,
        );

        let mut secondary_command_buffers: Vec<_> = secondary_command_buffer
            .into_par_iter()
            .map(|secondary_command_buffer| {
                secondary_command_buffer
                    .begin(frame_store.inheritance_info.clone())
                    .unwrap()
            })
            .collect();
        // TODO order and execute all command at in parallel at once
        self.on_render_ui(
            window_size,
            scale_factor,
            render_details,
            &mut secondary_command_buffers[0],
        );
        let cameras = render_details.cameras.as_slice();
        for camera in cameras {
            let mesh_renderers = camera.get_and_order_meshes();
            secondary_command_buffers
                .par_iter_mut()
                .enumerate()
                .for_each(|(index, command_buffer)| {
                    self.on_start(camera, command_buffer);
                    self.on_render_meshes(
                        render_device,
                        camera,
                        &mesh_renderers,
                        index,
                        command_buffer,
                    );
                });
        }

        let secondary_command_buffer: Vec<_> = secondary_command_buffers
            .into_par_iter()
            .map(|secondary_command_buffer| secondary_command_buffer.end().unwrap())
            .collect();
        primary_command_buffer.cmd_execute_commands(secondary_command_buffer);
        let primary_command_buffer = primary_command_buffer.cmd_end_render_pass();
        let primary_command_buffer = primary_command_buffer.end().unwrap();
        primary_command_buffer
    }
}
