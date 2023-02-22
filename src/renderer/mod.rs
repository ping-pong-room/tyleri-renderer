use crate::renderer::queue_manager::QueueManager;
use crate::renderer::rendering_function::forward_rendering_function::ForwardRenderingFunction;
use crate::renderer::rendering_function::RenderingFunction;
use raw_window_handle::HasRawWindowHandle;
use std::ffi::CStr;
use std::sync::Arc;
use tyleri_config::gpu_config::GpuConfig;
use yarvk::command::command_buffer::CommandBuffer;
use yarvk::command::command_buffer::Level::SECONDARY;
use yarvk::command::command_buffer::RenderPassScope::INSIDE;
use yarvk::command::command_buffer::State::RECORDING;
use yarvk::debug_utils_messenger::DebugUtilsMessengerCreateInfoEXT;
use yarvk::device::Device;
use yarvk::device_features::PhysicalDeviceFeatures::SamplerAnisotropy;
use yarvk::entry::Entry;
use yarvk::extensions::{PhysicalDeviceExtensionType, PhysicalInstanceExtensionType};
use yarvk::instance::{ApplicationInfo, Instance};
use yarvk::physical_device::SharingMode;
use yarvk::pipeline::pipeline_cache::{PipelineCache, PipelineCacheImpl};
use yarvk::pipeline::{PipelineBuilder, PipelineLayout};
use yarvk::sampler::Sampler;
use yarvk::surface::Surface;
use yarvk::swapchain::Swapchain;
use yarvk::window::enumerate_required_extensions;
use yarvk::{
    BorderColor, CompareOp, CompositeAlphaFlagsKHR, Extent2D, Filter, PresentModeKHR,
    SampleCountFlags, SamplerAddressMode, SamplerMipmapMode, SurfaceTransformFlagsKHR,
};

pub mod queue_manager;
pub mod rendering_function;

pub enum RenderingFunctionType {
    ForwardRendering,
}

pub struct Renderer {
    device: Arc<Device>,
    pub queue_manager: QueueManager,
    pub swapchain: Swapchain,
    forward_rendering_function: ForwardRenderingFunction,
    pub default_sampler: Arc<Sampler>,
    pub msaa_sample_counts: SampleCountFlags,
    pipeline_cache: PipelineCacheImpl<false>,
}

impl Renderer {
    pub fn new(
        config: &GpuConfig,
        window: &dyn HasRawWindowHandle,
        resolution: Extent2D,
    ) -> Result<Self, yarvk::Result> {
        let entry = Entry::load().unwrap();
        let application_info = ApplicationInfo::builder()
            .engine_name(&config.vulkan_application_name)
            .build();
        let mut instance_builder =
            Instance::builder(entry.clone()).application_info(application_info);
        if let Some(level) = config.validation_level {
            let layer =
                unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };
            let debug_utils_messenger_callback = DebugUtilsMessengerCreateInfoEXT::builder()
                .callback(|message_severity, message_type, p_callback_data| {
                    let message_id_number = p_callback_data.message_id_number;
                    let message_id_name = p_callback_data.p_message_id_name;
                    let message = p_callback_data.p_message;
                    println!(
                        "{:?}:\n{:?} [{} ({})] : {}\n",
                        message_severity,
                        message_type,
                        message_id_name,
                        message_id_number.to_string(),
                        message,
                    );
                })
                .severity(*level)
                .build();
            instance_builder = instance_builder
                .add_layer(layer)
                .debug_utils_messenger_exts(vec![debug_utils_messenger_callback]);
        }
        let surface_extensions = enumerate_required_extensions(&window)?;
        for ext in surface_extensions {
            instance_builder = instance_builder.add_extension(&ext);
        }
        let instance = instance_builder.build()?;
        let khr_surface_ext = instance
            .get_extension::<{ PhysicalInstanceExtensionType::KhrSurface }>()
            .unwrap();
        let pdevices = instance.enumerate_physical_devices()?;
        let pdevice = pdevices
            .iter()
            .find(|physical_device| {
                physical_device.get_physical_device_properties().device_id == config.device_id
            })
            .expect("device in config cannot be found");
        let mut queue_manager = QueueManager::new(pdevice.clone())?;
        let surface = Surface::get_physical_device_surface_support(
            khr_surface_ext.clone(),
            &window,
            &queue_manager.get_present_queue_family(),
        )?
        .expect("cannot find surface for a give device");
        let device = queue_manager.get_device();
        let swapchain = Self::create_swapchain(device.clone(), surface, resolution)?;
        let forward_rendering_function =
            ForwardRenderingFunction::new(config, &swapchain, &mut queue_manager)?;
        // create sampler
        let mut sampler_builder = Sampler::builder(&device)
            .mag_filter(Filter::LINEAR)
            .min_filter(Filter::LINEAR)
            .mipmap_mode(SamplerMipmapMode::LINEAR)
            .address_mode_u(SamplerAddressMode::MIRRORED_REPEAT)
            .address_mode_v(SamplerAddressMode::MIRRORED_REPEAT)
            .address_mode_w(SamplerAddressMode::MIRRORED_REPEAT)
            .border_color(BorderColor::FLOAT_OPAQUE_WHITE)
            .compare_op(CompareOp::NEVER);
        let device_limits = device
            .physical_device
            .get_physical_device_properties()
            .limits;
        // config for anisotropy
        if let Some(anisotropy_feature) = device.get_feature::<{ SamplerAnisotropy.into() }>() {
            if let Some(sampler_anisotropy) = &config.sampler_anisotropy {
                sampler_builder =
                    sampler_builder.max_anisotropy(*sampler_anisotropy, anisotropy_feature);
            }
        }
        // config for multisampling
        // use same sample counts for all framebuffer resources
        let sample_counts = device_limits.framebuffer_color_sample_counts.as_raw()
            & device_limits.framebuffer_depth_sample_counts.as_raw()
            & device_limits.framebuffer_stencil_sample_counts.as_raw()
            & device_limits
                .framebuffer_no_attachments_sample_counts
                .as_raw();
        let msaa_sample_counts = if sample_counts & SampleCountFlags::TYPE_64.as_raw() != 0 {
            SampleCountFlags::TYPE_64
        } else if sample_counts & SampleCountFlags::TYPE_32.as_raw() != 0 {
            SampleCountFlags::TYPE_32
        } else if sample_counts & SampleCountFlags::TYPE_16.as_raw() != 0 {
            SampleCountFlags::TYPE_16
        } else if sample_counts & SampleCountFlags::TYPE_8.as_raw() != 0 {
            SampleCountFlags::TYPE_8
        } else if sample_counts & SampleCountFlags::TYPE_4.as_raw() != 0 {
            SampleCountFlags::TYPE_4
        } else if sample_counts & SampleCountFlags::TYPE_2.as_raw() != 0 {
            SampleCountFlags::TYPE_2
        } else {
            SampleCountFlags::TYPE_1
        };

        // pipeline cache
        let pipeline_cache = PipelineCache::builder(&device)
            .initial_data(config.pipeline_data.as_slice())
            .build_internally_synchronized()?;
        let default_sampler = sampler_builder.build()?;
        Ok(Renderer {
            device,
            queue_manager,
            swapchain,
            forward_rendering_function,
            default_sampler,
            msaa_sample_counts,
            pipeline_cache,
        })
    }
    fn create_swapchain(
        device: Arc<Device>,
        surface: Arc<Surface>,
        resolution: Extent2D,
    ) -> Result<Swapchain, yarvk::Result> {
        let swapchian_extension = device
            .get_extension::<{ PhysicalDeviceExtensionType::KhrSwapchain }>()
            .unwrap();
        let surface_format = surface.get_physical_device_surface_formats()[0];
        let surface_capabilities = surface.get_physical_device_surface_capabilities();
        let mut desired_image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0
            && desired_image_count > surface_capabilities.max_image_count
        {
            desired_image_count = surface_capabilities.max_image_count;
        }
        let surface_resolution = match surface_capabilities.current_extent.width {
            u32::MAX => resolution,
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
        Swapchain::builder(surface.clone(), swapchian_extension.clone())
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
    }
    pub fn render_next_frame<
        F: FnMut(
            &mut [CommandBuffer<{ SECONDARY }, { RECORDING }, { INSIDE }>],
        ) -> Result<(), yarvk::Result>,
    >(
        &mut self,
        f: F,
    ) {
        let mut queue = self
            .queue_manager
            .take_present_queue_priority_high()
            .unwrap();
        self.forward_rendering_function
            .record_next_frame(&mut self.swapchain, &mut queue, f)
            .unwrap();
        self.queue_manager.push_queue(queue);
    }
    pub fn on_resolution_changed(
        &mut self,
        config: &GpuConfig,
        resolution: Extent2D,
    ) -> Result<(), yarvk::Result> {
        self.swapchain = Self::create_swapchain(
            self.device.clone(),
            self.swapchain.surface.clone(),
            resolution,
        )?;
        self.forward_rendering_function =
            ForwardRenderingFunction::new(config, &self.swapchain, &mut self.queue_manager)?;
        Ok(())
    }
    pub fn forward_rendering_pipeline_builder(
        &self,
        layout: Arc<PipelineLayout>,
    ) -> PipelineBuilder {
        self.forward_rendering_function
            .pipeline_builder(layout, &self.pipeline_cache, 0)
    }
}
