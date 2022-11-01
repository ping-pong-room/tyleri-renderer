use crate::allocator::Allocator;
use crate::queue_manager::QueueManager;

use crate::Renderer;

use raw_window_handle::HasRawWindowHandle;
use std::collections::BTreeMap;
use std::ffi::CStr;
use std::sync::Arc;

use yarvk::debug_utils_messenger::DebugUtilsMessengerCreateInfoEXT;

use yarvk::device_features::PhysicalDeviceFeatures::GeometryShader;

use yarvk::entry::Entry;
use yarvk::extensions::{PhysicalDeviceExtensionType, PhysicalInstanceExtensionType};

use yarvk::instance::{ApplicationInfo, Instance};
use yarvk::physical_device::{PhysicalDevice, SharingMode};

use crate::rendering_function::forward_rendering_function::ForwardRenderingFunction;

use yarvk::device::Device;

use crate::render_resource::texture::TextureSamplerUpdateInfo;
use crate::unlimited_descriptor_pool::UnlimitedDescriptorPool;



use yarvk::sampler::Sampler;
use yarvk::surface::Surface;
use yarvk::swapchain::Swapchain;
use yarvk::window::enumerate_required_extensions;
use yarvk::{
    CompositeAlphaFlagsKHR,
    DebugUtilsMessageSeverityFlagsEXT, Extent2D, PhysicalDeviceType, PresentModeKHR, QueueFlags, SurfaceTransformFlagsKHR,
};

pub struct RendererBuilder {
    validation_level: Option<DebugUtilsMessageSeverityFlagsEXT>,
    vulkan_application_name: String,
    sampler: Option<Arc<Sampler>>,
}

impl RendererBuilder {
    pub(super) fn new() -> Self {
        Self {
            validation_level: None,
            vulkan_application_name: "Tyleri".to_string(),
            sampler: None,
        }
    }
    pub fn enable_validation(mut self, level: DebugUtilsMessageSeverityFlagsEXT) -> Self {
        self.validation_level = Some(level);
        self
    }
    pub fn vulkan_application_name(mut self, name: String) -> Self {
        self.vulkan_application_name = name;
        self
    }
    pub fn sampler(mut self, sampler: Arc<Sampler>) -> Self {
        self.sampler = Some(sampler);
        self
    }
    fn device_score(physical_device: &PhysicalDevice) -> usize {
        let mut score = 0;
        let properties = physical_device.get_physical_device_properties();
        // Discrete GPUs have a significant performance advantage
        if properties.device_type == PhysicalDeviceType::DISCRETE_GPU {
            score += 1000;
        }

        // Maximum possible size of textures affects graphics quality
        score += properties.limits.max_image_dimension2_d;

        let features = physical_device.get_physical_device_features();
        // Application can't function without geometry shaders
        if !features.contains(&GeometryShader) {
            return 0;
        }
        return score as usize;
    }
    fn choose_device(
        instance: Arc<Instance>,
        window: &dyn HasRawWindowHandle,
    ) -> Result<Option<(Arc<PhysicalDevice>, Arc<Surface>)>, yarvk::Result> {
        let khr_surface_ext = instance
            .get_extension::<{ PhysicalInstanceExtensionType::KhrSurface }>()
            .unwrap();
        let mut rank = BTreeMap::new();
        let pdevices = instance.enumerate_physical_devices()?;
        for pdevice in &pdevices {
            for queue_family_properties in &pdevice.get_physical_device_queue_family_properties() {
                if let Some(surface) = Surface::get_physical_device_surface_support(
                    khr_surface_ext.clone(),
                    &window,
                    &queue_family_properties,
                )? {
                    if queue_family_properties
                        .queue_flags
                        .contains(QueueFlags::GRAPHICS)
                    {
                        rank.insert(Self::device_score(&pdevice), (pdevice.clone(), surface));
                    }
                }
            }
        }
        match rank.into_iter().next_back() {
            None => Ok(None),
            Some(last) => Ok(Some(last.1)),
        }
    }
    pub fn build(
        self,
        window: &dyn HasRawWindowHandle,
        resolution: Extent2D,
    ) -> Result<Renderer, yarvk::Result> {
        let entry = Entry::load().unwrap();
        let application_info = ApplicationInfo::builder()
            .engine_name(self.vulkan_application_name)
            .build();
        let mut instance_builder =
            Instance::builder(entry.clone()).application_info(application_info);
        if let Some(level) = self.validation_level {
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
                .severity(level)
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
        let (pdevice, surface) = Self::choose_device(instance.clone(), &window)?.unwrap();
        let mut queue_manager = QueueManager::new(pdevice)?;
        let device = queue_manager.get_device();
        let mut allocator = Allocator::new(device.clone());
        let swapchain = create_swapchain(device.clone(), surface, resolution)?;
        let forward_rendering_function =
            ForwardRenderingFunction::new(&swapchain, &mut queue_manager, &mut allocator)?;
        let texture_sampler_descriptor_pool = UnlimitedDescriptorPool::new(
            device.clone(),
            TextureSamplerUpdateInfo::descriptor_layout(device.clone())?,
        );
        Ok(Renderer {
            queue_manager,
            swapchain,
            allocator,
            forward_rendering_function,
            texture_sampler_descriptor_pool,
        })
    }
}

pub(crate) fn create_swapchain(
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
