use std::sync::Arc;

use yarvk::extensions::PhysicalDeviceExtensionType;
use yarvk::physical_device::SharingMode;
use yarvk::surface::Surface;
use yarvk::swapchain::Swapchain;
use yarvk::{CompositeAlphaFlagsKHR, Extent2D, PresentModeKHR, SurfaceTransformFlagsKHR};

use crate::render_device::RenderDevice;

pub struct ImageViewSwapchain {
    pub swapchain: Swapchain,
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
            .find(|&mode| mode == PresentModeKHR::FIFO)
            .unwrap();
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

        Self { swapchain }
    }
}
