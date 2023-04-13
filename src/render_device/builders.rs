use crossbeam_queue::SegQueue;
use std::collections::BTreeMap;
use std::ffi::CStr;
use std::sync::Arc;

use tyleri_gpu_utils::queue::parallel_recording_queue::ParallelRecordingQueue;
use yarvk::debug_utils_messenger::DebugUtilsMessengerCreateInfoEXT;
use yarvk::device::{Device, DeviceBuilder, DeviceQueueCreateInfo};
use yarvk::device_features::PhysicalDeviceFeatures::{GeometryShader, SamplerAnisotropy};
use yarvk::device_features::{DeviceFeatures, PhysicalDeviceFeatures};
use yarvk::entry::Entry;
use yarvk::extensions::{DeviceExtensionType, PhysicalInstanceExtensionType};
use yarvk::instance::{ApplicationInfo, Instance};
use yarvk::physical_device::PhysicalDevice;
use yarvk::pipeline::pipeline_cache::{PipelineCache, PipelineCacheImpl};
use yarvk::sampler::Sampler;
use yarvk::surface::Surface;
use yarvk::window::enumerate_required_extensions;
use yarvk::{
    BorderColor, CompareOp, DebugUtilsMessageSeverityFlagsEXT, Filter, Format, PhysicalDeviceType,
    QueueFlags, SamplerAddressMode, SamplerMipmapMode,
};

use crate::pipeline::single_image_descriptor_set_layout::SingleImageDescriptorLayout;
use crate::render_device::RenderDevice;
use crate::resource::resource_allocator::MemoryAllocator;
use crate::WindowHandle;

const DEFAULT_APP_NAME: &str = "Tyleri App";
const DEFAULT_ENGINE_NAME: &str = "Tyleri Engine";
const DEFAULT_DEPTH_IMAGE_FORMAT: Format = Format::D16_UNORM;
const PRESENT_QUEUE_PRIORITY: f32 = 1.0;
const TRANSFER_QUEUE_PRIORITY: f32 = 0.9;

pub struct RenderDeviceBuilder {
    vulkan_application_name: &'static str,
    sampler_anisotropy: Option<f32>,
    validation_level: Option<DebugUtilsMessageSeverityFlagsEXT>,
    device_id: Option<u32>,
    // msaa_sample_counts: Option<SampleCountFlags>,
    depth_image_format: Format,
    pipeline_cache_data: Option<Vec<u8>>,
    target_window_handles: Vec<WindowHandle>,
}

impl Default for RenderDeviceBuilder {
    fn default() -> Self {
        Self {
            vulkan_application_name: DEFAULT_APP_NAME,
            sampler_anisotropy: None,
            validation_level: None,
            device_id: None,
            depth_image_format: DEFAULT_DEPTH_IMAGE_FORMAT,
            pipeline_cache_data: None,
            target_window_handles: vec![],
        }
    }
}

impl RenderDeviceBuilder {
    pub fn application_name(mut self, name: &'static str) -> Self {
        self.vulkan_application_name = name;
        self
    }
    pub fn sampler_anisotropy(mut self, sampler_anisotropy: f32) -> Self {
        self.sampler_anisotropy = Some(sampler_anisotropy);
        self
    }
    pub fn validation_level(mut self, level: DebugUtilsMessageSeverityFlagsEXT) -> Self {
        self.validation_level = Some(level);
        self
    }
    pub fn device_id(mut self, device_id: u32) -> Self {
        self.device_id = Some(device_id);
        self
    }
    // pub fn msaa_sample_counts(mut self, msaa_sample_counts: SampleCountFlags) -> Self {
    //     self.msaa_sample_counts = Some(msaa_sample_counts);
    //     self
    // }
    pub fn depth_image_format(mut self, format: Format) -> Self {
        self.depth_image_format = format;
        self
    }
    pub fn pipeline_cache_data(mut self, data: Vec<u8>) -> Self {
        self.pipeline_cache_data = Some(data);
        self
    }
    pub fn target_windows(mut self, handles: Vec<WindowHandle>) -> Self {
        self.target_window_handles = handles;
        self
    }
    fn create_instance(&self) -> Arc<Instance> {
        let entry = Entry::load().unwrap();
        let application_info = ApplicationInfo::builder()
            .app_name(self.vulkan_application_name)
            .engine_name(DEFAULT_ENGINE_NAME)
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
        for window_handle in &self.target_window_handles {
            for exts in enumerate_required_extensions(window_handle.display_handle).unwrap() {
                instance_builder = instance_builder.add_extension(&exts);
            }
        }
        instance_builder.build().unwrap()
    }
    fn create_physical_device(&self, instance: &Arc<Instance>) -> Arc<PhysicalDevice> {
        if let Some(device_id) = self.device_id {
            instance
                .enumerate_physical_devices()
                .unwrap()
                .iter()
                .find(|physical_device| {
                    physical_device.get_physical_device_properties().device_id == device_id
                })
                .expect(format!("no device id {device_id} found").as_str())
                .clone()
        } else {
            self.choose_device(instance).expect("no available device")
        }
    }
    fn handle_sampler_anisotropy(
        &self,
        physical_device: &PhysicalDevice,
        mut device_builder: DeviceBuilder,
    ) -> DeviceBuilder {
        let support_sampler_anisotropy = physical_device
            .get_physical_device_features()
            .contains(&PhysicalDeviceFeatures::SamplerAnisotropy.into());
        if !support_sampler_anisotropy && self.sampler_anisotropy.is_some() {
            panic!("sampler anisotropy does not support")
        } else if support_sampler_anisotropy {
            device_builder = device_builder.add_feature(DeviceFeatures::SamplerAnisotropy);
            if let Some(sampler_anisotropy) = self.sampler_anisotropy {
                let device_limits = physical_device.get_physical_device_properties().limits;
                if device_limits.max_sampler_anisotropy < sampler_anisotropy {
                    panic!("sampler anisotropy is large than supported")
                }
            }
        }
        device_builder
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
        score as usize
    }
    fn choose_device(&self, instance: &Arc<Instance>) -> Option<Arc<PhysicalDevice>> {
        let khr_surface_ext = instance
            .get_extension::<{ PhysicalInstanceExtensionType::KhrSurface }>()
            .unwrap();
        let mut rank = BTreeMap::new();
        let pdevices = instance.enumerate_physical_devices().unwrap();
        for pdevice in &pdevices {
            'outer: for queue_family_properties in
                &pdevice.get_physical_device_queue_family_properties()
            {
                for window_handle in &self.target_window_handles {
                    let surface = Surface::get_physical_device_surface_support(
                        khr_surface_ext.clone(),
                        window_handle.display_handle,
                        window_handle.window_handle,
                        queue_family_properties,
                    );
                    match surface {
                        Ok(handle) => {
                            if handle.is_none()
                                || !queue_family_properties
                                    .queue_flags
                                    .contains(QueueFlags::GRAPHICS)
                            {
                                continue 'outer;
                            }
                        }
                        Err(_) => {
                            continue 'outer;
                        }
                    }
                }
                rank.insert(Self::device_score(pdevice), pdevice.clone());
            }
        }
        Some(rank.into_iter().next_back()?.1)
    }
    fn create_device(
        &self,
        physical_device: &Arc<PhysicalDevice>,
    ) -> (
        Arc<Device>,
        ParallelRecordingQueue, /*present*/
        ParallelRecordingQueue, /*transform*/
    ) {
        let mut present_queue_family = None;
        let mut transfer_queue_family = None;
        let properties = physical_device.get_physical_device_queue_family_properties();
        for queue_family_properties in &properties {
            let queue_flags = queue_family_properties.queue_flags;
            if queue_flags.contains(QueueFlags::TRANSFER)
                && !queue_flags.contains(QueueFlags::GRAPHICS)
            {
                // This is a dedicated transfer queue
                transfer_queue_family = Some(queue_family_properties);
            } else if queue_flags.contains(QueueFlags::GRAPHICS) {
                if present_queue_family.is_none() {
                    present_queue_family = Some(queue_family_properties);
                } else if transfer_queue_family.is_none() {
                    transfer_queue_family = Some(queue_family_properties);
                }
            }
        }
        let surface_ext = physical_device
            .instance
            .get_extension::<{ PhysicalInstanceExtensionType::KhrSurface }>()
            .unwrap();
        let mut device_builder = Device::builder(&physical_device)
            .add_extension(&DeviceExtensionType::KhrSwapchain(surface_ext));
        device_builder = self.handle_sampler_anisotropy(physical_device, device_builder);
        let present_queue_family = present_queue_family.unwrap();
        let mut present_queue_create_info_builder =
            DeviceQueueCreateInfo::builder(present_queue_family.clone());
        present_queue_create_info_builder =
            present_queue_create_info_builder.add_priority(PRESENT_QUEUE_PRIORITY);
        // Add proper transfer queue if exists.
        if let Some(transfer_queue_family) = transfer_queue_family {
            let transfer_queue_create_info =
                DeviceQueueCreateInfo::builder(transfer_queue_family.clone())
                    .add_priority(TRANSFER_QUEUE_PRIORITY)
                    .build();
            device_builder = device_builder.add_queue_info(transfer_queue_create_info);
        } else {
            if present_queue_family.queue_count > 1 {
                present_queue_create_info_builder =
                    present_queue_create_info_builder.add_priority(TRANSFER_QUEUE_PRIORITY);
            }
        }
        let present_queue_create_info = present_queue_create_info_builder.build();
        let (device, mut queues) = device_builder
            .add_queue_info(present_queue_create_info)
            .build()
            .unwrap();
        let mut present_queues = queues.remove(present_queue_family).unwrap();
        let present_queue = ParallelRecordingQueue::new(present_queues.pop().unwrap()).unwrap();

        let transfer_queue_family =
            transfer_queue_family.expect("tyleri renderer need at least two queues for now");
        let mut transfer_queues = queues.remove(transfer_queue_family).unwrap();
        let transfer_queue = ParallelRecordingQueue::new(transfer_queues.pop().unwrap()).unwrap();
        (device, present_queue, transfer_queue)
    }
    // fn handle_msaa_sample_counts(&self, device_limits: &PhysicalDeviceLimits) {
    //     let supported_sample_counts = device_limits.framebuffer_color_sample_counts.as_raw()
    //         & device_limits.framebuffer_depth_sample_counts.as_raw()
    //         & device_limits.framebuffer_stencil_sample_counts.as_raw()
    //         & device_limits
    //             .framebuffer_no_attachments_sample_counts
    //             .as_raw();
    //     if let Some(sample_counts) = self.msaa_sample_counts {
    //         if sample_counts.as_raw() & supported_sample_counts != sample_counts.as_raw() {
    //             panic!("asked sample counts does not support")
    //         }
    //     }
    // }
    fn create_sampler(&self, device: &Arc<Device>) -> Arc<Sampler> {
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
        // config for anisotropy
        if let Some(sampler_anisotropy) = self.sampler_anisotropy {
            let anisotropy_feature = device
                .get_feature::<{ SamplerAnisotropy.into() }>()
                .expect("internal error: SamplerAnisotropy feature not added");
            sampler_builder =
                sampler_builder.max_anisotropy(sampler_anisotropy, anisotropy_feature);
        }
        sampler_builder.build().unwrap()
    }
    fn create_pipeline_cache(&self, device: &Arc<Device>) -> PipelineCacheImpl<false> {
        let mut pipeline_cache_builder = PipelineCache::builder(&device);
        if let Some(pipeline_cache_data) = &self.pipeline_cache_data {
            // TODO check if cache is valid
            pipeline_cache_builder =
                pipeline_cache_builder.initial_data(pipeline_cache_data.as_slice());
        }
        pipeline_cache_builder
            .build_internally_synchronized()
            .unwrap()
    }
    pub fn build(self) -> RenderDevice {
        let instance = self.create_instance();
        let pdevice = self.create_physical_device(&instance);
        let (device, present_queue, transfer_queue) = self.create_device(&pdevice);
        let present_queue_family = present_queue.queue_family_property.clone();
        let present_queues = SegQueue::new();
        present_queues.push(present_queue);
        // self.handle_msaa_sample_counts(&pdevice.get_physical_device_properties().limits);
        let default_sampler = self.create_sampler(&device);
        let pipeline_cache = self.create_pipeline_cache(&device);
        let single_image_descriptor_set_layout = SingleImageDescriptorLayout::new(&default_sampler);
        let memory_allocator = MemoryAllocator::new(&device, transfer_queue);
        RenderDevice {
            device,
            single_image_descriptor_set_layout,
            present_queue_family,
            present_queues,
            memory_allocator,
            pipeline_cache,
            depth_image_format: self.depth_image_format,
        }
    }
}
