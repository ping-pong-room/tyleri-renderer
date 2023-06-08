pub mod builders;

use crossbeam_queue::SegQueue;
use std::sync::Arc;
use tyleri_gpu_utils::descriptor::single_image_descriptor_set_layout::SingleImageDescriptorLayout;

use tyleri_gpu_utils::queue::parallel_recording_queue::ParallelRecordingQueue;
use yarvk::device::Device;
use yarvk::physical_device::queue_family_properties::QueueFamilyProperties;
use yarvk::pipeline::pipeline_cache::PipelineCacheImpl;
use yarvk::Format;

use crate::resource::resource_allocator::MemoryAllocator;

pub struct RenderDevice {
    pub(crate) device: Arc<Device>,
    pub(crate) single_image_descriptor_set_layout: SingleImageDescriptorLayout,
    pub(crate) present_queue_family: QueueFamilyProperties,
    pub(crate) present_queues: SegQueue<ParallelRecordingQueue>,
    pub(crate) memory_allocator: MemoryAllocator,
    pub(crate) pipeline_cache: PipelineCacheImpl<false>,
    pub(crate) depth_image_format: Format,
}

impl RenderDevice {}
