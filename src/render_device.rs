use std::sync::Arc;

use tyleri_gpu_utils::queue::parallel_recording_queue::ParallelRecordingQueue;
use yarvk::device::Device;
use yarvk::pipeline::pipeline_cache::PipelineCacheImpl;
use yarvk::Format;

use crate::pipeline::single_image_descriptor_set_layout::SingleImageDescriptorLayout;
use crate::resource::resource_allocator::MemoryAllocator;

pub struct RenderDevice {
    pub device: Arc<Device>,
    pub single_image_descriptor_set_layout: SingleImageDescriptorLayout,
    pub present_queue: ParallelRecordingQueue,
    pub memory_allocator: MemoryAllocator,
    pub pipeline_cache: PipelineCacheImpl<false>,
    pub depth_image_format: Format,
}
