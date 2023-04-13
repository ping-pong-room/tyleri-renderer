use std::sync::Arc;

use parking_lot::Mutex;
use tyleri_api::data_structure::vertices::Vertex;
use tyleri_gpu_utils::memory::block_based_memory::bindless_buffer::BindlessBufferAllocator;
use tyleri_gpu_utils::memory::block_based_memory::BlockBasedAllocator;
use tyleri_gpu_utils::queue::parallel_recording_queue::ParallelRecordingQueue;
use yarvk::device::Device;
use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::Handle;

use crate::resource::resource_info::ResourcesInfo;
use crate::FxDashMap;

const DEFAULT_VERTICES_BUFFER_LEN: usize = 2 * 1024;
const DEFAULT_INDICES_BUFFER_LEN: usize = 1024;

pub struct MemoryAllocator {
    pub device: Arc<Device>,
    pub(crate) queue: Mutex<ParallelRecordingQueue>,
    block_based_allocators: FxDashMap<u64 /*memory type handler*/, Arc<BlockBasedAllocator>>,
    pub resource_infos: ResourcesInfo,
    pub static_vertices_buffer: Arc<BindlessBufferAllocator<Vertex>>,
    pub static_indices_buffer: Arc<BindlessBufferAllocator<u32>>,
}

impl MemoryAllocator {
    pub fn new(device: &Arc<Device>, queue: ParallelRecordingQueue) -> Self {
        let allocators = FxDashMap::default();
        let resource_infos = ResourcesInfo::new(device);
        let vertices_buffer = BindlessBufferAllocator::new(
            device,
            DEFAULT_VERTICES_BUFFER_LEN,
            &resource_infos.static_vertices_info.memory_type,
            resource_infos.static_vertices_info.usage,
        )
        .unwrap();
        let indices_buffer = BindlessBufferAllocator::new(
            device,
            DEFAULT_INDICES_BUFFER_LEN,
            &resource_infos.static_indices_info.memory_type,
            resource_infos.static_indices_info.usage,
        )
        .unwrap();
        Self {
            device: device.clone(),
            queue: Mutex::new(queue),
            block_based_allocators: allocators,
            resource_infos,
            static_vertices_buffer: vertices_buffer,
            static_indices_buffer: indices_buffer,
        }
    }
    pub fn get_block_based_allocator(&self, memory_type: &MemoryType) -> Arc<BlockBasedAllocator> {
        let allocator = self
            .block_based_allocators
            .entry(memory_type.handle())
            .or_insert(BlockBasedAllocator::new(&self.device, memory_type.clone()));
        allocator.clone()
    }
}
