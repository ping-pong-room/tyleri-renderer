use std::sync::Arc;

use parking_lot::Mutex;
use tyleri_gpu_utils::memory::block_based_memory::bindless_buffer::BindlessBufferAllocator;
use tyleri_gpu_utils::memory::block_based_memory::BlockBasedAllocator;
use tyleri_gpu_utils::queue::parallel_recording_queue::ParallelRecordingQueue;
use yarvk::device::Device;
use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::{DeviceSize, Handle};

use crate::resource::ResourcesInfo;
use crate::FxDashMap;

const DEFAULT_VERTICES_BUFFER_SIZE: DeviceSize = 4 * 1024;
const DEFAULT_INDICES_BUFFER_SIZE: DeviceSize = 4 * 1024;

pub struct MemoryAllocator {
    pub device: Arc<Device>,
    pub(crate) queue: Mutex<ParallelRecordingQueue>,
    block_based_allocators: FxDashMap<u64 /*memory type handler*/, Arc<BlockBasedAllocator>>,
    pub resource_infos: ResourcesInfo,
    pub vertices_buffer: Arc<BindlessBufferAllocator>,
    pub indices_buffer: Arc<BindlessBufferAllocator>,
}

impl MemoryAllocator {
    pub fn new(device: &Arc<Device>, queue: ParallelRecordingQueue) -> Self {
        let allocators = FxDashMap::default();
        let resource_infos = ResourcesInfo::new(device);
        let vertices_buffer = BindlessBufferAllocator::new(
            DEFAULT_VERTICES_BUFFER_SIZE,
            &resource_infos.vertices_info.memory_type,
            resource_infos.vertices_info.builder.clone(),
        )
        .unwrap();
        let indices_buffer = BindlessBufferAllocator::new(
            DEFAULT_INDICES_BUFFER_SIZE,
            &resource_infos.indices_info.memory_type,
            resource_infos.indices_info.builder.clone(),
        )
        .unwrap();
        Self {
            device: device.clone(),
            queue: Mutex::new(queue),
            block_based_allocators: allocators,
            resource_infos,
            vertices_buffer,
            indices_buffer,
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
