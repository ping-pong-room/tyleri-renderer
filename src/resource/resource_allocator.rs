use std::sync::Arc;

use parking_lot::Mutex;
use tyleri_gpu_utils::memory::block_based_memory::BlockBasedAllocator;
use tyleri_gpu_utils::queue::parallel_recording_queue::ParallelRecordingQueue;
use yarvk::device::Device;
use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::Handle;

use crate::resource::ResourcesInfo;
use crate::FxDashMap;

pub struct MemoryAllocator {
    pub device: Arc<Device>,
    pub(crate) queue: Mutex<ParallelRecordingQueue>,
    block_based_allocators: FxDashMap<u64 /*memory type handler*/, Arc<BlockBasedAllocator>>,
    pub resource_infos: ResourcesInfo,
}

impl MemoryAllocator {
    pub fn new(device: &Arc<Device>, queue: ParallelRecordingQueue) -> Self {
        let allocators = FxDashMap::default();
        let resource_infos = ResourcesInfo::new(device);
        Self {
            device: device.clone(),
            queue: Mutex::new(queue),
            block_based_allocators: allocators,
            resource_infos,
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
