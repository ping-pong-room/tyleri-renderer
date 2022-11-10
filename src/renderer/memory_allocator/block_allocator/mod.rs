use crate::renderer::memory_allocator::block_allocator::chunk_manager::{BlockIndex, ChunkManager};
use crate::renderer::memory_allocator::{Buffer, Image};
use derive_more::{Deref, DerefMut};
use parking_lot::RwLock;

use std::fmt::{Debug, Formatter};

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use yarvk::device::Device;
use yarvk::device_memory::State::Unbound;
use yarvk::device_memory::{BindMemory, DeviceMemory, MemoryRequirement};
use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::{
    ContinuousBuffer, ContinuousImage, DeviceSize, MemoryRequirements, RawBuffer, RawImage,
};

mod chunk_manager;

pub trait IBlockBasedResource {
    fn get_block_index(&self) -> &BlockIndex;
}

#[derive(Deref, DerefMut)]
pub struct BlockBasedResource<T: BindMemory> {
    #[deref]
    #[deref_mut]
    yarvk_resource: T::BoundType,
    block_index: BlockIndex,
    allocator: Arc<BlockBasedAllocator>,
}

impl<T: BindMemory> IBlockBasedResource for BlockBasedResource<T> {
    fn get_block_index(&self) -> &BlockIndex {
        &self.block_index
    }
}

impl Buffer for BlockBasedResource<ContinuousBuffer<{ Unbound }>> {
    fn raw(&self) -> &RawBuffer {
        self.yarvk_resource.raw()
    }

    fn raw_mut(&mut self) -> &mut RawBuffer {
        self.yarvk_resource.raw_mut()
    }
}

impl Image for BlockBasedResource<ContinuousImage<{ Unbound }>> {
    fn raw(&self) -> &RawImage {
        self.yarvk_resource.raw()
    }

    fn raw_mut(&mut self) -> &mut RawImage {
        self.yarvk_resource.raw_mut()
    }
}

pub struct BlockBasedAllocator {
    device: Arc<Device>,
    memory_type: MemoryType,
    chunk_manager: RwLock<ChunkManager<DeviceMemory>>,
    total_size: AtomicU64,
}

impl BlockBasedAllocator {
    pub fn new(device: Arc<Device>, memory_type: MemoryType) -> BlockBasedAllocator {
        BlockBasedAllocator {
            device,
            memory_type,
            chunk_manager: RwLock::new(ChunkManager::default()),
            total_size: AtomicU64::new(0),
        }
    }
    /// Ask vulkan device to allocate a device memory which is large enough to hold `len` bytes.
    pub fn capacity(&self, len: u64) -> Result<&Self, yarvk::Result> {
        // we allocate the space twice then asked, to make sure the memory is big enough to hold
        // resource with any alignment
        let len = len * 2;
        let device_memory = DeviceMemory::builder(&self.memory_type, self.device.clone())
            .allocation_size(len)
            .build()?;
        let _chunk_index = self.chunk_manager.write().add_chunk(device_memory);
        self.total_size.fetch_add(len, Relaxed);
        Ok(self)
    }
    pub fn capacity_with_allocate(
        &self,
        len: u64,
        allocate_len: u64,
    ) -> Result<BlockIndex, yarvk::Result> {
        // we allocate the space twice then asked, to make sure the memory is big enough to hold
        // resource with any alignment
        let len = len * 2;
        let device_memory = DeviceMemory::builder(&self.memory_type, self.device.clone())
            .allocation_size(len)
            .build()?;
        let block_index = self
            .chunk_manager
            .write()
            .add_chunk_and_allocate(device_memory, allocate_len);
        self.total_size.fetch_add(len, Relaxed);
        Ok(block_index)
    }
    /// allocate required sizeof memory, and bind it to passed in resource.
    pub fn allocate<T: BindMemory + MemoryRequirement>(
        self: Arc<Self>,
        t: T,
    ) -> Option<BlockBasedResource<T>> {
        let memory_requirements = t.get_memory_requirements();
        let block_index = self
            .chunk_manager
            .write()
            .allocate(memory_requirements.size, memory_requirements.alignment);
        match block_index {
            None => {
                if let Ok(block_index) = self.capacity_with_allocate(
                    std::cmp::max(self.total_size.load(Relaxed), memory_requirements.size),
                    memory_requirements.size,
                ) {
                    let mut chunk_manager = self.chunk_manager.write();
                    let device_memory = chunk_manager.get_device_memory(&block_index)?;
                    let res = t
                        .bind_memory(device_memory, block_index.offset)
                        .expect("internal error: bind_memory failed");
                    Some(BlockBasedResource {
                        yarvk_resource: res,
                        block_index,
                        allocator: self.clone(),
                    })
                } else {
                    None
                }
            }
            Some(block_index) => {
                let mut chunk_manager = self.chunk_manager.write();
                let device_memory = chunk_manager.get_device_memory(&block_index)?;
                let res = t
                    .bind_memory(device_memory, block_index.offset)
                    .expect("internal error: bind_memory failed");
                Some(BlockBasedResource {
                    yarvk_resource: res,
                    block_index,
                    allocator: self.clone(),
                })
            }
        }
    }

    fn free_block<T: IBlockBasedResource>(&self, t: T) {
        let block_index = t.get_block_index();
        unsafe {
            self.chunk_manager.write().free_unchecked(&block_index);
        }
    }

    /// free unused chunks(device memories)
    pub fn free_unused_chunks(&self) -> &Self {
        self.chunk_manager.write().free_unused_chunk();
        self
    }
    /// return the memory type this allocator used
    pub fn get_memory_type(&self) -> &MemoryType {
        &self.memory_type
    }
}
