use crate::memory_allocator::block_allocator::chunk_manager::{BlockIndex, ChunkManager};
use crate::memory_allocator::{Buffer, Image};
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
use yarvk::{ContinuousBuffer, ContinuousImage, MemoryRequirements, RawBuffer, RawImage};

mod chunk_manager;

pub trait BlockBasedResource {
    type InnerTy;
    fn get_block_index(&self) -> &BlockIndex;
    fn get_inner_mut(&mut self) -> &mut Self::InnerTy;
}

#[derive(Deref, DerefMut)]
pub struct BlockBasedBuffer {
    #[deref]
    #[deref_mut]
    yarvk_buffer: ContinuousBuffer,
    block_index: BlockIndex,
    allocator: Arc<BlockBasedAllocator>,
}

impl Buffer for BlockBasedBuffer {
    fn map_host_local_memory(&mut self, f: &dyn Fn(&mut [u8])) -> Result<(), yarvk::Result> {
        let block_index = self.get_block_index();
        let chunk_len = self.yarvk_buffer.get_memory_requirements().size;
        if let Some(device_memory) = self
            .allocator
            .chunk_manager
            .write()
            .get_device_memory(&block_index)
        {
            Ok(device_memory.map_memory(block_index.offset, chunk_len, f)?)
        } else {
            return Err(yarvk::Result::ERROR_MEMORY_MAP_FAILED);
        }
    }
}

impl MemoryRequirement for BlockBasedBuffer {
    fn get_memory_requirements(&self) -> &MemoryRequirements {
        self.yarvk_buffer.get_memory_requirements()
    }

    fn get_memory_requirements2<T: yarvk::ExtendsMemoryRequirements2 + Default>(&self) -> T {
        self.yarvk_buffer.get_memory_requirements2()
    }
}

impl Debug for BlockBasedBuffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Error").field(&self.block_index).finish()
    }
}

impl Drop for BlockBasedBuffer {
    fn drop(&mut self) {
        self.allocator.free_block(self);
    }
}

impl BlockBasedResource for BlockBasedBuffer {
    type InnerTy = ContinuousBuffer;

    fn get_block_index(&self) -> &BlockIndex {
        &self.block_index
    }

    fn get_inner_mut(&mut self) -> &mut Self::InnerTy {
        &mut self.yarvk_buffer
    }
}

impl yarvk::Buffer for BlockBasedBuffer {
    fn raw(&self) -> &RawBuffer {
        self.yarvk_buffer.raw()
    }

    fn raw_mut(&mut self) -> &mut RawBuffer {
        self.yarvk_buffer.raw_mut()
    }
}

#[derive(Deref, DerefMut)]
pub struct BlockBasedImage {
    #[deref]
    #[deref_mut]
    yarvk_image: ContinuousImage,
    block_index: BlockIndex,
    allocator: Arc<BlockBasedAllocator>,
}

impl Image for BlockBasedImage {
    fn map_host_local_memory(&mut self, f: &dyn Fn(&mut [u8])) -> Result<(), yarvk::Result> {
        let block_index = self.get_block_index();
        let chunk_len = self.yarvk_image.get_memory_requirements().size;
        if let Some(device_memory) = self
            .allocator
            .chunk_manager
            .write()
            .get_device_memory(&block_index)
        {
            Ok(device_memory.map_memory(block_index.offset, chunk_len, f)?)
        } else {
            return Err(yarvk::Result::ERROR_MEMORY_MAP_FAILED);
        }
    }
}

impl MemoryRequirement for BlockBasedImage {
    fn get_memory_requirements(&self) -> &MemoryRequirements {
        self.yarvk_image.get_memory_requirements()
    }

    fn get_memory_requirements2<T: yarvk::ExtendsMemoryRequirements2 + Default>(&self) -> T {
        self.yarvk_image.get_memory_requirements2()
    }
}

impl Drop for BlockBasedImage {
    fn drop(&mut self) {
        self.allocator.free_block(self);
    }
}

impl BlockBasedResource for BlockBasedImage {
    type InnerTy = ContinuousImage;

    fn get_block_index(&self) -> &BlockIndex {
        &self.block_index
    }

    fn get_inner_mut(&mut self) -> &mut Self::InnerTy {
        &mut self.yarvk_image
    }
}

impl yarvk::Image for BlockBasedImage {
    fn raw(&self) -> &RawImage {
        self.yarvk_image.raw()
    }

    fn raw_mut(&mut self) -> &mut RawImage {
        self.yarvk_image.raw_mut()
    }
}

pub trait UnboundResource: BindMemory + MemoryRequirement {
    type CustomType: BlockBasedResource;
    fn new(
        t: Self::BoundType,
        block_index: BlockIndex,
        allocator: Arc<BlockBasedAllocator>,
    ) -> Self::CustomType;
}

impl UnboundResource for ContinuousBuffer<{ Unbound }> {
    type CustomType = BlockBasedBuffer;

    fn new(
        t: Self::BoundType,
        block_index: BlockIndex,
        allocator: Arc<BlockBasedAllocator>,
    ) -> Self::CustomType {
        BlockBasedBuffer {
            yarvk_buffer: t,
            block_index,
            allocator,
        }
    }
}

impl UnboundResource for ContinuousImage<{ Unbound }> {
    type CustomType = BlockBasedImage;

    fn new(
        t: Self::BoundType,
        block_index: BlockIndex,
        allocator: Arc<BlockBasedAllocator>,
    ) -> Self::CustomType {
        BlockBasedImage {
            yarvk_image: t,
            block_index,
            allocator,
        }
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
    pub fn allocate<T: UnboundResource>(self: Arc<Self>, t: T) -> Option<T::CustomType> {
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
                    Some(T::new(res, block_index, self.clone()))
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
                Some(T::new(res, block_index, self.clone()))
            }
        }
    }
    // /// Map a memory object into application address space
    // pub fn update_memory<T: BlockBasedResource>(
    //     &self,
    //     t: &mut T,
    //     f: impl FnOnce(&mut [u8]),
    // ) -> Result<(), yarvk::Result> {
    //     let block_index = t.get_block_index();
    //     let mut chunk_manager = self.chunk_manager.write();
    //     if let Some(chunk_len) = chunk_manager.get_block_len(block_index) {
    //         if let Some(device_memory) = chunk_manager.get_device_memory(&block_index) {
    //             device_memory.update_memory(block_index.offset, chunk_len, f)?
    //         }
    //     }
    //     return Err(yarvk::Result::ERROR_MEMORY_MAP_FAILED);
    // }

    fn free_block<T: BlockBasedResource>(&self, t: &T) {
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
