use crate::allocator::block_allocator::BlockBasedAllocator;
use crate::allocator::dedicated_resource::{DedicatedBuffer, DedicatedImage};

use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use yarvk::device::Device;
use yarvk::device_memory::MemoryRequirement;

use yarvk::physical_device::memory_properties::MemoryType;

use yarvk::physical_device::PhysicalDevice;

use yarvk::{
    ContinuousBufferBuilder, ContinuousImageBuilder, Handle, MemoryPropertyFlags,
    MemoryRequirements,
};

pub mod block_allocator;
pub mod dedicated_resource;

pub trait Image: yarvk::Image {
    fn size(&self) -> u64 {
        self.raw().get_memory_requirements().size
    }
    fn map_host_local_memory(&mut self, f: &dyn Fn(&mut [u8])) -> Result<(), yarvk::Result>;
}

impl Deref for dyn Image {
    type Target = dyn yarvk::Image;

    fn deref(&self) -> &Self::Target {
        self.raw()
    }
}

impl DerefMut for dyn Image {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.raw_mut()
    }
}

pub trait Buffer: yarvk::Buffer {
    fn size(&self) -> u64 {
        self.raw().get_memory_requirements().size
    }
    fn map_host_local_memory(&mut self, f: &dyn Fn(&mut [u8])) -> Result<(), yarvk::Result>;
}

impl Deref for dyn Buffer {
    type Target = dyn yarvk::Buffer;

    fn deref(&self) -> &Self::Target {
        self.raw()
    }
}

impl DerefMut for dyn Buffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.raw_mut()
    }
}

pub trait MemoryBindingBuilder {
    type Ty;
    fn build_and_bind(
        self,
        allocator: &mut Allocator,
        property_flags: MemoryPropertyFlags,
        dedicated: bool,
    ) -> Result<Self::Ty, yarvk::Result>;
}

impl MemoryBindingBuilder for ContinuousBufferBuilder {
    type Ty = Arc<dyn Buffer>;

    fn build_and_bind(
        self,
        allocator: &mut Allocator,
        property_flags: MemoryPropertyFlags,
        dedicated: bool,
    ) -> Result<Self::Ty, yarvk::Result> {
        let buffer = self.build()?;
        let memory_type = allocator
            .get_memory_type(&buffer, property_flags)
            .unwrap()
            .clone();
        if dedicated {
            Ok(Arc::new(DedicatedBuffer::new(buffer, &memory_type)?))
        } else {
            Ok(Arc::new(
                allocator
                    .get_block_based_allocator(&memory_type)
                    .allocate(buffer)
                    .unwrap(),
            ))
        }
    }
}

impl MemoryBindingBuilder for ContinuousImageBuilder {
    type Ty = Arc<dyn Image>;

    fn build_and_bind(
        self,
        allocator: &mut Allocator,
        property_flags: MemoryPropertyFlags,
        dedicated: bool,
    ) -> Result<Self::Ty, yarvk::Result> {
        let image = self.build()?;
        let memory_type = allocator
            .get_memory_type(&image, property_flags)
            .unwrap()
            .clone();
        if dedicated {
            Ok(Arc::new(DedicatedImage::new(image, &memory_type)?))
        } else {
            Ok(Arc::new(
                allocator
                    .get_block_based_allocator(&memory_type)
                    .allocate(image)
                    .unwrap(),
            ))
        }
    }
}

#[derive(Eq, PartialEq, Hash)]
struct MemoryTypeIndex {
    memory_type_bits: u32,
    property_flags: MemoryPropertyFlags,
}

pub struct Allocator {
    pub device: Arc<Device>,
    block_based_allocators: FxHashMap<u64 /*memory type handler*/, Arc<BlockBasedAllocator>>,
    memory_type_map: FxHashMap<MemoryTypeIndex, MemoryType>,
}

impl Allocator {
    pub fn new(device: Arc<Device>) -> Self {
        let allocators = FxHashMap::default();
        Self {
            device,
            block_based_allocators: allocators,
            memory_type_map: Default::default(),
        }
    }
    pub fn get_block_based_allocator(
        &mut self,
        memory_type: &MemoryType,
    ) -> Arc<BlockBasedAllocator> {
        let allocator = self
            .block_based_allocators
            .entry(memory_type.handle())
            .or_insert(Arc::new(BlockBasedAllocator::new(
                self.device.clone(),
                memory_type.clone(),
            )));
        allocator.clone()
    }

    pub fn get_memory_type<T: MemoryRequirement>(
        &mut self,
        t: &T,
        property_flags: MemoryPropertyFlags,
    ) -> Option<&MemoryType> {
        let memory_requirements = t.get_memory_requirements();
        let memory_type_index = MemoryTypeIndex {
            memory_type_bits: memory_requirements.memory_type_bits,
            property_flags,
        };

        let memory_type = match self.memory_type_map.entry(memory_type_index) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                if let Some(memory_type) = Self::get_suitable_memory_type(
                    &self.device.physical_device,
                    memory_requirements,
                    property_flags,
                ) {
                    entry.insert(memory_type)
                } else {
                    return None;
                }
            }
        };
        Some(memory_type)
    }
    fn get_suitable_memory_type(
        physical_device: &PhysicalDevice,
        memory_requirements: &MemoryRequirements,
        property_flags: MemoryPropertyFlags,
    ) -> Option<MemoryType> {
        let memory_properties = physical_device.memory_properties();
        let mut max_heap_size = 0;
        let mut result = None;
        memory_properties
            .memory_types
            .iter()
            .enumerate()
            .for_each(|(index, memory_type)| {
                if (1 << index) & memory_requirements.memory_type_bits != 0
                    && memory_type.property_flags & property_flags == property_flags
                {
                    if memory_type.heap.size > max_heap_size {
                        // find the largest heap
                        max_heap_size = memory_type.heap.size;
                        result = Some(memory_type.clone());
                    }
                }
            });
        return result;
    }
}
