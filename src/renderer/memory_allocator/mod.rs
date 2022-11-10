use crate::renderer::memory_allocator::block_allocator::BlockBasedAllocator;
use crate::renderer::memory_allocator::dedicated_resource::{DedicatedBuffer, DedicatedImage};

use rustc_hash::{FxHashMap, FxHasher};
use std::hash::BuildHasherDefault;

use dashmap::mapref::entry::Entry;
use dashmap::mapref::one::Ref;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use yarvk::device::Device;
use yarvk::device_memory::{BindMemory, DeviceMemory, MemoryRequirement};

use yarvk::physical_device::memory_properties::{MemoryType, PhysicalDeviceMemoryProperties};

use yarvk::physical_device::PhysicalDevice;

use crate::{FxDashMap, FxRef, FxRefMut};
use yarvk::device_memory::State::{Bound, Unbound};
use yarvk::{
    Buffer, ContinuousBuffer, ContinuousBufferBuilder, ContinuousImage, ContinuousImageBuilder,
    DeviceSize, ExtendsMemoryRequirements2, Extent3D, Handle, Image, MemoryDedicatedRequirements,
    MemoryPropertyFlags, MemoryRequirements,
};

pub mod block_allocator;
pub mod dedicated_resource;
pub mod staging_vector;

fn find_memory_type_index(
    memory_req: &MemoryRequirements,
    memory_prop: &PhysicalDeviceMemoryProperties,
    flags: MemoryPropertyFlags,
) -> Option<MemoryType> {
    memory_prop
        .memory_types
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_req.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(_index, memory_type)| memory_type.clone())
}

pub trait MemoryBindingBuilder {
    type Ty;
    fn build_and_bind(
        self,
        allocator: &Arc<MemoryAllocator>,
        property_flags: MemoryPropertyFlags,
        dedicated: bool,
    ) -> Result<Self::Ty, yarvk::Result>;
}

impl MemoryBindingBuilder for ContinuousBufferBuilder {
    type Ty = Arc<dyn Buffer>;

    fn build_and_bind(
        self,
        allocator: &Arc<MemoryAllocator>,
        property_flags: MemoryPropertyFlags,
        dedicated: bool,
    ) -> Result<Self::Ty, yarvk::Result> {
        let buffer = self.build()?;
        let memory_type = allocator
            .get_memory_type(&buffer, property_flags)
            .unwrap()
            .clone();
        let dedicated_requirements =
            buffer.get_memory_requirements2::<MemoryDedicatedRequirements>();
        if dedicated
            || dedicated_requirements.prefers_dedicated_allocation != 0
            || dedicated_requirements.requires_dedicated_allocation != 0
        {
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
        allocator: &Arc<MemoryAllocator>,
        property_flags: MemoryPropertyFlags,
        dedicated: bool,
    ) -> Result<Self::Ty, yarvk::Result> {
        let image = self.build()?;
        let memory_type = allocator
            .get_memory_type(&image, property_flags)
            .unwrap()
            .clone();
        let dedicated_requirements =
            image.get_memory_requirements2::<MemoryDedicatedRequirements>();
        if dedicated
            || dedicated_requirements.prefers_dedicated_allocation != 0
            || dedicated_requirements.requires_dedicated_allocation != 0
        {
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

pub struct MemoryAllocator {
    pub device: Arc<Device>,
    block_based_allocators: FxDashMap<u64 /*memory type handler*/, Arc<BlockBasedAllocator>>,
    memory_type_map: FxDashMap<MemoryTypeIndex, MemoryType>,
}

impl MemoryAllocator {
    pub fn new(device: Arc<Device>) -> Self {
        let allocators = FxDashMap::default();
        Self {
            device,
            block_based_allocators: allocators,
            memory_type_map: Default::default(),
        }
    }
    pub fn get_block_based_allocator(&self, memory_type: &MemoryType) -> Arc<BlockBasedAllocator> {
        let allocator = self
            .block_based_allocators
            .entry(memory_type.handle())
            .or_insert(Arc::new(BlockBasedAllocator::new(
                self.device.clone(),
                memory_type.clone(),
            )));
        allocator.clone()
    }
    fn get_memory_type<T: MemoryRequirement>(
        &self,
        t: &T,
        property_flags: MemoryPropertyFlags,
    ) -> Option<FxRefMut<MemoryTypeIndex, MemoryType>> {
        let memory_requirements = t.get_memory_requirements();
        let memory_type_index = MemoryTypeIndex {
            memory_type_bits: memory_requirements.memory_type_bits,
            property_flags,
        };

        let memory_type = match self.memory_type_map.entry(memory_type_index) {
            Entry::Occupied(entry) => entry.into_ref(),
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
