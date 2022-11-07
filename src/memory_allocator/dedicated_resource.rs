use crate::memory_allocator::{Buffer, Image};
use derive_more::{Deref, DerefMut};
use yarvk::device_memory::dedicated_memory::{DedicatedResource, MemoryDedicatedAllocateInfo};
use yarvk::device_memory::State::Unbound;
use yarvk::device_memory::{BindMemory, DeviceMemory, MemoryRequirement};
use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::{ContinuousBuffer, ContinuousImage, RawBuffer, RawImage};

#[derive(Deref, DerefMut)]
pub struct DedicatedBuffer {
    #[deref]
    #[deref_mut]
    yarvk_buffer: ContinuousBuffer,
    device_memory: DeviceMemory,
}

impl Buffer for DedicatedBuffer {
    fn map_host_local_memory(&mut self, f: &dyn Fn(&mut [u8])) -> Result<(), yarvk::Result> {
        self.device_memory
            .map_memory(0, self.device_memory.size, f)?;
        Ok(())
    }
}

impl yarvk::Buffer for DedicatedBuffer {
    fn raw(&self) -> &RawBuffer {
        self.yarvk_buffer.raw()
    }

    fn raw_mut(&mut self) -> &mut RawBuffer {
        self.yarvk_buffer.raw_mut()
    }
}

impl DedicatedBuffer {
    pub fn new(
        buffer: ContinuousBuffer<{ Unbound }>,
        memory_type: &MemoryType,
    ) -> Result<Self, yarvk::Result> {
        let memory_requirements = buffer.get_memory_requirements();
        let device_memory = DeviceMemory::builder(memory_type, buffer.device.clone())
            .allocation_size(memory_requirements.size)
            .dedicated_info(MemoryDedicatedAllocateInfo {
                resource: DedicatedResource::Buffer(&buffer),
            })
            .build()?;
        let buffer = buffer.bind_memory(&device_memory, 0)?;
        Ok(Self {
            yarvk_buffer: buffer,
            device_memory,
        })
    }
}

#[derive(Deref, DerefMut)]
pub struct DedicatedImage {
    #[deref]
    #[deref_mut]
    yarvk_image: ContinuousImage,
    device_memory: DeviceMemory,
}

impl Image for DedicatedImage {
    fn map_host_local_memory(&mut self, f: &dyn Fn(&mut [u8])) -> Result<(), yarvk::Result> {
        self.device_memory
            .map_memory(0, self.device_memory.size, f)?;
        Ok(())
    }
}

impl yarvk::Image for DedicatedImage {
    fn raw(&self) -> &RawImage {
        self.yarvk_image.raw()
    }

    fn raw_mut(&mut self) -> &mut RawImage {
        self.yarvk_image.raw_mut()
    }
}

impl DedicatedImage {
    pub fn new(
        image: ContinuousImage<{ Unbound }>,
        memory_type: &MemoryType,
    ) -> Result<Self, yarvk::Result> {
        let memory_requirements = image.get_memory_requirements();
        let device_memory = DeviceMemory::builder(memory_type, image.device.clone())
            .allocation_size(memory_requirements.size)
            .dedicated_info(MemoryDedicatedAllocateInfo {
                resource: DedicatedResource::Image(&image),
            })
            .build()?;
        let image = image.bind_memory(&device_memory, 0)?;
        Ok(Self {
            yarvk_image: image,
            device_memory,
        })
    }
}
