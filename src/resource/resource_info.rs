use std::sync::Arc;
use tyleri_gpu_utils::memory::{try_memory_type, MemoryObjectBuilder};
use yarvk::device::Device;
use yarvk::device_memory::IMemoryRequirements;
use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::physical_device::SharingMode;
use yarvk::{
    BufferUsageFlags, ContinuousBuffer, ContinuousBufferBuilder, ContinuousImage,
    ContinuousImageBuilder, Extent3D, Format, ImageTiling, ImageType, ImageUsageFlags,
    MemoryPropertyFlags, SampleCountFlags,
};

pub struct ResourcesInfo {
    pub static_vertices_info: ResCreateInfo<ContinuousBufferBuilder>,
    pub static_indices_info: ResCreateInfo<ContinuousBufferBuilder>,
    pub ui_vertices_info: ResCreateInfo<ContinuousBufferBuilder>,
    pub ui_indices_info: ResCreateInfo<ContinuousBufferBuilder>,
    pub texture_info: ResCreateInfo<ContinuousImageBuilder>,
}

impl ResourcesInfo {
    pub fn new(device: &Arc<Device>) -> Self {
        Self {
            static_vertices_info: Self::create_vertices_info(device, false),
            static_indices_info: Self::create_indices_info(device, false),
            ui_vertices_info: Self::create_vertices_info(device, true),
            ui_indices_info: Self::create_indices_info(device, true),
            texture_info: Self::create_texture_info(device),
        }
    }
    fn create_indices_info(
        device: &Arc<Device>,
        host_memory: bool,
    ) -> ResCreateInfo<ContinuousBufferBuilder> {
        let device_memory_properties = device.physical_device.memory_properties();
        let mut buffer_builder = ContinuousBuffer::builder(&device);
        buffer_builder.sharing_mode(SharingMode::EXCLUSIVE);
        buffer_builder.size(1);
        let usage = if host_memory {
            BufferUsageFlags::INDEX_BUFFER
        } else {
            BufferUsageFlags::INDEX_BUFFER | BufferUsageFlags::TRANSFER_DST
        };
        buffer_builder.usage(usage);
        let index_buffer = buffer_builder.build().unwrap();
        let index_buffer_memory_req = index_buffer.get_memory_requirements();
        let memory_type = try_memory_type(
            index_buffer_memory_req,
            device_memory_properties,
            if host_memory {
                Some(MemoryPropertyFlags::HOST_VISIBLE)
            } else {
                None
            },
            1024 * 1024 * 1024,
            |memory_type| Some(memory_type.clone()),
        )
        .unwrap();
        ResCreateInfo { usage, memory_type }
    }

    fn create_vertices_info(
        device: &Arc<Device>,
        host_memory: bool,
    ) -> ResCreateInfo<ContinuousBufferBuilder> {
        let device_memory_properties = device.physical_device.memory_properties();
        let mut buffer_builder = ContinuousBuffer::builder(&device);
        buffer_builder.sharing_mode(SharingMode::EXCLUSIVE);
        buffer_builder.size(1);
        let usage = if host_memory {
            BufferUsageFlags::VERTEX_BUFFER
        } else {
            BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::TRANSFER_DST
        };
        buffer_builder.usage(usage);
        let vertices = buffer_builder.build().unwrap();
        let vertices_buffer_memory_req = vertices.get_memory_requirements();
        let memory_type = try_memory_type(
            vertices_buffer_memory_req,
            device_memory_properties,
            if host_memory {
                Some(MemoryPropertyFlags::HOST_VISIBLE)
            } else {
                None
            },
            1024 * 1024 * 1024,
            |memory_type| Some(memory_type.clone()),
        )
        .unwrap();
        ResCreateInfo { usage, memory_type }
    }

    fn create_texture_info(device: &Arc<Device>) -> ResCreateInfo<ContinuousImageBuilder> {
        let device_memory_properties = device.physical_device.memory_properties();
        let mut image_builder = ContinuousImage::builder(&device);
        image_builder.image_type(ImageType::TYPE_2D);
        image_builder.format(Format::R8G8B8A8_UNORM);
        image_builder.extent(Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        });
        image_builder.mip_levels(1);
        image_builder.array_layers(1);
        image_builder.samples(SampleCountFlags::TYPE_1);
        image_builder.tiling(ImageTiling::OPTIMAL);
        image_builder.usage(ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST);
        image_builder.sharing_mode(SharingMode::EXCLUSIVE);
        let texture_image = image_builder.build().unwrap();
        let texture_image_memory_req = texture_image.get_memory_requirements();
        let memory_type = try_memory_type(
            texture_image_memory_req,
            device_memory_properties,
            None,
            1024 * 1024 * 1024,
            |memory_type| Some(memory_type.clone()),
        )
        .unwrap();
        ResCreateInfo {
            usage: ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST,
            memory_type,
        }
    }
}

pub struct ResCreateInfo<T: MemoryObjectBuilder> {
    pub usage: T::Usage,
    pub memory_type: MemoryType,
}
