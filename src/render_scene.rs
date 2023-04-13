use std::sync::Arc;

use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rustc_hash::FxHashMap;
use tyleri_api::data_structure::vertices::UIVertex;
use tyleri_gpu_utils::memory::variable_length_buffer::VariableLengthBuffer;
use yarvk::command::command_buffer::Level::{PRIMARY, SECONDARY};
use yarvk::command::command_buffer::RenderPassScope::OUTSIDE;
use yarvk::command::command_buffer::State::INITIAL;
use yarvk::command::command_buffer::{CommandBuffer, TransientCommandBuffer};
use yarvk::device::Device;
use yarvk::fence::{Fence, UnsignaledFence};
use yarvk::physical_device::queue_family_properties::QueueFamilyProperties;
use yarvk::semaphore::Semaphore;

use crate::render_device::RenderDevice;
use crate::render_objects::camera::Camera;
use crate::render_objects::render_group::RenderGroup;
use crate::render_objects::ui::{UIElement, UI};

const DEFAULT_VERTICES_BUFFER_LEN: usize = 2 * 1024;
const DEFAULT_INDICES_BUFFER_LEN: usize = 1024;

pub(crate) struct PresentResources {
    pub(crate) present_complete_semaphore: Semaphore,
    pub(crate) rendering_complete_semaphore: Semaphore,
}

impl PresentResources {
    pub fn new(device: &Arc<Device>) -> Self {
        Self {
            present_complete_semaphore: Semaphore::new(&device).unwrap(),
            rendering_complete_semaphore: Semaphore::new(&device).unwrap(),
        }
    }
}

pub(crate) struct RecordResources {
    pub(crate) fence: UnsignaledFence,
    pub(crate) primary_command_buffer: CommandBuffer<{ PRIMARY }, { INITIAL }, { OUTSIDE }>,
    pub(crate) secondary_command_buffers:
        Vec<CommandBuffer<{ SECONDARY }, { INITIAL }, { OUTSIDE }>>,
}

impl RecordResources {
    pub fn new(device: &Arc<Device>, queue_family: &QueueFamilyProperties) -> Self {
        Self {
            fence: Fence::new(device).unwrap(),
            primary_command_buffer: TransientCommandBuffer::<{ PRIMARY }>::new(
                device,
                queue_family.clone(),
            )
            .unwrap(),
            secondary_command_buffers: (0..rayon::current_num_threads())
                .into_par_iter()
                .map(|_| {
                    TransientCommandBuffer::<{ SECONDARY }>::new(device, queue_family.clone())
                        .unwrap()
                })
                .collect(),
        }
    }
}

pub struct RenderResources {
    pub(crate) ui_vertices: Arc<VariableLengthBuffer<UIVertex>>,
    pub(crate) ui_indices: Arc<VariableLengthBuffer<u32>>,
    pub(crate) render_group: FxHashMap<usize, RenderGroup>,
    pub(crate) cameras: Vec<Camera>,
    pub(crate) ui: Vec<UIElement>,
}

impl RenderResources {
    pub fn new(render_device: &RenderDevice) -> Self {
        let ui_vertices = Arc::new(VariableLengthBuffer::new(
            &render_device.device,
            &render_device
                .memory_allocator
                .resource_infos
                .ui_vertices_info
                .memory_type,
            render_device
                .memory_allocator
                .resource_infos
                .ui_vertices_info
                .usage,
            DEFAULT_VERTICES_BUFFER_LEN,
        ));
        let ui_dices = Arc::new(VariableLengthBuffer::new(
            &render_device.device,
            &render_device
                .memory_allocator
                .resource_infos
                .ui_indices_info
                .memory_type,
            render_device
                .memory_allocator
                .resource_infos
                .ui_indices_info
                .usage,
            DEFAULT_INDICES_BUFFER_LEN,
        ));
        let render_group = (0..rayon::current_num_threads())
            .into_iter()
            .map(|index| (index, RenderGroup::new()))
            .collect();
        Self {
            ui_vertices,
            ui_indices: ui_dices,
            render_group,
            cameras: vec![],
            ui: Default::default(),
        }
    }
    pub(crate) fn clear(&mut self) {
        let ui_indices = Arc::get_mut(&mut self.ui_indices)
            .expect("internal error: index buffer is holding by others");
        ui_indices.clear();
        let ui_vertices = Arc::get_mut(&mut self.ui_vertices)
            .expect("internal error: vertex buffer is holding by others");
        ui_vertices.clear();
        self.cameras.clear();
        self.render_group.clear();
    }
}

pub struct RenderScene {
    pub(crate) present_resources: PresentResources,
    pub(crate) record_resources: RecordResources,
    pub(crate) render_resources: RenderResources,
}

impl RenderScene {
    pub fn new(render_device: &RenderDevice) -> Self {
        Self {
            present_resources: PresentResources::new(&render_device.device),
            record_resources: RecordResources::new(
                &render_device.device,
                &render_device.present_queue_family,
            ),
            render_resources: RenderResources::new(render_device),
        }
    }
}
