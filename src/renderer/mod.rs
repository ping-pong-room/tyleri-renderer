use dashmap::DashMap;
use std::hash::BuildHasherDefault;
use std::sync::Arc;

use rayon::iter::IntoParallelIterator;
use rustc_hash::FxHasher;
use yarvk::command::command_buffer::CommandBuffer;
use yarvk::command::command_buffer::Level::SECONDARY;
use yarvk::command::command_buffer::RenderPassScope::INSIDE;
use yarvk::command::command_buffer::State::RECORDING;
use yarvk::pipeline::{PipelineBuilder, PipelineLayout};
use yarvk::sampler::Sampler;
use yarvk::swapchain::Swapchain;
use yarvk::{Extent2D, SampleCountFlags};

pub use renderer_builder::*;

use crate::renderer::memory_allocator::MemoryAllocator;
use crate::renderer::mesh::Mesh;
use crate::renderer::queue_manager::recordable_queue::RecordableQueue;
use crate::renderer::queue_manager::QueueManager;
use crate::renderer::renderer_builder::RendererBuilder;
use crate::renderer::rendering_function::forward_rendering_function::ForwardRenderingFunction;
use crate::renderer::rendering_function::RenderingFunction;

pub mod memory_allocator;
pub mod mesh;
pub mod queue_manager;
pub mod renderer_builder;
pub mod rendering_function;

pub enum RenderingFunctionType {
    ForwardRendering,
}

pub struct Renderer {
    pub queue_manager: QueueManager,
    pub swapchain: Swapchain,
    pub memory_allocator: Arc<MemoryAllocator>,
    forward_rendering_function: ForwardRenderingFunction,
    default_sampler: Arc<Sampler>,
    msaa_sample_counts: SampleCountFlags,
    meshes: DashMap<u64, Mesh, BuildHasherDefault<FxHasher>>,
}

impl Renderer {
    pub fn builder() -> RendererBuilder {
        RendererBuilder::new()
    }
    pub fn render_next_frame<
        F: FnMut(
            &mut [CommandBuffer<{ SECONDARY }, { RECORDING }, { INSIDE }>],
        ) -> Result<(), yarvk::Result>,
    >(
        &mut self,
        f: F,
    ) {
        let mut queue = self
            .queue_manager
            .take_present_queue_priority_high()
            .unwrap();
        self.forward_rendering_function
            .record_next_frame(&mut self.swapchain, &mut queue, f)
            .unwrap();
        self.queue_manager.push_queue(queue);
    }
    pub fn on_resolution_changed(&mut self, resolution: Extent2D) -> Result<(), yarvk::Result> {
        self.swapchain = create_swapchain(
            self.swapchain.device.clone(),
            self.swapchain.surface.clone(),
            resolution,
        )?;
        self.forward_rendering_function = ForwardRenderingFunction::new(
            &self.swapchain,
            &mut self.queue_manager,
            &mut self.memory_allocator,
        )?;
        Ok(())
    }
    pub fn forward_rendering_pipeline_builder(
        &self,
        layout: Arc<PipelineLayout>,
    ) -> PipelineBuilder {
        self.forward_rendering_function.pipeline_builder(layout, 0)
    }
}
