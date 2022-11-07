#![feature(trait_upcasting)]
#![feature(map_first_last)]
#![feature(const_trait_impl)]
#![feature(const_convert)]

extern crate core;

use std::hash::BuildHasherDefault;
use crate::memory_allocator::MemoryAllocator;
use crate::queue_manager::QueueManager;
use crate::renderer_builder::RendererBuilder;
use std::sync::Arc;
use dashmap::DashMap;
use dashmap::mapref::one::{Ref, RefMut};
use rustc_hash::FxHasher;

use yarvk::swapchain::Swapchain;

use yarvk::command::command_buffer::CommandBuffer;
use yarvk::command::command_buffer::Level::SECONDARY;
use yarvk::command::command_buffer::RenderPassScope::INSIDE;
use yarvk::command::command_buffer::State::RECORDING;


use yarvk::pipeline::{PipelineBuilder, PipelineLayout};

use yarvk::{Extent2D, SampleCountFlags};
use yarvk::sampler::Sampler;

pub mod memory_allocator;
pub mod queue_manager;
pub mod render_objects;
pub mod render_resource;
pub mod renderer_builder;
pub mod rendering_function;
pub mod unlimited_descriptor_pool;

use crate::render_resource::texture::{TextureAllocator, TextureSamplerUpdateInfo};
use crate::rendering_function::forward_rendering_function::ForwardRenderingFunction;
use crate::rendering_function::RenderingFunction;
use crate::unlimited_descriptor_pool::UnlimitedDescriptorPool;
pub use renderer_builder::*;

type FxDashMap<K, V> = DashMap<K, V,BuildHasherDefault<FxHasher>>;
type FxRef<'a, K,V> = Ref<'a, K, V,BuildHasherDefault<FxHasher>>;
type FxRefMut<'a, K,V> = RefMut<'a, K, V,BuildHasherDefault<FxHasher>>;

pub enum RenderingFunctionType {
    ForwardRendering,
}

pub struct Renderer {
    pub queue_manager: QueueManager,
    pub swapchain: Swapchain,
    pub memory_allocator: Arc<MemoryAllocator>,
    forward_rendering_function: ForwardRenderingFunction,
    default_sampler: Arc<Sampler>,
    pub texture_allocator: TextureAllocator,
    msaa_sample_counts: SampleCountFlags,
}

impl Renderer {
    pub fn builder() -> RendererBuilder {
        RendererBuilder::new()
    }
    pub fn render_next_frame<
        F: FnMut(
            &UnlimitedDescriptorPool<TextureSamplerUpdateInfo>,
            &mut [CommandBuffer<{ SECONDARY }, { RECORDING }, { INSIDE }, true>],
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
            .record_next_frame(
                &mut self.swapchain,
                &mut queue,
                &self.texture_allocator.descriptor_pool,
                f,
            )
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
