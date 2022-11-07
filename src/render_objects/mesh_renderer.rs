use std::sync::Arc;
use yarvk::pipeline::Pipeline;
use yarvk::sampler::Sampler;
use crate::memory_allocator::{Buffer, Image};

pub struct MeshRender {
    vertex_buffer: Arc<dyn Buffer>,
    index_buffer: Arc<dyn Buffer>,
    texture_image: Arc<dyn Image>,
    sampler: Arc<Sampler>,
    pipeline: Arc<Pipeline>,
}