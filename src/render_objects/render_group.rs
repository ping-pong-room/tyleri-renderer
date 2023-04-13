use std::sync::Arc;

use crate::render_objects::mesh_renderer::MeshRenderer;

pub struct RenderGroup {
    meshes: Vec<Arc<MeshRenderer>>,
}

impl RenderGroup {
    pub fn new() -> Self {
        Self { meshes: Vec::new() }
    }
    pub fn add_mesh_renderer(&mut self, mesh_renderer: Arc<MeshRenderer>) {
        self.meshes.push(mesh_renderer);
    }
    pub(crate) fn get_meshes(&self) -> &[Arc<MeshRenderer>] {
        self.meshes.as_slice()
    }
}
