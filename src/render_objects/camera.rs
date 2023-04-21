use std::sync::Arc;

use glam::Mat4;
use yarvk::{Rect2D, Viewport};

use crate::render_objects::mesh_renderer::MeshRenderer;
use crate::render_objects::ParallelGroup;
use crate::render_scene::{RenderResources, RenderScene};

pub struct Camera {
    view_matrix: Mat4,
    projection_matrix: Mat4,
    view_port: Viewport,
    scissor: Rect2D,
}

impl Camera {
    pub fn new() -> Camera {
        Camera {
            view_matrix: Default::default(),
            projection_matrix: Default::default(),
            view_port: Default::default(),
            scissor: Default::default(),
        }
    }
    pub fn set_view(&mut self, view: Mat4) {
        self.view_matrix = view;
    }
    pub(crate) fn get_viewport(&self) -> &Viewport {
        &self.view_port
    }
    pub(crate) fn get_scissor(&self) -> &Rect2D {
        &self.scissor
    }
    pub(crate) fn get_view_matrix(&self) -> &Mat4 {
        &self.view_matrix
    }
    pub(crate) fn get_projection_matrix(&self) -> &Mat4 {
        &self.projection_matrix
    }
    pub(crate) fn get_and_order_meshes(
        &self,
        render_detail: &RenderResources,
    ) -> ParallelGroup<Arc<MeshRenderer>> {
        // TODO order by distance
        let mut parallel_group = ParallelGroup::new();
        for mesh_renderer in &render_detail.mesh_renderers {
            parallel_group.push(mesh_renderer.clone())
        }
        parallel_group
    }
    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.view_port = viewport;
    }
    pub fn set_scissor(&mut self, scissor: Rect2D) {
        self.scissor = scissor;
    }
}

impl RenderScene {
    pub fn add_camera(&mut self, camera: Camera) {
        self.render_resources.cameras.push(camera)
    }
}
