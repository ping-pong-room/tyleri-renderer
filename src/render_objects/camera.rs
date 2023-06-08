use std::sync::Arc;

use crate::render_objects::mesh_renderer::MeshRenderer;
use glam::Mat4;
use yarvk::{Rect2D, Viewport};

use crate::render_objects::ParallelGroup;
use crate::render_scene::RenderScene;

pub struct Camera {
    pub view_matrix: Mat4,
    pub z_near: f32,
    pub z_far: f32,
    pub fov: f32, // in degree
    pub viewport: Viewport,
    pub scissor: Rect2D,
    pub mesh_renderers: Vec<Arc<MeshRenderer>>,
}

impl Camera {
    pub fn new() -> Camera {
        Camera {
            view_matrix: Default::default(),
            z_near: 0.1,
            z_far: 100.0,
            fov: 45.0,
            viewport: Default::default(),
            scissor: Default::default(),
            mesh_renderers: vec![],
        }
    }
    pub(crate) fn get_and_order_meshes(&self) -> ParallelGroup<Arc<MeshRenderer>> {
        // TODO order by distance
        let mut parallel_group = ParallelGroup::new();
        for mesh_renderer in &self.mesh_renderers {
            parallel_group.push(mesh_renderer.clone())
        }
        parallel_group
    }
    pub(crate) fn get_projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(
            self.fov.to_radians(),
            self.viewport.width / self.viewport.height,
            self.z_near,
            self.z_far,
        )
    }
}

impl RenderScene {
    pub fn add_camera(&mut self, camera: Camera) {
        self.render_resources.cameras.push(camera)
    }
}
