use std::sync::Arc;

use raw_window_handle::RawWindowHandle;
use rustc_hash::FxHashMap;

use crate::render_objects::camera::Camera;
use crate::render_objects::mesh_renderer::MeshRenderer;
use crate::Renderer;

pub mod camera;
pub mod mesh_renderer;

pub struct RenderScene {
    mesh_group: Vec<Vec<Arc<MeshRenderer>>>,
    mesh_group_cursor: usize,
    cameras: FxHashMap<RawWindowHandle, Vec<Arc<Camera>>>,
}

impl RenderScene {
    pub fn new(renderer: &Renderer) -> Self {
        let cameras = renderer
            .windows
            .iter()
            .map(|display| (display.handle(), Vec::new()))
            .collect();
        Self {
            mesh_group: vec![Vec::new(); rayon::current_num_threads()],
            mesh_group_cursor: 0,
            cameras,
        }
    }
    pub fn add_mesh_renderer(&mut self, mesh_renderer: &Arc<MeshRenderer>) {
        self.mesh_group[self.mesh_group_cursor].push(mesh_renderer.clone());
        self.mesh_group_cursor = (self.mesh_group_cursor + 1) % self.mesh_group.len();
    }
    pub fn add_camera(&mut self, camera: &Arc<Camera>) {
        self.cameras
            .get_mut(&camera.get_display_handle())
            .unwrap()
            .push(camera.clone());
    }
    pub fn clear(&mut self, renderer: &Renderer) {
        for mesh_renderers in &mut self.mesh_group {
            mesh_renderers.clear();
        }
        for window in &renderer.windows {
            if let Some(cameras) = self.cameras.get_mut(&window.handle()) {
                cameras.clear()
            }
        }
    }
    pub(crate) fn get_mesh_group(&self) -> &[Vec<Arc<MeshRenderer>>] {
        self.mesh_group.as_slice()
    }
    pub(crate) fn get_cameras(&self, window: &RawWindowHandle) -> Option<&[Arc<Camera>]> {
        Some(self.cameras.get(window)?.as_slice())
    }
}
