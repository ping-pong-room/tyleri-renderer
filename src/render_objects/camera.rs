use glam::Mat4;
use raw_window_handle::RawWindowHandle;
use yarvk::{Rect2D, Viewport};

use crate::Renderer;

pub struct Camera {
    view_matrix: Mat4,
    projection_matrix: Mat4,
    display_handle: RawWindowHandle,
    view_port: Viewport,
    scissor: Rect2D,
}

impl Camera {
    pub fn new(renderer: &Renderer, display_handle: &RawWindowHandle) -> Camera {
        let display = renderer
            .windows
            .iter()
            .find(|display| display.handle() == *display_handle)
            .unwrap();
        let resolution = display.resolution();
        Camera {
            view_matrix: Default::default(),
            projection_matrix: Default::default(),
            display_handle: *display_handle,
            view_port: Viewport {
                x: 0.0,
                y: 0.0,
                width: resolution.width as f32,
                height: resolution.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            },
            scissor: Rect2D {
                extent: resolution,
                ..Default::default()
            },
        }
    }
    pub fn set_view(&mut self, view: Mat4) {
        self.view_matrix = view;
    }
    pub(crate) fn get_display_handle(&self) -> RawWindowHandle {
        self.display_handle
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
}
