#![feature(trait_upcasting)]
#![feature(const_trait_impl)]
#![feature(const_convert)]

use std::hash::BuildHasherDefault;

use dashmap::DashMap;
use raw_window_handle::HasRawWindowHandle;
use rustc_hash::FxHasher;
use yarvk::extensions::PhysicalInstanceExtensionType;
use yarvk::surface::Surface;
use yarvk::Extent2D;

pub use rendering_function::forward_rendering::ForwardRenderingFunction;

use crate::builders::RenderDeviceBuilder;
use crate::display::Display;
use crate::render_device::RenderDevice;
use crate::render_objects::RenderScene;
use crate::rendering_function::RenderingFunction;

pub mod builders;
mod display;
mod pipeline;
mod render_device;
pub mod render_objects;
mod rendering_function;
mod resource;

type FxDashMap<K, V> = DashMap<K, V, BuildHasherDefault<FxHasher>>;

pub struct Renderer<T: RenderingFunction = ForwardRenderingFunction> {
    render_device: RenderDevice,
    windows: Vec<Display<T>>,
}

impl<T: RenderingFunction> Renderer<T> {
    pub fn new(windows: &[(&dyn HasRawWindowHandle, Extent2D)]) -> Self {
        let render_device = RenderDeviceBuilder::default()
            .target_displays(
                windows
                    .iter()
                    .map(|(window, _)| window.raw_window_handle())
                    .collect(),
            )
            .build();

        let khr_surface_ext = render_device
            .device
            .physical_device
            .instance
            .get_extension::<{ PhysicalInstanceExtensionType::KhrSurface }>()
            .unwrap();
        let windows = windows
            .iter()
            .map(|(window, extent)| {
                let surface = Surface::get_physical_device_surface_support(
                    khr_surface_ext.clone(),
                    *window,
                    &render_device.present_queue.queue_family_property,
                )
                .unwrap()
                .expect("cannot find surface for a give device");

                Display::new(window.raw_window_handle(), &render_device, &surface, extent)
            })
            .collect();
        Renderer {
            render_device,
            windows,
        }
    }
    pub fn render(&mut self, render_scene: &RenderScene) {
        for window in &mut self.windows {
            window.present(&mut self.render_device, &render_scene);
        }
    }
    pub fn on_resolution_changed(&mut self, window: &dyn HasRawWindowHandle, resolution: Extent2D) {
        let khr_surface_ext = self
            .render_device
            .device
            .physical_device
            .instance
            .get_extension::<{ PhysicalInstanceExtensionType::KhrSurface }>()
            .unwrap();
        match self
            .windows
            .iter_mut()
            .find(|old_window| old_window.handle() == window.raw_window_handle())
        {
            None => {
                panic!("window not registered")
            }
            Some(old_window) => {
                let surface = Surface::get_physical_device_surface_support(
                    khr_surface_ext.clone(),
                    window,
                    &self.render_device.present_queue.queue_family_property,
                )
                .unwrap()
                .expect("cannot find surface for a give device");
                *old_window = Display::new(
                    window.raw_window_handle(),
                    &self.render_device,
                    &surface,
                    &resolution,
                );
            }
        }
    }
}
