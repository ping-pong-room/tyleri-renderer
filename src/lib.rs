#![feature(trait_upcasting)]
#![feature(const_trait_impl)]
#![feature(const_convert)]

extern crate core;

use std::hash::BuildHasherDefault;

use dashmap::DashMap;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use rustc_hash::FxHasher;

pub use rendering_function::forward_rendering::ForwardRenderingFunction;

mod pipeline;
pub mod render_device;
pub mod render_objects;
pub mod render_scene;
pub mod render_window;
mod rendering_function;
pub mod resource;

type FxDashMap<K, V> = DashMap<K, V, BuildHasherDefault<FxHasher>>;

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct WindowHandle {
    pub window_handle: RawWindowHandle,
    pub display_handle: RawDisplayHandle,
}

// Only use RawWindowHandle for hashing, don't know it is safe to send, I'll come back later
unsafe impl Send for WindowHandle {}

unsafe impl Sync for WindowHandle {}
