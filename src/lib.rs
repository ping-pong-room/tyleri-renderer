#![feature(trait_upcasting)]
#![feature(map_first_last)]
#![feature(const_trait_impl)]
#![feature(const_convert)]

use dashmap::mapref::one::{Ref, RefMut};
use dashmap::DashMap;
use rustc_hash::FxHasher;
use std::hash::BuildHasherDefault;

pub mod renderer;

type FxDashMap<K, V> = DashMap<K, V, BuildHasherDefault<FxHasher>>;
type FxRef<'a, K, V> = Ref<'a, K, V, BuildHasherDefault<FxHasher>>;
type FxRefMut<'a, K, V> = RefMut<'a, K, V, BuildHasherDefault<FxHasher>>;
