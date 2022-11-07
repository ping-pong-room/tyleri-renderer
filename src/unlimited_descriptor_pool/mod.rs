use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::sync::Arc;
use yarvk::descriptor::descriptor_pool::{DescriptorPool, DescriptorPoolCreateFlag};
use yarvk::descriptor::descriptor_set::DescriptorSet;
use yarvk::descriptor::descriptor_set_layout::DescriptorSetLayout;
use yarvk::descriptor::write_descriptor_sets::WriteDescriptorSets;
use yarvk::device::Device;
use yarvk::Handle;

pub trait DescriptorUpdateInfo: Sync + Send + Clone {
    fn add_to_write_descriptor_set<'a>(
        &'a self,
        descriptor_set_index: usize,
        updatable: &mut WriteDescriptorSets<'a>,
    );
}

pub struct SingleSetTypeDescriptorPool<T: DescriptorUpdateInfo> {
    max_sets: u32,
    descriptor_pool: DescriptorPool,
    single_descriptor_set_layout: Arc<DescriptorSetLayout>,
    allocated_descriptors: FxHashMap<usize, T>,
}

impl<T: DescriptorUpdateInfo> SingleSetTypeDescriptorPool<T> {
    pub fn new(
        device: Arc<Device>,
        max_sets: u32,
        single_descriptor_set_layout: Arc<DescriptorSetLayout>,
    ) -> Result<SingleSetTypeDescriptorPool<T>, yarvk::Result> {
        let mut builder = DescriptorPool::builder(device.clone())
            .add_flag(DescriptorPoolCreateFlag::DescriptorPoolCreateFreeDescriptorSet);
        for _ in 0..max_sets {
            for (_, binding) in &single_descriptor_set_layout.bindings {
                builder = builder.add_descriptor_pool_size(
                    &binding.descriptor_type(),
                    binding.descriptor_count(),
                );
            }
        }
        let descriptor_pool = builder.build()?;
        Ok(SingleSetTypeDescriptorPool {
            max_sets,
            descriptor_pool,
            single_descriptor_set_layout,
            allocated_descriptors: Default::default(),
        })
    }
    pub fn allocate<'a>(
        &'a mut self,
        descriptor_set_indices: &'a mut Vec<(usize, T)>,
    ) -> Result<WriteDescriptorSets, yarvk::Result> {
        let mut allocatable = self.descriptor_pool.allocatable();
        for (descriptor_set_index, _) in descriptor_set_indices.iter() {
            allocatable = allocatable.add_descriptor_set_layout(
                *descriptor_set_index,
                self.single_descriptor_set_layout.clone(),
            )
        }
        allocatable.allocate()?;
        let mut write_descriptor_sets = self.descriptor_pool.write_descriptor_sets();
        for (descriptor_set_index, update_value) in descriptor_set_indices.iter() {
            update_value
                .add_to_write_descriptor_set(*descriptor_set_index, &mut write_descriptor_sets);
            self.allocated_descriptors
                .insert(*descriptor_set_index, update_value.clone());
        }
        Ok(write_descriptor_sets)
    }
    pub fn free(&mut self, descriptor_set_indices: &[usize]) -> Result<(), yarvk::Result> {
        for index in descriptor_set_indices {
            self.allocated_descriptors.remove(index);
        }
        self.descriptor_pool
            .free_descriptor_sets(&descriptor_set_indices)
    }
    pub fn get_descriptor_set(&self, descriptor_index: &usize) -> Option<&DescriptorSet> {
        self.descriptor_pool.get_descriptor_set(descriptor_index)
    }
}

struct PendingBufferPool<T: DescriptorUpdateInfo> {
    pool: SingleSetTypeDescriptorPool<T>,
    update_buffer: Vec<(usize, T)>,
    free_buffer: Vec<usize>,
}

pub struct UnlimitedDescriptorPool<T: DescriptorUpdateInfo> {
    device: Arc<Device>,
    // the descriptor set layout the pools used.
    descriptor_layout: Arc<DescriptorSetLayout>,
    // mapping descriptor index into descriptor pool index, used to lookup which pool this descriptor set belongs to.
    descriptor_set_indices: FxHashMap<usize, u64>,
    // pools that have free descriptor set slots.
    not_full_pools: BTreeMap<u64, PendingBufferPool<T>>,
    // pools that are full, cannot allocate descriptor sets any more.
    full_pools: BTreeMap<u64, PendingBufferPool<T>>,
    // a buffer used to hold pools temporary.
    ready_to_go: BTreeMap<u64, PendingBufferPool<T>>,
}

impl<T: DescriptorUpdateInfo> UnlimitedDescriptorPool<T> {
    pub fn new(device: Arc<Device>, descriptor_layout: Arc<DescriptorSetLayout>) -> Self {
        Self {
            device,
            descriptor_layout,
            descriptor_set_indices: Default::default(),
            not_full_pools: Default::default(),
            full_pools: Default::default(),
            ready_to_go: Default::default(),
        }
    }
    // TODO do not unwrap
    pub fn allocate<I: IntoIterator<Item = (usize, T)>>(&mut self, descriptor_indices: I) {
        let mut descriptor_indices = descriptor_indices.into_iter();
        self.prepare_allocate(&mut descriptor_indices);
        // not enough room for wanted descriptor sets, allocate more.
        while descriptor_indices.size_hint().0 > 0 {
            let max_set = descriptor_indices.size_hint().0;
            let pool = SingleSetTypeDescriptorPool::new(
                self.device.clone(),
                max_set as _,
                self.descriptor_layout.clone(),
            )
            .unwrap();
            self.not_full_pools.insert(
                pool.descriptor_pool.handle(),
                PendingBufferPool {
                    pool,
                    update_buffer: Vec::with_capacity(max_set),
                    free_buffer: Vec::with_capacity(max_set),
                },
            );
            self.prepare_allocate(&mut descriptor_indices);
        }
        let mut descriptor_wirtes = self
            .ready_to_go
            .par_iter_mut()
            .map(|(_, buffer_pool)| {
                buffer_pool
                    .pool
                    .allocate(&mut buffer_pool.update_buffer)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        self.device
            .par_update_descriptor_sets(&mut descriptor_wirtes);
        // put pools back to full vector or not full vector
        // only the last element can be not full
        if let Some((index, buffer_pool)) = self.ready_to_go.pop_first() {
            if buffer_pool.pool.max_sets <= buffer_pool.pool.allocated_descriptors.len() as _ {
                self.full_pools.insert(index, buffer_pool);
            } else {
                self.not_full_pools.insert(index, buffer_pool);
            }
        }
        while let Some((index, buffer_pool)) = self.ready_to_go.pop_first() {
            debug_assert!(
                buffer_pool.pool.max_sets <= buffer_pool.pool.allocated_descriptors.len() as _
            );
            self.full_pools.insert(index, buffer_pool);
        }
    }
    fn prepare_allocate<It: Iterator<Item = (usize, T)>>(&mut self, descriptor_indices: &mut It) {
        while let Some((index, mut buffer_pool)) = self.not_full_pools.pop_first() {
            let pool = &mut buffer_pool.pool;
            let buffer = &mut buffer_pool.update_buffer;
            let remain = pool.max_sets as usize - pool.allocated_descriptors.len();
            buffer.clear();
            for _ in 0..remain {
                if let Some(item) = descriptor_indices.next() {
                    self.descriptor_set_indices.insert(item.0, index);
                    buffer.push(item);
                } else {
                    self.ready_to_go.insert(index, buffer_pool);
                    return;
                }
            }
            self.ready_to_go.insert(index, buffer_pool);
        }
    }
    pub fn free<It: Iterator<Item = usize>>(&mut self, descriptor_indices: It) {
        // collect all the involved pools
        for index in descriptor_indices {
            if let Some(pool_index) = self.descriptor_set_indices.get(&index) {
                if let Some(mut pool) = self.not_full_pools.remove(&pool_index) {
                    pool.free_buffer.push(index);
                    self.ready_to_go.insert(*pool_index, pool);
                    continue;
                }
                if let Some(mut pool) = self.full_pools.remove(&pool_index) {
                    pool.free_buffer.push(index);
                    self.ready_to_go.insert(*pool_index, pool);
                    continue;
                }
                if let Some(pool) = self.ready_to_go.get_mut(&pool_index) {
                    pool.free_buffer.push(index);
                    continue;
                }
            }
        }
        self.ready_to_go
            .par_iter_mut()
            .for_each(|(_, pool)| pool.pool.free(pool.free_buffer.as_slice()).unwrap());
        // now all the pool is not full.
        while let Some((index, pool)) = self.ready_to_go.pop_first() {
            self.not_full_pools.insert(index, pool);
        }
    }
    pub fn get_descriptor_set(&self, descriptor_index: &usize) -> Option<&DescriptorSet> {
        if let Some(pool_index) = self.descriptor_set_indices.get(descriptor_index) {
            if let Some(pool) = self.not_full_pools.get(&pool_index) {
                return pool.pool.get_descriptor_set(descriptor_index);
            }
            if let Some(pool) = self.full_pools.get(&pool_index) {
                return pool.pool.get_descriptor_set(descriptor_index);
            }
        }
        None
    }
}
