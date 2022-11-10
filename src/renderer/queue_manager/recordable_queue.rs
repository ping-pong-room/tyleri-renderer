use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::pin::Pin;
use std::sync::Arc;
use std::thread::ThreadId;

use dashmap::mapref::entry::Entry;
use dashmap::mapref::one::{Ref, RefMut};
use dashmap::{DashMap, DashSet};
use derive_more::{Deref, DerefMut};
use parking_lot::{Condvar, Mutex};
use rayon::iter::ParallelIterator;
use rayon::iter::{IntoParallelIterator, ParallelDrainRange};
use rustc_hash::{FxHashMap, FxHasher};
use yarvk::command::command_buffer::Level::{PRIMARY, SECONDARY};
use yarvk::command::command_buffer::RenderPassScope::{INSIDE, OUTSIDE};
use yarvk::command::command_buffer::State::{EXECUTABLE, INITIAL, INVALID, RECORDING};
use yarvk::command::command_buffer::{
    CommandBuffer, CommandBufferInheritanceInfo, TransientCommandBuffer,
};
use yarvk::device::Device;
use yarvk::fence::{Fence, UnsignaledFence};
use yarvk::physical_device::queue_family_properties::QueueFamilyProperties;
use yarvk::queue::submit_info::{SubmitInfo, Submittable};
use yarvk::queue::Queue;
use yarvk::{Handle, Rect2D, SubpassContents, Viewport};

#[derive(Deref, DerefMut)]
pub struct ThreadLocalSecondaryBuffer {
    dirty: bool,
    #[deref]
    #[deref_mut]
    buffer: CommandBuffer<{ SECONDARY }, { RECORDING }, { OUTSIDE }>,
}

pub struct ThreadLocalSecondaryBufferMap {
    queue_family: QueueFamilyProperties,
    device: Arc<Device>,
    command_buffer_inheritance_info: Arc<CommandBufferInheritanceInfo>,
    secondary_buffers: DashMap<ThreadId, ThreadLocalSecondaryBuffer, BuildHasherDefault<FxHasher>>,
    secondary_buffer_handles: DashMap<u64, ThreadId, BuildHasherDefault<FxHasher>>,
}

impl ThreadLocalSecondaryBufferMap {
    fn new(queue: &Queue) -> Self {
        Self {
            queue_family: queue.queue_family_property.clone(),
            device: queue.device.clone(),
            command_buffer_inheritance_info: CommandBufferInheritanceInfo::builder().build(),
            secondary_buffers: Default::default(),
            secondary_buffer_handles: Default::default(),
        }
    }
    fn get_thread_local_secondary_buffer(
        &self,
    ) -> Result<
        RefMut<ThreadId, ThreadLocalSecondaryBuffer, BuildHasherDefault<FxHasher>>,
        yarvk::Result,
    > {
        let thread_id = std::thread::current().id();
        Ok(match self.secondary_buffers.entry(thread_id) {
            Entry::Occupied(mut entry) => {
                let mut ref_mut = entry.into_ref();
                ref_mut.dirty = true;
                ref_mut
            }
            Entry::Vacant(entry) => {
                let secondary_buffer = TransientCommandBuffer::<{ SECONDARY }>::new(
                    &self.device,
                    self.queue_family.clone(),
                )?;
                let secondary_buffer =
                    secondary_buffer.begin(self.command_buffer_inheritance_info.clone())?;
                let secondary_buffer = ThreadLocalSecondaryBuffer {
                    dirty: true,
                    buffer: secondary_buffer,
                };
                let mut ref_mut = entry.insert(secondary_buffer);
                self.secondary_buffer_handles
                    .insert(ref_mut.buffer.handle(), thread_id);
                ref_mut
            }
        })
    }
    fn collect_dirty(&self) -> Vec<CommandBuffer<{ SECONDARY }, { EXECUTABLE }, { OUTSIDE }>> {
        self.secondary_buffers
            .iter()
            .filter(|cb| cb.dirty)
            .map(|cb| *cb.key())
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|thread_id| {
                let (_, cb) = self.secondary_buffers.remove(&thread_id).unwrap();
                cb.buffer.end().unwrap()
            })
            .collect()
    }
    fn return_buffers(
        self: &Arc<Self>,
        buffers: &mut Vec<CommandBuffer<{ SECONDARY }, { INVALID }, { OUTSIDE }>>,
    ) {
        buffers.par_drain(..).for_each(|command_buffer| {
            let command_buffer_inheritance_info = self.command_buffer_inheritance_info.clone();
            let this = self.clone();
            let command_buffer = command_buffer
                .reset()
                .unwrap()
                .begin(command_buffer_inheritance_info)
                .unwrap();
            let thread_id = this
                .secondary_buffer_handles
                .get(&command_buffer.handle())
                .expect("inner error: command handle not exists any more");
            this.secondary_buffers.insert(
                *thread_id,
                ThreadLocalSecondaryBuffer {
                    dirty: false,
                    buffer: command_buffer,
                },
            );
        });
    }
}

#[derive(Deref, DerefMut)]
pub struct RecordableQueue {
    #[deref]
    #[deref_mut]
    queue: Queue,
    command_buffer: Option<CommandBuffer<{ PRIMARY }, { INITIAL }, { OUTSIDE }>>,
    thread_local_secondary_buffer_map: Arc<ThreadLocalSecondaryBufferMap>,
    fence: Option<UnsignaledFence>,
}

impl RecordableQueue {
    pub fn new(queue: Queue) -> Result<Self, yarvk::Result> {
        let device = queue.device.clone();
        let command_buffer = TransientCommandBuffer::<{ PRIMARY }>::new(
            &device,
            queue.queue_family_property.clone(),
        )?;
        let fence = Fence::new(device.clone())?;
        let thread_local_secondary_buffer_map =
            Arc::new(ThreadLocalSecondaryBufferMap::new(&queue));
        Ok(Self {
            queue,
            command_buffer: Some(command_buffer),
            thread_local_secondary_buffer_map,
            fence: Some(fence),
        })
    }
    pub fn get_thread_local_secondary_buffer(
        &self,
    ) -> Result<
        RefMut<ThreadId, ThreadLocalSecondaryBuffer, BuildHasherDefault<FxHasher>>,
        yarvk::Result,
    > {
        self.thread_local_secondary_buffer_map
            .get_thread_local_secondary_buffer()
    }
    // Used command buffer to some simple recording job, wait until job finished executing.
    pub fn simple_record(
        &mut self,
        f: impl FnOnce(
            &mut CommandBuffer<{ PRIMARY }, { RECORDING }, { OUTSIDE }>,
        ) -> Result<(), yarvk::Result>,
    ) -> Result<(), yarvk::Result> {
        let command_buffer = self.command_buffer.take().unwrap();
        let command_handle = command_buffer.handle();
        let fence = self.fence.take().unwrap();
        let command_buffer = command_buffer.record(f).unwrap();
        let submit_info = SubmitInfo::builder()
            .add_one_time_submit_command_buffer(command_buffer)
            .build();
        let fence = Submittable::new()
            .add_submit_info(submit_info)
            .submit(&mut self.queue, fence)?;
        let (signaled_fence, mut submit_result) = fence.wait()?;
        let fence = signaled_fence.reset()?;
        let command_buffer = submit_result
            .take_invalid_primary_buffer(&command_handle)
            .unwrap();
        let command_buffer = command_buffer.reset()?;
        self.command_buffer = Some(command_buffer);
        self.fence = Some(fence);
        Ok(())
    }

    pub fn simple_secondary_record(&mut self) -> Result<(), yarvk::Result> {
        let command_buffer = self.command_buffer.take().unwrap();
        let command_handle = command_buffer.handle();
        let fence = self.fence.take().unwrap();
        let command_buffer = command_buffer
            .record(|primary_command_buffer| {
                primary_command_buffer.cmd_execute_commands(
                    &mut self.thread_local_secondary_buffer_map.collect_dirty(),
                );
                Ok(())
            })
            .unwrap();
        let submit_info = SubmitInfo::builder()
            .add_one_time_submit_command_buffer(command_buffer)
            .build();
        let fence = Submittable::new()
            .add_submit_info(submit_info)
            .submit(&mut self.queue, fence)?;
        let (signaled_fence, mut submit_result) = fence.wait()?;
        let fence = signaled_fence.reset()?;
        let mut primary_command_buffer = submit_result
            .take_invalid_primary_buffer(&command_handle)
            .unwrap();
        let secondary_buffers = primary_command_buffer.secondary_buffers();
        self.thread_local_secondary_buffer_map
            .return_buffers(secondary_buffers);
        let primary_command_buffer = primary_command_buffer.reset()?;
        self.command_buffer = Some(primary_command_buffer);
        self.fence = Some(fence);
        Ok(())
    }
}
