use derive_more::{Deref, DerefMut};
use yarvk::command::command_buffer::CommandBuffer;
use yarvk::command::command_buffer::Level::PRIMARY;
use yarvk::command::command_buffer::RenderPassScope::OUTSIDE;
use yarvk::command::command_buffer::State::{INITIAL, RECORDING};
use yarvk::fence::UnsignaledFence;
use yarvk::queue::submit_info::{SubmitInfo, Submittable};
use yarvk::queue::Queue;
use yarvk::Handle;

#[derive(Deref, DerefMut)]
pub struct RecordableQueue {
    #[deref]
    #[deref_mut]
    pub(crate) queue: Queue,
    pub(crate) command_buffer: Option<CommandBuffer<{ PRIMARY }, { INITIAL }, { OUTSIDE }, true>>,
    pub(crate) fence: Option<UnsignaledFence>,
}

impl RecordableQueue {
    // Used command buffer to some simple recording job, wait until job finished executing.
    pub fn simple_record(
        &mut self,
        f: impl FnOnce(
            &mut CommandBuffer<{ PRIMARY }, { RECORDING }, { OUTSIDE }, true>,
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
}
