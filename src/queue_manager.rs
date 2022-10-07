use crate::queue_manager::recordable_queue::RecordableQueue;
use float_ord::FloatOrd;
use std::collections::BTreeMap;
use std::sync::Arc;

use yarvk::command::command_buffer::Level::PRIMARY;

use yarvk::command::command_pool::CommandPool;
use yarvk::device::{Device, DeviceQueueCreateInfo};
use yarvk::extensions::{DeviceExtensionType, PhysicalInstanceExtensionType};
use yarvk::fence::Fence;
use yarvk::physical_device::queue_family_properties::QueueFamilyProperties;
use yarvk::physical_device::PhysicalDevice;

use yarvk::QueueFlags;

pub mod recordable_queue;

const PRESENT_QUEUE_PRIORITY: f32 = 1.0;
const TRANSFER_QUEUE_PRIORITY: f32 = 0.9;

pub struct QueueManager {
    device: Arc<Device>,
    present_queues: (
        QueueFamilyProperties,
        BTreeMap<FloatOrd<f32> /*priority*/, RecordableQueue>,
    ),
    transfer_queues: Option<(QueueFamilyProperties, Vec<RecordableQueue>)>,
}

impl QueueManager {
    pub(crate) fn new(pdevice: Arc<PhysicalDevice>) -> Result<Self, yarvk::Result> {
        let mut present_queue_family = None;
        let mut transfer_queue_family = None;
        let properties = pdevice.get_physical_device_queue_family_properties();
        for queue_family_properties in &properties {
            let queue_flags = queue_family_properties.queue_flags;
            if queue_flags.contains(QueueFlags::TRANSFER)
                && !queue_flags.contains(QueueFlags::GRAPHICS)
            {
                // This is a dedicated transfer queue
                transfer_queue_family = Some(queue_family_properties);
            }
            if queue_flags.contains(QueueFlags::GRAPHICS) {
                present_queue_family = Some(queue_family_properties);
            }
        }

        let surface_ext = pdevice
            .instance
            .get_extension::<{ PhysicalInstanceExtensionType::KhrSurface }>()
            .unwrap();
        let mut device_builder = Device::builder(pdevice.clone())
            .add_extension(&DeviceExtensionType::KhrSwapchain(surface_ext));
        let present_queue_family = present_queue_family.unwrap();
        let mut present_queue_create_info_builder =
            DeviceQueueCreateInfo::builder(present_queue_family.clone());
        present_queue_create_info_builder =
            present_queue_create_info_builder.add_priority(PRESENT_QUEUE_PRIORITY);
        // Add proper transfer queue if exists.
        if let Some(transfer_queue_family) = transfer_queue_family {
            let transfer_queue_create_info =
                DeviceQueueCreateInfo::builder(transfer_queue_family.clone())
                    .add_priority(TRANSFER_QUEUE_PRIORITY)
                    .build();
            device_builder = device_builder.add_queue_info(transfer_queue_create_info);
        } else {
            if present_queue_family.queue_count > 1 {
                present_queue_create_info_builder =
                    present_queue_create_info_builder.add_priority(TRANSFER_QUEUE_PRIORITY);
            }
        }
        let present_queue_create_info = present_queue_create_info_builder.build();
        let (device, mut queues) = device_builder
            .add_queue_info(present_queue_create_info)
            .build()?;
        let present_queues = queues.remove(present_queue_family).unwrap();
        let present_queues = present_queues
            .into_iter()
            .map(|queue| {
                let command_buffer =
                    CommandPool::builder(queue.queue_family_property.clone(), device.clone())
                        .build()
                        .unwrap()
                        .allocate_command_buffer::<{ PRIMARY }>()?;
                let fence = Fence::new(device.clone())?;
                Ok((
                    FloatOrd(queue.priority),
                    RecordableQueue {
                        queue,
                        command_buffer: Some(command_buffer),
                        fence: Some(fence),
                    },
                ))
            })
            .collect::<Result<_, yarvk::Result>>()?;
        let present_queues = (present_queue_family.clone(), present_queues);
        let transfer_queues = if let Some(transfer_queue_family) = transfer_queue_family {
            let transfer_queues = queues.remove(transfer_queue_family).unwrap();
            let transfer_queues = transfer_queues
                .into_iter()
                .map(|queue| {
                    let command_buffer =
                        CommandPool::builder(queue.queue_family_property.clone(), device.clone())
                            .build()
                            .unwrap()
                            .allocate_command_buffer::<{ PRIMARY }>()?;
                    let fence = Fence::new(device.clone())?;
                    Ok(RecordableQueue {
                        queue,
                        command_buffer: Some(command_buffer),
                        fence: Some(fence),
                    })
                })
                .collect::<Result<Vec<_>, yarvk::Result>>()?;
            Some((transfer_queue_family.clone(), transfer_queues))
        } else {
            None
        };
        Ok(Self {
            device,
            present_queues,
            transfer_queues,
        })
    }

    pub fn get_present_queue_family(&self) -> &QueueFamilyProperties {
        &self.present_queues.0
    }

    pub fn take_present_queue_priority_high(&mut self) -> Option<RecordableQueue> {
        match self.present_queues.1.iter().next() {
            None => None,
            Some((priority, _)) => {
                let priority = priority.clone();
                self.present_queues.1.remove(&priority)
            }
        }
    }

    pub fn take_present_queue_priority_low(&mut self) -> Option<RecordableQueue> {
        match self.present_queues.1.iter().next_back() {
            None => None,
            Some((priority, _)) => {
                let priority = priority.clone();
                self.present_queues.1.remove(&priority)
            }
        }
    }

    pub fn take_transfer_queue(&mut self) -> Option<RecordableQueue> {
        if let Some((_, queues)) = &mut self.transfer_queues {
            queues.pop()
        } else {
            self.take_present_queue_priority_low()
        }
    }

    pub fn push_queue(&mut self, queue: RecordableQueue) {
        if queue.queue_family_property == self.present_queues.0 {
            self.present_queues
                .1
                .insert(FloatOrd(queue.priority), queue);
        } else if let Some((queue_properties, queues)) = &mut self.transfer_queues {
            if *queue_properties == queue.queue_family_property {
                queues.push(queue);
            }
        }
    }

    pub fn get_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}
