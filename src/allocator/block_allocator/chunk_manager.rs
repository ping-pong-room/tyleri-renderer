use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BTreeMap;
use yarvk::device_memory::DeviceMemory;

pub trait DeviceMemoryTrait {
    fn size(&self) -> u64;
}

impl DeviceMemoryTrait for DeviceMemory {
    fn size(&self) -> u64 {
        self.size
    }
}

struct Block {
    len: u64,
    used: bool,
    pre: Option<u64>,
    next: Option<u64>,
}

struct AllocatedChuck<DeviceMemory: DeviceMemoryTrait> {
    blocks: FxHashMap<u64 /*offset*/, Block>,
    device_memory: DeviceMemory,
}

impl<DeviceMemory: DeviceMemoryTrait> AllocatedChuck<DeviceMemory> {
    pub fn new(device_memory: DeviceMemory) -> Self {
        assert_ne!(device_memory.size(), 0);
        let mut blocks = FxHashMap::default();
        blocks.insert(
            0,
            Block {
                len: device_memory.size(),
                used: false,
                pre: None,
                next: None,
            },
        );
        Self {
            blocks,
            device_memory,
        }
    }

    pub fn new_and_allocated(device_memory: DeviceMemory, allocated_len: u64) -> Self {
        let len = device_memory.size();
        assert_ne!(len, 0);
        assert_ne!(allocated_len, 0);
        let perfect_match = len == allocated_len;
        let mut blocks = FxHashMap::default();
        blocks.insert(
            0,
            Block {
                len: allocated_len,
                used: true,
                pre: None,
                next: if perfect_match {
                    None
                } else {
                    Some(allocated_len)
                },
            },
        );
        if !perfect_match {
            blocks.insert(
                allocated_len,
                Block {
                    len: len - allocated_len,
                    used: false,
                    pre: Some(0),
                    next: None,
                },
            );
        }
        Self {
            blocks,
            device_memory,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub struct BlockIndex {
    pub offset: u64,
    pub chunk_index: usize,
}

impl BlockIndex {
    pub fn new(offset: u64, chunk_index: usize) -> Self {
        Self {
            offset,
            chunk_index,
        }
    }
}

pub struct ChunkManager<DeviceMemory: DeviceMemoryTrait> {
    // a map to save all unused blocks, speed up allocation lookup
    unused_blocks: BTreeMap<u64 /*len*/, FxHashSet<BlockIndex>>,
    // all chunks
    chunks: FxHashMap<usize, AllocatedChuck<DeviceMemory>>,
}

impl<DeviceMemory: DeviceMemoryTrait> Default for ChunkManager<DeviceMemory> {
    fn default() -> Self {
        Self {
            unused_blocks: Default::default(),
            chunks: FxHashMap::default(),
        }
    }
}

impl<DeviceMemory: DeviceMemoryTrait> ChunkManager<DeviceMemory> {
    pub fn add_chunk(&mut self, device_memory: DeviceMemory) -> usize {
        let len = device_memory.size();
        let chunks_index = self.chunks.len();
        self.chunks
            .insert(chunks_index, AllocatedChuck::new(device_memory));
        Self::insert_unused(&mut self.unused_blocks, 0, len, chunks_index);
        chunks_index
    }
    pub fn get_device_memory(&mut self, block_index: &BlockIndex) -> Option<&mut DeviceMemory> {
        Some(&mut self.chunks.get_mut(&block_index.chunk_index)?.device_memory)
    }
    pub fn add_chunk_and_allocate(
        &mut self,
        device_memory: DeviceMemory,
        allocate_len: u64,
    ) -> BlockIndex {
        let len = device_memory.size();
        let chunk_index = self.chunks.len();
        self.chunks.insert(
            chunk_index,
            AllocatedChuck::new_and_allocated(device_memory, allocate_len),
        );
        Self::insert_unused(
            &mut self.unused_blocks,
            allocate_len,
            len - allocate_len,
            chunk_index,
        );
        BlockIndex {
            offset: 0,
            chunk_index,
        }
    }
    pub fn insert_unused(
        blocks: &mut BTreeMap<u64 /*len*/, FxHashSet<BlockIndex>>,
        offset: u64,
        len: u64,
        chunk_index: usize,
    ) {
        match blocks.get_mut(&len) {
            None => {
                let mut set = FxHashSet::default();
                set.insert(BlockIndex::new(offset, chunk_index));
                blocks.insert(len, set);
            }
            Some(set) => {
                set.insert(BlockIndex::new(offset, chunk_index));
            }
        }
    }
    pub fn remove_unused(
        blocks: &mut BTreeMap<u64 /*len*/, FxHashSet<BlockIndex>>,
        offset: u64,
        len: u64,
        chunk_index: usize,
    ) -> Option<u64 /*offset*/> {
        let block_index = BlockIndex::new(offset, chunk_index);
        return match blocks.get_mut(&len) {
            None => None,
            Some(set) => {
                set.remove(&block_index);
                if set.is_empty() {
                    blocks.remove(&len);
                    return Some(offset);
                } else {
                    None
                }
            }
        };
    }
    pub fn allocate(&mut self, len: u64, alignment: u64) -> Option<BlockIndex> {
        if len == 0 {
            return None;
        }
        let result = None;
        // find available unused blocks
        for (unused_len, set) in self.unused_blocks.range_mut(len..) {
            let unused_len = *unused_len;
            for block_index in set.iter() {
                let offset = block_index.offset;
                let chunk_index = block_index.chunk_index;
                let chunk = unsafe { self.chunks.get_mut(&chunk_index).unwrap_unchecked() };
                let block = unsafe { chunk.blocks.get_mut(&offset).unwrap_unchecked() };
                let offset_mod_alignment = offset % alignment;
                if offset_mod_alignment == 0 {
                    if block.len < len {
                        // too small
                        continue;
                    } else {
                        Self::remove_unused(
                            &mut self.unused_blocks,
                            offset,
                            unused_len,
                            chunk_index,
                        );
                        block.used = true;
                        if len != block.len {
                            let new_offset = offset + len;
                            let new_len = block.len - len;
                            let new_block = Block {
                                len: new_len,
                                used: false,
                                pre: Some(offset),
                                next: block.next,
                            };
                            block.next = Some(new_offset);
                            block.len = len;
                            chunk.blocks.insert(new_offset, new_block);
                            Self::insert_unused(
                                &mut self.unused_blocks,
                                new_offset,
                                new_len,
                                chunk_index,
                            );
                        }
                        return Some(BlockIndex::new(offset, chunk_index));
                    }
                } else {
                    let wasted_len = alignment - offset_mod_alignment;
                    let available_len = block.len - wasted_len;
                    if available_len < len {
                        // not enough for alignment
                        continue;
                    } else {
                        Self::remove_unused(
                            &mut self.unused_blocks,
                            offset,
                            unused_len,
                            chunk_index,
                        );
                        let block_len = block.len;
                        block.len = wasted_len;

                        let new_offset = offset + wasted_len;
                        let new_pre = Some(offset);
                        let new_next = block.next;
                        block.next = Some(new_offset);
                        let mut new_block = Block {
                            len,
                            used: true,
                            pre: new_pre,
                            next: new_next,
                        };
                        Self::insert_unused(
                            &mut self.unused_blocks,
                            offset,
                            block.len,
                            chunk_index,
                        );

                        if available_len > len {
                            let tail_block_offset = offset + wasted_len + len;
                            let tail_block_len = block_len - wasted_len - len;
                            chunk.blocks.insert(
                                tail_block_offset,
                                Block {
                                    len: tail_block_len,
                                    used: false,
                                    pre: Some(new_offset),
                                    next: new_next,
                                },
                            );
                            new_block.next = Some(tail_block_offset);
                            Self::insert_unused(
                                &mut self.unused_blocks,
                                tail_block_offset,
                                tail_block_len,
                                chunk_index,
                            );
                        }

                        chunk.blocks.insert(new_offset, new_block);
                        return Some(BlockIndex::new(new_offset, chunk_index));
                    }
                }
            }
            continue;
        }
        return result;
    }
    pub unsafe fn free_unchecked(&mut self, block_index: &BlockIndex) {
        let chunk_index = block_index.chunk_index;
        let offset = block_index.offset;
        let chunk = self.chunks.get_mut(&chunk_index).unwrap_unchecked();
        let mut block = chunk.blocks.remove(&offset).unwrap_unchecked();
        self.merge_with_next(&mut block, chunk_index);
        self.merge_with_pre(offset, block, chunk_index);
    }
    unsafe fn merge_with_pre(&mut self, offset: u64, mut block: Block, chunk_index: usize) {
        let chunk = self.chunks.get_mut(&chunk_index).unwrap_unchecked();
        if let Some(pre_offset) = block.pre {
            let pre_block = chunk.blocks.get_mut(&pre_offset).unwrap_unchecked();
            // if pre is unused.
            if pre_block.used == false {
                Self::remove_unused(
                    &mut self.unused_blocks,
                    pre_offset,
                    pre_block.len,
                    chunk_index,
                );
                pre_block.len += block.len;
                pre_block.next = block.next;
                Self::insert_unused(
                    &mut self.unused_blocks,
                    pre_offset,
                    pre_block.len,
                    chunk_index,
                );
                return;
            }
        }
        Self::insert_unused(&mut self.unused_blocks, offset, block.len, chunk_index);
        block.used = false;
        chunk.blocks.insert(offset, block);
    }
    unsafe fn merge_with_next(&mut self, block: &mut Block, chunk_index: usize) {
        if let Some(i) = block.next {
            let chunk = self.chunks.get_mut(&chunk_index).unwrap_unchecked();
            let next_block = chunk.blocks.get(&i).unwrap_unchecked();
            if next_block.used == false {
                block.len += next_block.len;
                block.next = next_block.next;
                Self::remove_unused(&mut self.unused_blocks, i, next_block.len, chunk_index);
                chunk.blocks.remove(&i);
            }
        }
    }
    pub fn free_unused_chunk(&mut self) {
        self.unused_blocks.retain(|_, infos| {
            infos.retain(|info| {
                let index = info.chunk_index;
                return if let Some(chunk) = self.chunks.get(&index) {
                    // only one block in chunk, which means the whole chunk is unused
                    if chunk.blocks.len() == 1 {
                        self.chunks.remove(&index);
                        false
                    } else {
                        true
                    }
                } else {
                    false
                };
            });
            if infos.is_empty() {
                return false;
            }
            true
        });
    }
}

#[cfg(test)]
#[derive(Default)]
struct TestDeviceMemory(u64);

#[cfg(test)]
impl DeviceMemoryTrait for TestDeviceMemory {
    fn size(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
fn iter_blocks(blocks: &FxHashMap<u64 /*offset*/, Block>) -> Vec<u64> {
    let mut p_next = blocks.get(&0).unwrap().next;
    let block = blocks.get(&0).unwrap();
    let mut vec = Vec::new();
    let mut last_used = block.used;
    vec.push(0);
    while !p_next.is_none() {
        let next_offset = p_next.unwrap();
        let next_block = blocks.get(&next_offset).unwrap();
        p_next = next_block.next;
        if last_used == false && !next_block.used {
            panic!("two connected unused block")
        }
        last_used = next_block.used;
        vec.push(next_offset)
    }
    vec
}

#[test]
fn test() {
    let mut allocator = ChunkManager::default();
    // allocator.add_chunk(128);
    // allocator.allocate(1, 1);
    allocator.add_chunk_and_allocate(TestDeviceMemory(128), 1);
    let chunk = allocator.chunks.get(&0).unwrap();
    assert_eq!(chunk.blocks.get(&0).unwrap().len, 1);
    assert_eq!(chunk.blocks.get(&1).unwrap().len, 127);
    assert!(allocator
        .unused_blocks
        .get(&127)
        .unwrap()
        .contains(&BlockIndex::new(1, 0)));
    assert_eq!(chunk.blocks.get(&0).unwrap().pre, None);
    assert_eq!(chunk.blocks.get(&0).unwrap().next, Some(1));
    assert_eq!(chunk.blocks.get(&1).unwrap().pre, Some(0));
    assert_eq!(chunk.blocks.get(&1).unwrap().next, None);

    allocator.allocate(2, 2);
    let chunk = allocator.chunks.get(&0).unwrap();
    assert_eq!(chunk.blocks.get(&0).unwrap().len, 1);
    assert_eq!(chunk.blocks.get(&1).unwrap().len, 1);
    assert_eq!(chunk.blocks.get(&2).unwrap().len, 2);
    assert_eq!(chunk.blocks.get(&4).unwrap().len, 124);
    assert!(allocator
        .unused_blocks
        .get(&1)
        .unwrap()
        .contains(&BlockIndex::new(1, 0)));
    assert!(allocator
        .unused_blocks
        .get(&124)
        .unwrap()
        .contains(&BlockIndex::new(4, 0)));
    assert_eq!(chunk.blocks.get(&1).unwrap().pre, Some(0));
    assert_eq!(chunk.blocks.get(&1).unwrap().next, Some(2));
    assert_eq!(chunk.blocks.get(&2).unwrap().pre, Some(1));
    assert_eq!(chunk.blocks.get(&2).unwrap().next, Some(4));
    assert_eq!(chunk.blocks.get(&4).unwrap().pre, Some(2));
    assert_eq!(chunk.blocks.get(&4).unwrap().next, None);

    allocator.allocate(32, 32);
    let chunk = allocator.chunks.get(&0).unwrap();
    assert_eq!(iter_blocks(&chunk.blocks).as_slice(), [0, 1, 2, 4, 32, 64]);

    allocator.allocate(8, 8);
    let chunk = allocator.chunks.get(&0).unwrap();
    assert_eq!(
        iter_blocks(&chunk.blocks).as_slice(),
        [0, 1, 2, 4, 8, 16, 32, 64]
    );
    assert_eq!(allocator.unused_blocks.len(), 4);

    unsafe { allocator.free_unchecked(&BlockIndex::new(8, 0)) }
    unsafe { allocator.free_unchecked(&BlockIndex::new(32, 0)) }

    unsafe { allocator.free_unchecked(&BlockIndex::new(2, 0)) };
    let chunk = allocator.chunks.get(&0).unwrap();
    assert_eq!(chunk.blocks.get(&0).unwrap().len, 1);
    assert_eq!(chunk.blocks.get(&1).unwrap().len, 127);
    assert_eq!(chunk.blocks.len(), 2);
    assert!(allocator
        .unused_blocks
        .get(&127)
        .unwrap()
        .contains(&BlockIndex::new(1, 0)));
    assert_eq!(allocator.unused_blocks.len(), 1);

    unsafe { allocator.free_unchecked(&BlockIndex::new(0, 0)) };
    let chunk = allocator.chunks.get(&0).unwrap();
    assert_eq!(chunk.blocks.get(&0).unwrap().len, 128);
    assert!(allocator
        .unused_blocks
        .get(&128)
        .unwrap()
        .contains(&BlockIndex::new(0, 0)));
    assert_eq!(chunk.blocks.get(&0).unwrap().pre, None);
    assert_eq!(chunk.blocks.get(&0).unwrap().next, None);
}

#[test]
fn multiple_candidate_test() {
    let mut allocator = ChunkManager::default();
    allocator.add_chunk(TestDeviceMemory(128));
    for _i in 0..4 {
        allocator.allocate(1, 1);
    }
    let chunk = allocator.chunks.get(&0).unwrap();
    assert_eq!(iter_blocks(&chunk.blocks).as_slice(), [0, 1, 2, 3, 4]);
    allocator.allocate(2, 2);
    allocator.allocate(1, 1);
    let chunk = allocator.chunks.get(&0).unwrap();
    assert_eq!(iter_blocks(&chunk.blocks).as_slice(), [0, 1, 2, 3, 4, 6, 7]);
    unsafe {
        allocator.free_unchecked(&BlockIndex::new(1, 0));
        let chunk = allocator.chunks.get(&0).unwrap();
        assert_eq!(iter_blocks(&chunk.blocks).as_slice(), [0, 1, 2, 3, 4, 6, 7]);
        allocator.free_unchecked(&BlockIndex::new(2, 0));
        let chunk = allocator.chunks.get(&0).unwrap();
        assert_eq!(iter_blocks(&chunk.blocks).as_slice(), [0, 1, 3, 4, 6, 7]);
        allocator.free_unchecked(&BlockIndex::new(4, 0));
    }
    let chunk = allocator.chunks.get(&0).unwrap();
    assert_eq!(iter_blocks(&chunk.blocks).as_slice(), [0, 1, 3, 4, 6, 7]);
    assert_eq!(allocator.unused_blocks.get(&2).unwrap().len(), 2);
    assert_eq!(allocator.allocate(2, 2), Some(BlockIndex::new(4, 0)))
}

#[test]
fn free_unused_test() {
    let mut allocator = ChunkManager::default();
    allocator.add_chunk(TestDeviceMemory(128));
    allocator.add_chunk(TestDeviceMemory(128));
    allocator.free_unused_chunk();
    assert!(allocator.unused_blocks.is_empty());
    assert!(allocator.chunks.is_empty());
}
