pub mod camera;
pub mod mesh_renderer;
pub mod ui;

pub struct ParallelGroup<T> {
    groups: Vec<Vec<T>>,
    cursor: usize,
}

impl<T> ParallelGroup<T> {
    pub fn new() -> Self {
        let groups = (0..rayon::current_num_threads())
            .into_iter()
            .map(|_| Vec::new())
            .collect();
        Self { groups, cursor: 0 }
    }
    pub fn push(&mut self, t: T) {
        self.groups[self.cursor].push(t);
        self.cursor = (self.cursor + 1) % self.groups.len();
    }
    pub fn clear(&mut self) {
        for ts in &mut self.groups {
            ts.clear();
        }
    }
    pub fn get_group_by_thread(&self, thread_index: usize) -> Option<&[T]> {
        Some(self.groups.get(thread_index)?.as_slice())
    }
}
