use std::sync::Arc;
use yarvk::Buffer;

pub struct Mesh {
    vertices: Arc<dyn Buffer>,
    indices: Arc<dyn Buffer>,
}
