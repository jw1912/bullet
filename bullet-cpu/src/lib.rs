pub mod ops;
pub mod util;

#[derive(Clone, Copy, Default)]
pub struct DeviceHandles {
    pub(crate) threads: usize,
}

impl DeviceHandles {
    pub fn set_thread_info(&mut self, threads: usize) {
        self.threads = threads;
    }
}

