use super::logger::ansi;

pub struct LocalSettings<'a> {
    /// Number of threads to make available for training, in addition
    /// to the main trainer thread and data loader thread.
    pub threads: usize,
    /// Directory to write checkpoints to.
    pub output_directory: &'a str,
    /// Number of batches that the dataloader can prepare and put in a queue before
    /// they are processed in training.
    pub batch_queue_size: usize,
}

impl LocalSettings<'_> {
    pub fn display(&self) {
        println!("Threads                : {}", ansi(self.threads, 31));
        println!("Output Path            : {}", ansi(self.output_directory, "32;1"));
    }
}
