use bullet_trainer::run::logger::ansi;

#[derive(Clone, Copy)]
pub struct TestDataset<'a> {
    /// Path to test dataset.
    pub path: &'a str,
    /// Frequency of validation loss (run validation every `freq` batches).
    pub freq: usize,
    /// number of validation batches per validation-event
    pub batches: usize,
}

impl<'a> TestDataset<'a> {
    pub fn at(path: &'a str) -> TestDataset<'a> {
        Self { path, freq: 32, batches: 1 }
    }

    pub fn freq(mut self, freq: usize) -> Self {
        assert!(freq > 0, "Validation frequency must be positive!");
        self.freq = freq;
        self
    }

    pub fn batches(mut self, batches: usize) -> Self {
        assert!(batches > 0, "Validation batch count must be positive!");
        self.batches = batches;
        self
    }
}

pub struct LocalSettings<'a> {
    /// Number of threads to make available for training, in addition
    /// to the main trainer thread (used only for loading data if training
    /// with GPU).
    pub threads: usize,
    /// Path to a test dataset, will calculate vaidation loss over this dataset.
    pub test_set: Option<TestDataset<'a>>,
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

        if let Some(test) = self.test_set {
            println!("Validation Data        : {}", ansi(test.path, "32;1"));
            println!("   Frequency           : {}", ansi(test.freq, 31));
            println!("   Batches             : {}", ansi(test.batches, 31));
        }
    }
}
