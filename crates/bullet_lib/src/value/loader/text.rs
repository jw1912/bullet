use std::{
    fmt::Debug,
    fs::File,
    io::{BufRead, BufReader},
    str::FromStr,
};

use super::DataLoader;

#[derive(Clone)]
pub struct InMemoryTextLoader {
    file_path: String,
}

impl InMemoryTextLoader {
    pub fn new(file_path: &str) -> Self {
        Self { file_path: file_path.to_string() }
    }
}

impl<T: FromStr> DataLoader<T> for InMemoryTextLoader
where
    <T as FromStr>::Err: Debug,
{
    fn data_file_paths(&self) -> &[String] {
        std::slice::from_ref(&self.file_path)
    }

    fn count_positions(&self) -> Option<u64> {
        Some(BufReader::new(File::open(&self.file_path).unwrap()).lines().count() as u64)
    }

    fn map_batches<F: FnMut(&[T]) -> bool>(&self, _: usize, batch_size: usize, mut f: F) {
        let file = File::open(&self.file_path).unwrap();
        let reader = BufReader::new(file);
        let data = reader.lines().map(|ln| ln.unwrap().parse::<T>().unwrap()).collect::<Vec<_>>();

        'dataloading: loop {
            for batch in data.chunks(batch_size) {
                if f(batch) {
                    break 'dataloading;
                }
            }
        }
    }
}
