use std::{
    fmt::Debug,
    fs::File,
    io::{BufRead, BufReader},
    str::FromStr,
};

use bullet_trainer::reader::DataReader;

#[derive(Clone)]
pub struct InMemoryTextLoader {
    file_path: String,
}

impl InMemoryTextLoader {
    pub fn new(file_path: &str) -> Self {
        Self { file_path: file_path.to_string() }
    }
}

impl<T: FromStr> DataReader<T> for InMemoryTextLoader
where
    <T as FromStr>::Err: Debug,
{
    fn read_chunks<F: FnMut(&[T]) -> bool>(&self, _: usize, mut f: F) {
        let file = File::open(&self.file_path).unwrap();
        let reader = BufReader::new(file);
        let data = reader.lines().map(|ln| ln.unwrap().parse::<T>().unwrap()).collect::<Vec<_>>();

        'dataloading: loop {
            if f(&data) {
                break 'dataloading;
            }
        }
    }
}
