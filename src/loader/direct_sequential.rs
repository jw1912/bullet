use std::{fs::File, io::{BufRead, BufReader}, marker::PhantomData};

use crate::{util, loader::{BulletFormat, DataLoader}};

#[derive(Clone)]
pub struct DirectSequentialDataLoader<T> {
    file_paths: Vec<String>,
    phantom: PhantomData<T>,  
}

impl<T> DirectSequentialDataLoader<T> {
    pub fn new(file_paths: &[&str]) -> Self {
        Self {
            file_paths: file_paths.iter().map(|path| path.to_string()).collect::<Vec<_>>(),
            phantom: PhantomData,
        }
    }
}

impl<T: BulletFormat + 'static> DataLoader<T> for DirectSequentialDataLoader<T> {
    fn data_file_paths(&self) -> &[String] {
        &self.file_paths
    }

    fn map_batches<F: FnMut(&[T]) -> bool>(&self, batch_size: usize, mut f: F) {
        let buffer_size_mb = 256;
        let buffer_size = buffer_size_mb * 1024 * 1024;
        let data_size: usize = std::mem::size_of::<T>();
        let batches_per_load = buffer_size / data_size / batch_size;
        let cap = data_size * batch_size * batches_per_load;

        'dataloading: loop {
            let mut loader_files = vec![];
            for file in self.file_paths.iter() {
                loader_files.push(File::open(file).unwrap_or_else(|_| panic!("Invalid File Path: {file}")));
            }

            for loader_file in loader_files.iter() {
                let mut file = BufReader::with_capacity(cap, loader_file);
                while let Ok(buf) = file.fill_buf() {
                    if buf.is_empty() {
                        break;
                    }

                    let data: &[T] = util::to_slice_with_lifetime(buf);

                    for batch in data.chunks(batch_size) {
                        let should_break = f(batch);

                        if should_break {
                            break 'dataloading;
                        }
                    }

                    let consumed = buf.len();
                    file.consume(consumed);
                }
            }
        }
    }
}
