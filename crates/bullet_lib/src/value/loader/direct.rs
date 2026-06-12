use std::path::PathBuf;

use bullet_trainer::reader::{DataReader, FixedSizeData, FixedSizeDataReader};

/// ### Safety
/// This indicates that the type can be validly transmuted from
/// *any* sequence of bytes of the same size as the struct.
pub unsafe trait CanBeDirectlySequentiallyLoaded: Copy + Send + Sync + 'static {}

#[derive(Clone)]
pub struct DirectSequentialDataLoader {
    file_paths: Vec<String>,
}

impl DirectSequentialDataLoader {
    pub fn new(file_paths: &[&str]) -> Self {
        let file_paths = file_paths.iter().map(|path| path.to_string()).collect::<Vec<_>>();

        for path in &file_paths {
            let path_buf: PathBuf = path.parse().unwrap();
            assert!(path_buf.exists(), "File not found: {path}");
        }

        Self { file_paths }
    }

    pub fn map_file_sizes<F: FnMut(&str, u64)>(&self, mut f: F) {
        for file in self.file_paths.iter() {
            f(file, std::fs::metadata(file).unwrap().len());
        }
    }
}

#[repr(transparent)]
#[derive(Clone, Copy)]
struct Wrap<T>(T);
unsafe impl<T: CanBeDirectlySequentiallyLoaded> FixedSizeData for Wrap<T> {}

impl<T: CanBeDirectlySequentiallyLoaded> DataReader<T> for DirectSequentialDataLoader {
    fn read_chunks<F: FnMut(&[T]) -> bool>(&self, skip_count: usize, mut f: F) {
        let paths = self.file_paths.iter().map(String::as_str).collect::<Vec<_>>();

        FixedSizeDataReader::<Wrap<T>>::new(&paths).read_chunks(skip_count, |wrapped| {
            let ptr = wrapped.as_ptr().cast();
            f(unsafe { std::slice::from_raw_parts(ptr, wrapped.len()) })
        });
    }
}
