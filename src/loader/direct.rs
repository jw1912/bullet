use std::{fs::File, io::Read, mem::MaybeUninit, path::PathBuf, slice};

use super::DataLoader;

/// ### Safety
/// This indicates that the type can be validly transmuted from
/// *any* sequence of bytes of the same size as the struct.
pub unsafe trait CanBeDirectlySequentiallyLoaded: Copy + 'static {}

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
}

impl<T: CanBeDirectlySequentiallyLoaded> DataLoader<T> for DirectSequentialDataLoader {
    fn data_file_paths(&self) -> &[String] {
        &self.file_paths
    }

    fn count_positions(&self) -> Option<u64> {
        let data_size = std::mem::size_of::<T>() as u64;

        let mut file_size = 0;

        for file in self.file_paths.iter() {
            let this_size = std::fs::metadata(file).unwrap().len();

            if this_size % data_size != 0 {
                panic!("File [{file}] does not have a multiple of {data_size} size!");
            }

            file_size += this_size;
        }

        Some(file_size / data_size)
    }

    fn map_batches<F: FnMut(&[T]) -> bool>(&self, batch_size: usize, mut f: F) {
        let buffer_size_mb = 256;
        let buffer_size = buffer_size_mb * 1024 * 1024;
        let data_size = size_of::<T>();
        let batches_per_load = buffer_size / data_size / batch_size;
        let cap = batch_size * batches_per_load;

        let mut buf = unsafe { zeroed_boxed_slice::<T>(cap) };

        'dataloading: loop {
            let mut loader_files = vec![];
            for file in self.file_paths.iter() {
                loader_files.push(File::open(file).unwrap());
            }

            for mut loader_file in loader_files {
                loop {
                    let count = loader_file
                        .read(
                            // we can cast the type `T` to an array of bytes
                            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), cap * size_of::<T>()) },
                        )
                        .unwrap_or(0);

                    if count == 0 {
                        break;
                    }

                    assert_eq!(count % size_of::<T>(), 0);
                    let len = count / size_of::<T>();

                    for batch in buf[..len].chunks(batch_size) {
                        let should_break = f(batch);

                        if should_break {
                            break 'dataloading;
                        }
                    }
                }
            }
        }
    }
}

unsafe fn zeroed_boxed_slice<T: CanBeDirectlySequentiallyLoaded>(cap: usize) -> Box<[T]> {
    let mut buf = Box::<[T]>::new_uninit_slice(cap);

    // safe as `T` can be any bit pattern, including 0s
    let zeroed = unsafe {
        let mut t: MaybeUninit<T> = MaybeUninit::uninit();
        let tslice: &mut [MaybeUninit<u8>] = slice::from_raw_parts_mut(t.as_mut_ptr().cast(), size_of::<T>());

        for elem in tslice {
            elem.write(0);
        }

        t.assume_init()
    };

    for elem in buf.iter_mut() {
        elem.write(zeroed);
    }

    unsafe { buf.assume_init() }
}
