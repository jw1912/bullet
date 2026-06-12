use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    marker::PhantomData,
    mem::MaybeUninit,
    path::PathBuf,
};

use crate::reader::DataReader;

/// ## Safety
/// Type must be `repr(C)`, have no padding or uninitialised
/// bytes, valid as any bit pattern and of fixed size.
pub unsafe trait FixedSizeData: Copy + Send + Sync + 'static {}

#[derive(Clone)]
pub struct FixedSizeDataReader<T: FixedSizeData> {
    file_paths: Vec<String>,
    phantom: PhantomData<T>,
}

impl<T: FixedSizeData> FixedSizeDataReader<T> {
    pub fn new(file_paths: &[&str]) -> Self {
        let file_paths = file_paths.iter().map(|path| path.to_string()).collect::<Vec<_>>();

        for path in &file_paths {
            let path_buf = path.parse::<PathBuf>().unwrap();
            assert!(path_buf.exists(), "File not found: {path}");
        }

        Self { file_paths, phantom: PhantomData }
    }

    pub fn map_file_sizes<F: FnMut(&str, u64)>(&self, mut f: F) {
        for file in self.file_paths.iter() {
            f(file, std::fs::metadata(file).unwrap().len());
        }
    }
}

impl<T: FixedSizeData> DataReader<T> for FixedSizeDataReader<T> {
    fn read_chunks<F: FnMut(&[T]) -> bool>(&self, mut start_position: usize, mut f: F) {
        let data_size = std::mem::size_of::<T>() as u64;
        let cap = (128usize * 1024 * 1024).div_ceil(data_size as usize);

        let mut positions_per_epoch = 0;
        self.map_file_sizes(|_, this_size| positions_per_epoch += this_size / data_size);
        start_position %= positions_per_epoch as usize;

        let mut start_file_idx = 0;
        let mut net_positions = 0;
        for file in self.file_paths.iter() {
            let this_size = std::fs::metadata(file).unwrap().len();
            let this_positions = this_size / data_size;

            net_positions += this_positions;

            if start_position < net_positions as usize {
                net_positions -= this_positions;
                break;
            } else {
                start_file_idx += 1;
            }
        }

        let mut file_paths = self.file_paths.clone();
        file_paths.rotate_left(start_file_idx);

        let mut to_skip = start_position - net_positions as usize;

        let mut buf = unsafe { zeroed_boxed_slice::<T>(cap) };

        'dataloading: loop {
            let mut loader_files = vec![];
            for file in file_paths.iter() {
                loader_files.push(File::open(file).unwrap());
            }

            for (mut loader_file, file_path) in loader_files.into_iter().zip(file_paths.iter()) {
                if to_skip > 0 {
                    println!("Skipping to {to_skip}th entry in file [{file_path}]");
                    loader_file.seek(SeekFrom::Current((to_skip * data_size as usize) as i64)).unwrap();
                    to_skip = 0;
                }

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

                    if f(&buf[..len]) {
                        break 'dataloading;
                    }
                }
            }
        }
    }
}

unsafe fn zeroed_boxed_slice<T: FixedSizeData>(cap: usize) -> Box<[T]> {
    let mut buf = Box::<[T]>::new_uninit_slice(cap);

    // safe as `T` can be any bit pattern, including 0s
    let zeroed = unsafe { MaybeUninit::zeroed().assume_init() };

    for elem in buf.iter_mut() {
        elem.write(zeroed);
    }

    unsafe { buf.assume_init() }
}
