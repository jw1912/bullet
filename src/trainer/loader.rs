use std::{slice, fs::File, io::Read, mem::MaybeUninit, path::PathBuf};

use bulletformat::ChessBoard;

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

    pub fn count_positions(&self) -> u64 {
        let data_size = size_of::<ChessBoard>() as u64;

        let mut file_size = 0;

        for file in self.file_paths.iter() {
            let this_size = std::fs::metadata(file).unwrap().len();

            if this_size % data_size != 0 {
                panic!("File [{file}] does not have a multiple of {data_size} size!");
            }

            file_size += this_size;
        }

        file_size / data_size
    }

    pub fn map_batches<F: FnMut(&[ChessBoard]) -> bool>(&self, batch_size: usize, mut f: F) {
        let buffer_size_mb = 256;
        let buffer_size = buffer_size_mb * 1024 * 1024;
        let data_size = size_of::<ChessBoard>();
        let batches_per_load = buffer_size / data_size / batch_size;
        let cap = batch_size * batches_per_load;

        let mut buf = unsafe { zeroed_boxed_slice(cap) };

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
                            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), cap * data_size) },
                        )
                        .unwrap_or(0);

                    if count == 0 {
                        break;
                    }

                    assert_eq!(count % data_size, 0);
                    let len = count / data_size;

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

unsafe fn zeroed_boxed_slice(cap: usize) -> Box<[ChessBoard]> {
    let mut buf = Box::<[ChessBoard]>::new_uninit_slice(cap);

    // safe as `T` can be any bit pattern, including 0s
    let zeroed = unsafe {
        let mut t: MaybeUninit<ChessBoard> = MaybeUninit::uninit();
        let tslice: &mut [MaybeUninit<u8>] = slice::from_raw_parts_mut(t.as_mut_ptr().cast(), size_of::<ChessBoard>());

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
