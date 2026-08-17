use std::{
    fs::File,
    io::{self, Read, Write},
    sync::Arc,
};

use bullet_compiler::tensor::TValue;
use bullet_gpu::{buffer::Buffer, runtime::Gpu};

use crate::model::utils::{read_from_byte_buffer, write_to_byte_buffer};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    Before,
    After,
}

/// Write a set of labelled weights from a `BTreeMap` into a file.
pub fn write_weights_to_file<G: Gpu>(map: &[(impl AsRef<str>, &Arc<Buffer<G>>)], path: &str) -> Result<(), G::Error> {
    write_mapped_weights_to_file(map, path, |buf| {
        let this_buf = (*buf).clone().to_host().unwrap();
        let TValue::F32(this_buf) = this_buf else { panic!() };
        this_buf
    })
    .map_err(|e| G::Error::from(e.to_string()))
}

/// Write a set of labelled weights from a `BTreeMap` into a file.
pub fn write_mapped_weights_to_file<T>(
    map: &[(impl AsRef<str>, &T)],
    path: &str,
    f: impl Fn(&T) -> Vec<f32>,
) -> io::Result<()> {
    let mut buf = Vec::new();

    for (id, weights) in map {
        let byte_buf = write_to_byte_buffer(&f(weights), id.as_ref()).unwrap();
        buf.extend_from_slice(&byte_buf);
    }

    let mut file = File::create(path).unwrap();
    file.write_all(&buf).unwrap();

    Ok(())
}

/// Loads a set of labelled weights from a file into a `BTreeMap`.
pub fn load_weights_from_file(path: &str) -> Vec<(String, Vec<f32>)> {
    let mut buf = Vec::new();
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buf).unwrap();

    let mut offset = 0;

    let mut res = Vec::new();

    while offset < buf.len() {
        let (buffer, id, bytes_read) = read_from_byte_buffer(&buf[offset..]);
        res.push((id, buffer));
        offset += bytes_read;
    }

    res
}
