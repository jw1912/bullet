use crate::{
    device::Device,
    graph::tensor::{read_from_byte_buffer, DenseMatrix},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    Before,
    After,
}

/// Write a set of labelled weights from a `HashMap` into a file.
pub fn write_weights_to_file<D: Device>(
    map: &[(impl AsRef<str>, &DenseMatrix<D>)],
    path: &str,
) -> Result<(), D::DeviceError> {
    use std::{fs::File, io::Write};

    let mut buf = Vec::new();

    for (id, weights) in map {
        let this_buf = weights.write_to_byte_buffer(id.as_ref()).unwrap();
        buf.extend_from_slice(&this_buf);
    }

    let mut file = File::create(path).unwrap();
    file.write_all(&buf).unwrap();

    Ok(())
}

/// Loads a set of labelled weights from a file into a `HashMap`.
pub fn load_weights_from_file(path: &str, old_format: bool) -> Vec<(String, Vec<f32>)> {
    use std::{fs::File, io::Read};

    let mut buf = Vec::new();
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buf).unwrap();

    let mut offset = 0;

    let mut res = Vec::new();

    while offset < buf.len() {
        let (buffer, id, bytes_read) = read_from_byte_buffer(&buf[offset..], old_format);
        res.push((id, buffer));
        offset += bytes_read;
    }

    res
}
