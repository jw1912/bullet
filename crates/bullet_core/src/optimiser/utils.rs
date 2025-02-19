use crate::{device::Device, graph::Graph, tensor::DenseMatrix};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    Before,
    After,
}

/// Writes the weights of a graph to a file. If `gradients` is true,
/// it will instead write the gradients of those weights.
pub fn write_graph_weights_to_file<D: Device>(graph: &Graph<D>, path: &str) {
    use std::{fs::File, io::Write};

    let weight_ids = graph.weight_ids();

    let mut buf = Vec::new();

    for id in &weight_ids {
        let weights = graph.get_weights(id);
        let this_buf = weights.values.dense().write_to_byte_buffer(id).unwrap();

        buf.extend_from_slice(&this_buf);
    }

    let mut file = File::create(path).unwrap();
    file.write_all(&buf).unwrap();
}

/// Loads the weights of a graph from a file. If `gradients` is true,
/// it will instead load the gradients of those weights.
pub fn load_graph_weights_from_file<D: Device>(graph: &mut Graph<D>, path: &str, old_format: bool) {
    use std::{fs::File, io::Read};

    let mut buf = Vec::new();
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buf).unwrap();

    let mut offset = 0;

    while offset < buf.len() {
        let (buffer, id, bytes_read) = read_from_byte_buffer(&buf[offset..], old_format);
        graph.get_weights_mut(&id).load_dense_from_slice(None, &buffer);

        offset += bytes_read;
    }
}

/// Write a set of labelled weights from a `HashMap` into a file.
pub fn write_weights_to_file<D: Device>(map: &[(impl AsRef<str>, &DenseMatrix<D>)], path: &str) {
    use std::{fs::File, io::Write};

    let mut buf = Vec::new();

    for (id, weights) in map {
        let this_buf = weights.write_to_byte_buffer(id.as_ref()).unwrap();
        buf.extend_from_slice(&this_buf);
    }

    let mut file = File::create(path).unwrap();
    file.write_all(&buf).unwrap();
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

/// Reads a matrix from a byte buffer, returning how many bytes were read
/// and the matrix ID that was read.
pub fn read_from_byte_buffer(bytes: &[u8], old_format: bool) -> (Vec<f32>, String, usize) {
    const USIZE: usize = std::mem::size_of::<usize>();

    let mut offset = 0;

    let mut id = String::new();
    loop {
        let ch = bytes[offset];
        offset += 1;

        if ch == b'\n' {
            break;
        }

        id.push(char::from(ch));
    }

    let mut single_size = [0u8; USIZE];
    single_size.copy_from_slice(&bytes[offset..offset + USIZE]);
    offset += USIZE;

    let mut single_size = usize::from_le_bytes(single_size);

    if old_format {
        let mut cols = [0u8; USIZE];
        cols.copy_from_slice(&bytes[offset..offset + USIZE]);
        offset += USIZE;
        single_size *= usize::from_le_bytes(cols);
    }

    let total_read = offset + single_size * 4;

    let mut values = vec![0.0; single_size];

    for (word, val) in bytes[offset..total_read].chunks_exact(4).zip(values.iter_mut()) {
        let mut buf = [0; 4];
        buf.copy_from_slice(word);
        *val = f32::from_le_bytes(buf);
    }

    (values, id, total_read)
}
