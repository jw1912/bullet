use std::collections::HashMap;

use crate::{nn::Graph, tensor::DenseMatrix};

/// Writes the weights of a graph to a file. If `gradients` is true,
/// it will instead write the gradients of those weights.
pub fn write_graph_weights_to_file(graph: &Graph, path: &str) {
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
pub fn load_graph_weights_from_file(graph: &mut Graph, path: &str) {
    use std::{fs::File, io::Read};

    let mut buf = Vec::new();
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buf).unwrap();

    let mut offset = 0;
    let mut matrix_buffer = DenseMatrix::default();

    while offset < buf.len() {
        let (id, bytes_read) = matrix_buffer.read_from_byte_buffer(&buf[offset..]);

        let weights = graph.get_weights_mut(&id);
        matrix_buffer.copy_into(weights.values.dense_mut());

        offset += bytes_read;
    }
}

/// Write a set of labelled weights from a `HashMap` into a file.
pub fn write_weight_hashmap_to_file(map: &HashMap<String, DenseMatrix>, path: &str) {
    use std::{fs::File, io::Write};

    let mut buf = Vec::new();

    for (id, weights) in map {
        let this_buf = weights.write_to_byte_buffer(id).unwrap();
        buf.extend_from_slice(&this_buf);
    }

    let mut file = File::create(path).unwrap();
    file.write_all(&buf).unwrap();
}

/// Loads a set of labelled weights from a file into a `HashMap`.
pub fn load_weight_hashmap_from_file(map: &mut HashMap<String, DenseMatrix>, path: &str) {
    use std::{fs::File, io::Read};

    let mut buf = Vec::new();
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buf).unwrap();

    let mut offset = 0;
    let mut matrix_buffer = DenseMatrix::default();

    while offset < buf.len() {
        let (id, bytes_read) = matrix_buffer.read_from_byte_buffer(&buf[offset..]);

        matrix_buffer.copy_into(map.get_mut(&id).unwrap());

        offset += bytes_read;
    }
}
