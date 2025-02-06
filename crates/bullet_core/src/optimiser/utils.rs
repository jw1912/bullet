use std::{collections::HashMap, sync::Arc};

use crate::{device::Device, graph::Graph, tensor::DenseMatrix};

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
pub fn load_graph_weights_from_file<D: Device>(graph: &mut Graph<D>, path: &str) {
    use std::{fs::File, io::Read};

    let mut buf = Vec::new();
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buf).unwrap();

    let mut offset = 0;

    while offset < buf.len() {
        let (buffer, id, bytes_read) = DenseMatrix::read_from_byte_buffer(graph.device(), &buf[offset..]);
        *graph.get_weights_mut(&id).values.dense_mut() = buffer;

        offset += bytes_read;
    }
}

/// Write a set of labelled weights from a `HashMap` into a file.
pub fn write_weight_hashmap_to_file<D: Device>(map: &HashMap<String, DenseMatrix<D>>, path: &str) {
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
pub fn load_weight_hashmap_from_file<D: Device>(device: Arc<D>, map: &mut HashMap<String, DenseMatrix<D>>, path: &str) {
    use std::{fs::File, io::Read};

    let mut buf = Vec::new();
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buf).unwrap();

    let mut offset = 0;

    while offset < buf.len() {
        let (buffer, id, bytes_read) = DenseMatrix::read_from_byte_buffer(device.clone(), &buf[offset..]);

        *map.get_mut(&id).unwrap() = buffer;

        offset += bytes_read;
    }
}
