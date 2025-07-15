/// Simple architecture to fit to the MNIST dataset (download from https://www.kaggle.com/datasets/hojjatk/mnist-dataset/)
/// Move the following four files into data/mnist and run with `cargo r -r --example mnist`
/// - train-images.idx3-ubyte
/// - train-labels.idx1-ubyte
/// - t10k-images.idx3-ubyte
/// - t10k-labels.idx1-ubyte
use std::{fs::File, io::Read, time::Instant};

use bullet_core::{
    device::OperationError,
    graph::{
        builder::{GraphBuilder, Shape},
        Graph, Node, NodeId, NodeIdTy,
    },
};
use bullet_hip_backend::{DeviceError, ExecutionContext};

fn main() -> Result<(), OperationError<DeviceError>> {
    let images = Images::new("data/mnist/train-images.idx3-ubyte");
    let labels = Labels::new("data/mnist/train-labels.idx1-ubyte");
    let batch_size = labels.vals.len() / 10;

    let validation_images = Images::new("data/mnist/t10k-images.idx3-ubyte");
    let validation_labels = Labels::new("data/mnist/t10k-labels.idx1-ubyte");
    let validation_batch_size = validation_labels.vals.len() / 10;

    assert_eq!(batch_size, images.batch_size());
    assert_eq!(validation_batch_size, validation_images.batch_size());

    let builder = GraphBuilder::default();

    let inputs = builder.new_dense_input("inputs", Shape::new(images.rows, images.cols));
    let targets = builder.new_dense_input("targets", Shape::new(10, 1));

    let l0 = builder.new_affine("l0", 28 * 28, 128);
    let l1 = builder.new_affine("l1", 128, 128);
    let l2 = builder.new_affine("l2", 128, 10);

    let f0 = l0.forward(inputs.reshape(Shape::new(28 * 28, 1))).sigmoid();
    let f1 = l1.forward(f0).sigmoid();
    let f2 = l2.forward(f1);
    let losses = f2.softmax_crossentropy_loss(targets);

    let ones = builder.new_constant(Shape::new(1, 10), &[1.0; 10]);
    ones.matmul(losses);

    let outputs = f2.node();
    let mut graph = builder.build(ExecutionContext::default());

    graph.load_from_file("checkpoints/mnist.bin", false)?;

    graph.get_input_mut("inputs").load_dense_from_slice(Some(batch_size), &images.vals)?;
    graph.get_input_mut("targets").load_dense_from_slice(Some(batch_size), &labels.vals)?;

    let t = Instant::now();
    let lr = 0.0001;

    for epoch in 1..=10 {
        graph.zero_grads()?;
        graph.forward()?;
        graph.backward()?;

        if epoch % 10 == 0 {
            let valid_acc = calculate_accuracy(&mut graph, outputs, &validation_images, &validation_labels)?;
            let train_acc = calculate_accuracy(&mut graph, outputs, &images, &labels)?;

            println!(
                "epoch {epoch} train accuracy {train_acc:.2}% validation accuarcy {valid_acc:.2}% time {:.3}s",
                t.elapsed().as_secs_f32()
            );

            graph.get_input_mut("inputs").load_dense_from_slice(Some(batch_size), &images.vals)?;
            graph.get_input_mut("targets").load_dense_from_slice(Some(batch_size), &labels.vals)?;
        }

        for id in &graph.weight_ids() {
            let idx = graph.weight_idx(id).unwrap();
            let weight = &mut *graph.get_mut(NodeId::new(idx, NodeIdTy::Values)).unwrap();

            if let Ok(grd) = graph.get(NodeId::new(idx, NodeIdTy::Gradients)) {
                weight.values.dense_mut()?.add(-lr, grd.dense()?)?;
            }
        }
    }

    Ok(())
}

fn calculate_accuracy(
    graph: &mut Graph<ExecutionContext>,
    output_node: Node,
    images: &Images,
    labels: &Labels,
) -> Result<f32, OperationError<DeviceError>> {
    let batch_size = images.batch_size();
    graph.get_input_mut("inputs").load_dense_from_slice(Some(batch_size), &images.vals)?;
    graph.get_input_mut("targets").load_dense_from_slice(Some(batch_size), &labels.vals)?;
    let _ = graph.forward()?;

    let vals = graph.get(NodeId::new(output_node.idx(), NodeIdTy::Values))?.get_dense_vals()?;
    let mut correct = 0;
    for (predicted, &expected) in vals.chunks(10).zip(labels.indices.iter()) {
        let mut max = f32::NEG_INFINITY;
        let mut best = 110;

        for (i, &x) in predicted.iter().enumerate() {
            if x > max {
                max = x;
                best = i;
            }
        }

        assert!(best < 10);

        if best == expected.into() {
            correct += 1;
        }
    }

    assert_eq!(batch_size, labels.indices.len());

    Ok(100.0 * correct as f32 / batch_size as f32)
}

struct Images {
    vals: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Images {
    pub fn new(file: &str) -> Self {
        let mut reader = File::open(file).unwrap();

        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf).unwrap();
        let magic = u32::from_be_bytes(buf);
        reader.read_exact(&mut buf).unwrap();
        let size = u32::from_be_bytes(buf);
        reader.read_exact(&mut buf).unwrap();
        let rows = u32::from_be_bytes(buf);
        reader.read_exact(&mut buf).unwrap();
        let cols = u32::from_be_bytes(buf);

        assert_eq!(magic, 2051);

        let mut bytes = Vec::new();
        let num_bytes = reader.read_to_end(&mut bytes).unwrap();

        assert_eq!(size * rows * cols, num_bytes as u32);

        Self {
            vals: bytes.iter().map(|&x| f32::from(x) / f32::from(u8::MAX)).collect(),
            rows: rows as usize,
            cols: cols as usize,
        }
    }

    pub fn batch_size(&self) -> usize {
        self.vals.len() / (self.rows * self.cols)
    }
}

struct Labels {
    vals: Vec<f32>,
    indices: Vec<u8>,
}

impl Labels {
    pub fn new(file: &str) -> Self {
        let mut reader = File::open(file).unwrap();

        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf).unwrap();
        let magic = u32::from_be_bytes(buf);
        reader.read_exact(&mut buf).unwrap();
        let size = u32::from_be_bytes(buf);

        assert_eq!(magic, 2049);

        let mut bytes = Vec::new();
        let num_bytes = reader.read_to_end(&mut bytes).unwrap();

        assert_eq!(size, num_bytes as u32);

        let mut vals = vec![0.0; 10 * size as usize];

        for (i, &byte) in bytes.iter().enumerate() {
            vals[10 * i + usize::from(byte)] = 1.0;
        }

        Self { vals, indices: bytes }
    }
}
