/// Simple architecture to fit to the MNIST dataset (download from https://www.kaggle.com/datasets/hojjatk/mnist-dataset/)
/// Move the following four files into data/mnist and run with `cargo r -r --example mnist`
/// - train-images.idx3-ubyte
/// - train-labels.idx1-ubyte
/// - t10k-images.idx3-ubyte
/// - t10k-labels.idx1-ubyte
use std::{collections::HashMap, fs::File, io::Read};

use acyclib::{
    device::{OperationError, tensor::Shape},
    graph::{Graph, GraphNodeId, GraphNodeIdTy, Node, builder::GraphBuilder, like::GraphLike},
    trainer::{
        DataLoadingError, Trainer,
        dataloader::{DataLoader, HostDenseMatrix, HostMatrix, PreparedBatchDevice, PreparedBatchHost},
        optimiser::{Optimiser, adam::AdamW},
        schedule::{TrainingSchedule, TrainingSteps},
    },
};
use bullet_lib::nn::{DeviceError, ExecutionContext, optimiser::AdamWParams};

fn main() -> Result<(), OperationError<DeviceError>> {
    let images = Images::new("data/mnist/train-images.idx3-ubyte");
    let labels = Labels::new("data/mnist/train-labels.idx1-ubyte");

    let valid_images = Images::new("data/mnist/t10k-images.idx3-ubyte");
    let valid_labels = Labels::new("data/mnist/t10k-labels.idx1-ubyte");

    let (graph, outputs) = make_model(&images);
    let params = AdamWParams { min_weight: -1000.0, max_weight: 1000.0, ..Default::default() };
    let optimiser = Optimiser::<_, _, AdamW<_>>::new(graph, params)?;
    let mut trainer = Trainer { optimiser, state: () };

    let schedule = TrainingSchedule {
        steps: TrainingSteps {
            batch_size: labels.vals.len() / 10,
            batches_per_superbatch: 100,
            start_superbatch: 1,
            end_superbatch: 20,
        },
        log_rate: 10,
        lr_schedule: Box::new(|_, sb| 0.001 * 0.9f32.powi(sb as i32 - 1)),
    };

    let dataloader = ImageDataLoader { images: images.clone(), labels: labels.clone() };

    let valid = prepare(&valid_images, &valid_labels);
    let valid_gpu = PreparedBatchDevice::new(trainer.optimiser.graph.devices(), &valid)?;

    trainer
        .train_custom(
            schedule,
            dataloader,
            |_, _, _, _| {},
            |trainer, _| {
                let graph = &mut trainer.optimiser.graph;
                let train_accuracy = calculate_accuracy(graph, outputs, &images, &labels).unwrap();
                valid_gpu.copy_into_graph(graph).unwrap();
                let valid_accuracy = calculate_accuracy(graph, outputs, &valid_images, &valid_labels).unwrap();
                println!("Train accuracy {train_accuracy:.2}% validation accuracy {valid_accuracy:.2}%");
            },
        )
        .unwrap();

    Ok(())
}

fn make_model(images: &Images) -> (Graph<ExecutionContext>, Node) {
    let builder = GraphBuilder::default();

    let inputs = builder.new_dense_input("inputs", images.shape);
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
    (builder.build(ExecutionContext::default()), outputs)
}

fn calculate_accuracy(
    graph: &mut Graph<ExecutionContext>,
    output_node: Node,
    images: &Images,
    labels: &Labels,
) -> Result<f32, OperationError<DeviceError>> {
    let batch_size = images.batch_size();
    let _ = graph.forward()?;

    let vals = graph.get(GraphNodeId::new(output_node.idx(), GraphNodeIdTy::Values))?.borrow().get_dense_vals()?;
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

    Ok(100.0 * correct as f32 / batch_size as f32)
}

fn prepare(images: &Images, labels: &Labels) -> PreparedBatchHost {
    let batch_size = labels.vals.len() / 10;
    let mut inputs = HashMap::new();

    let wrap = |x: &Vec<f32>, s| HostMatrix::Dense(HostDenseMatrix::new(x.clone(), Some(batch_size), s));

    let x = wrap(&images.vals, images.shape);
    inputs.insert("inputs".to_string(), x);

    let y = wrap(&labels.vals, Shape::new(10, 1));
    inputs.insert("targets".to_string(), y);

    PreparedBatchHost { batch_size, inputs }
}

struct ImageDataLoader {
    images: Images,
    labels: Labels,
}

impl DataLoader for ImageDataLoader {
    type Error = DataLoadingError;

    fn map_batches<F: FnMut(PreparedBatchHost) -> bool>(self, batch_size: usize, mut f: F) -> Result<(), Self::Error> {
        assert_eq!(batch_size, self.labels.vals.len() / 10);

        while !f(prepare(&self.images, &self.labels)) {}

        Ok(())
    }
}

#[derive(Clone)]
struct Images {
    vals: Vec<f32>,
    shape: Shape,
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
            shape: Shape::new(rows as usize, cols as usize),
        }
    }

    pub fn batch_size(&self) -> usize {
        self.vals.len() / self.shape.size()
    }
}

#[derive(Clone)]
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
