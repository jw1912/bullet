/*
Code to relabel a bulletformat dataset with a network
*/

use bullet_lib::{
    nn::{Activation, ExecutionContext, Graph, NetworkBuilder, Node, Shape},
    trainer::default::{
        formats::bulletformat::{ChessBoard, DataLoader},
        inputs::{self, SparseInputType},
        load_into_graph,
        loader::DefaultDataPreparer,
        outputs,
    },
};
use bulletformat::BulletFormat;
use std::{fs::File, io::BufWriter, time::Instant};

const NETWORK_PATH: &str = "checkpoints/monty-datagen25-240/optimiser_state/weights.bin";
const DATA_PATH: &str = "data/baseline.data";
const OUTPUT_PATH: &str = "data/relabled.data";

fn main() {
    #[rustfmt::skip]
    let inputs = inputs::ChessBucketsMirrored::new([
        0, 0, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        3, 3, 3, 3,
        3, 3, 3, 3,
        3, 3, 3, 3,
        3, 3, 3, 3,
        3, 3, 3, 3,
    ]);
    let output_buckets = outputs::Single;
    let hl_size = 1024;
    let batch_size = 16384;
    let eval_scale = 400.0;

    let (sender, receiver) = std::sync::mpsc::sync_channel(2);

    std::thread::spawn(move || {
        let loader = DataLoader::new(DATA_PATH, 128).unwrap();

        loader.map_batches(batch_size, |batch: &[ChessBoard]| {
            let prepared = DefaultDataPreparer::prepare(inputs, output_buckets, false, batch, 4, 0.0, eval_scale);
            sender.send((batch.to_vec(), prepared)).unwrap();
        });

        drop(sender);
    });

    let (sender2, receiver2) = std::sync::mpsc::sync_channel(2);

    std::thread::spawn(move || {
        let (mut graph, output_node) = build_network(inputs.num_inputs(), inputs.max_active(), hl_size);
        graph.load_from_file(NETWORK_PATH, true).unwrap();

        let mut error = 0.0;
        let mut batches = 0;
        let mut positions = 0;
        let t = Instant::now();

        while let Ok((mut batch, prepared)) = receiver.recv() {
            unsafe {
                load_into_graph(&mut graph, &prepared).unwrap();
            }

            error += f64::from(graph.forward().unwrap());
            batches += 1;
            positions += batch.len();

            let scores = graph.get_node(output_node).get_dense_vals().unwrap();

            assert_eq!(batch.len(), scores.len());

            for (pos, result) in batch.iter_mut().zip(scores.iter()) {
                pos.score = (result * eval_scale).clamp(-32000.0, 32000.0) as i16;
            }

            sender2.send(batch).unwrap();

            if batches % 256 == 0 {
                let err = error / positions as f64;
                let pps = positions as f64 / t.elapsed().as_secs_f64() / 1000.0;
                println!("Avg Error: {err:.6}, Pos/Sec {pps:.1}k");
            }
        }

        println!("Total Positions: {positions}");
    });

    let mut writer = BufWriter::new(File::create(OUTPUT_PATH).unwrap());
    while let Ok(batch) = receiver2.recv() {
        ChessBoard::write_to_bin(&mut writer, &batch).unwrap();
    }
}

fn build_network(num_inputs: usize, nnz: usize, hl: usize) -> (Graph, Node) {
    let builder = NetworkBuilder::default();

    // inputs
    let stm = builder.new_sparse_input("stm", Shape::new(num_inputs, 1), nnz);
    let nstm = builder.new_sparse_input("nstm", Shape::new(num_inputs, 1), nnz);
    let targets = builder.new_dense_input("targets", Shape::new(1, 1));

    // trainable weights
    let l0 = builder.new_affine("l0", num_inputs, hl);
    let l1 = builder.new_affine("l1", 2 * hl, 1);

    // inference
    let mut out = l0.forward_sparse_dual_with_activation(stm, nstm, Activation::SCReLU);
    out = l1.forward(out);

    let pred = out.sigmoid();
    pred.squared_error(targets);

    // graph, output node
    let output_node = out.node();
    (builder.build(ExecutionContext::default()), output_node)
}
