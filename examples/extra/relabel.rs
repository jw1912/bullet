/// This is much slower than it could be for the sake of simplicity.
/// To improve speed you would need to asynchronously read and write data,
/// and asynchronously copy next batch to GPU whilst current batch is computing.
use std::{fs::File, io::BufWriter, time::Instant};

use acyclib::{graph::like::GraphLike, trainer::dataloader::PreparedBatchDevice};
use bullet_lib::{
    game::{formats::bulletformat::DataLoader, inputs::Chess768},
    nn::optimiser::AdamW,
    value::ValueTrainerBuilder,
};
use bulletformat::{BulletFormat, ChessBoard};

const HIDDEN_SIZE: usize = 128;

fn main() {
    let threads = 4;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(Chess768)
        .save_format(&[])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", 768, HIDDEN_SIZE);
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, 1);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer)
        });

    trainer.load_from_checkpoint("checkpoints/simple-40");

    let loader = DataLoader::new("data/baseline.data", 1024).unwrap();

    let state = trainer.state.clone();

    let mut file = BufWriter::new(File::create("data/relabelled.data").unwrap());

    let total = loader.len();
    println!("Positions to relabel: {total}");

    let t = Instant::now();

    let mut relabelled = 0usize;

    loader.map_batches(16384 * 128, |batch| {
        let prepared = state.prepare(batch, threads, 1.0, 1.0);
        let mut device_data = PreparedBatchDevice::new(trainer.optimiser.graph.devices(), &prepared).unwrap();
        device_data.load_into_graph(&mut trainer.optimiser.graph).unwrap();
        trainer.optimiser.graph.execute_fn("forward").unwrap();
        trainer.optimiser.graph.synchronise().unwrap();

        let mut batch = batch.to_vec();
        let new_evals = trainer.get_output_values();

        assert_eq!(batch.len(), new_evals.len());

        for (position, new_eval) in batch.iter_mut().zip(new_evals) {
            position.score = (400.0 * new_eval) as i16;
        }

        relabelled += batch.len();

        let pos_per_sec = relabelled as f64 / t.elapsed().as_secs_f64();
        println!("Relabelled {relabelled} / {total} (~{pos_per_sec:.0} pos/sec)");

        ChessBoard::write_to_bin(&mut file, &batch).unwrap();
    });
}
