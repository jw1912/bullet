mod filter;
mod inputs;

use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, SparseInputType, get_num_buckets},
        outputs::MaterialCount,
    },
    trainer::schedule::{
        lr::{self, LrScheduler},
        wdl,
    },
    value::loader::{ViriBinpackLoader, viribinpack::ViriFilter},
};
use bullet_trainer::{
    model::{InitSettings, ModelDefinition, ModelEvaluator, ModelInputs, ModelWeights},
    optimiser::{
        Optimiser,
        adam::{AdamW, AdamWParams},
    },
    reader::ReadMapLoader,
    run::{DefaultDevice, TrainingSchedule, TrainingSteps, train},
};

const _NET_NAME: &str = "pawnocchio_new_relabeller";
const MAP_THREADS: u8 = 8;

const SUPERBATCHES_STAGE0: usize = 100;
const SUPERBATCHES_STAGE1: usize = 800;
const SUPERBATCHES_STAGE2: usize = 200;
const L1: usize = 768;
const L2: usize = 16;
const L3: usize = 32;
const Q0: i16 = 255;
const Q1: i16 = 128;
const _Q: i16 = 64;
const _INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);
const OUTPUT_BUCKETS: usize = 8;

const FT_SHIFT: usize = 8;
const FT_SHIFT_SCALE: f32 = Q0 as f32 / ((1 << FT_SHIFT) as f32);
const I8_RANGE: f32 = i8::MAX as f32 / (Q1 as f32);
const L1_RANGE: f32 = I8_RANGE * FT_SHIFT_SCALE * FT_SHIFT_SCALE;

#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  8,  9,  9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
];

fn main() {
    let pp = inputs::PawnPawnInputs::new(inputs::three_file_band_mask());
    let psqt = ChessBucketsMirrored::new(BUCKET_LAYOUT);
    let output_buckets = MaterialCount::<OUTPUT_BUCKETS>;

    let inputs = ModelInputs::default()
        .add_sparse("stm/pp", (pp.num_inputs(), 1), pp.max_active())
        .add_sparse("ntm/pp", (pp.num_inputs(), 1), pp.max_active())
        .add_sparse("stm/psqt", (psqt.num_inputs(), 1), psqt.max_active())
        .add_sparse("ntm/psqt", (psqt.num_inputs(), 1), psqt.max_active())
        .add_sparse("bucekts", (1, 1), 1)
        .add_dense("targets", (1, 1));

    let defn = ModelDefinition::build(
        &inputs,
        |builder, (((((stm_pp, ntm_pp), stm_psqt), ntm_psqt), output_buckets), target)| {
            let l0_pp = builder.new_affine("l0/pp/", pp.num_inputs(), L1);

            let l0f = builder.new_weights("l0/fac", (L1, 768), InitSettings::Zeroed);
            let mut l0_psqt = builder.new_affine("l0/psqt/", psqt.num_inputs(), L1);
            l0_psqt.weights = l0_psqt.weights + l0f.repeat(psqt.num_inputs() / 768);

            let l1 = builder.new_affine("l1/", L1, OUTPUT_BUCKETS * L2);
            let l2 = builder.new_affine("l2/", L2 * 2, OUTPUT_BUCKETS * L3);
            let l3 = builder.new_affine("l3/", L3, OUTPUT_BUCKETS);

            let ft = |pp, psqt, start, end| {
                (l0_pp.slice(start, end).forward(pp) + l0_psqt.slice(start, end).forward(psqt)).crelu()
            };
            let stm_hidden = ft(stm_pp, stm_psqt, 0, L1 / 2) * ft(stm_pp, stm_psqt, L1 / 2, L1);
            let ntm_hidden = ft(ntm_pp, ntm_psqt, 0, L1 / 2) * ft(ntm_pp, ntm_psqt, L1 / 2, L1);
            let l0_out = stm_hidden.concat(ntm_hidden);

            let ones_l1_vec = builder.new_constant((1, L1), &[1.0 / L1 as f32; L1]);
            let l0_out_norm = ones_l1_vec.matmul(l0_out);

            let l1_out = l1.forward(l0_out).select(output_buckets);
            let hl2 = l1_out.concat(l1_out.abs_pow(2.0)).crelu();

            let l2_out = l2.forward(hl2).select(output_buckets);
            let hl3 = l2_out.crelu();

            let l3_out = l3.forward(hl3).select(output_buckets);

            let loss = l3_out.sigmoid().squared_error(target);

            let loss = loss + 0.005 * l0_out_norm;

            (Some(loss), vec![("output".to_string(), l3_out)])
        },
    );

    let weights = ModelWeights::new(&defn, 12412421);
    let device = DefaultDevice::new(0).unwrap();
    let params = AdamWParams::default();

    let mut evaluator = ModelEvaluator::new(&defn, device.clone()).unwrap();
    let mut optimiser = Optimiser::<_, AdamW<_>>::new(defn, weights, device.clone(), params).unwrap();

    let l0_clip = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    optimiser.set_params_for_weight("l0/fac", l0_clip);
    optimiser.set_params_for_weight("l0/psqt/w", l0_clip);

    let l1_clip = AdamWParams { max_weight: L1_RANGE, min_weight: -L1_RANGE, ..Default::default() };
    optimiser.set_params_for_weight("l1/w", l1_clip);

    let reader = ViriBinpackLoader::new(
        "/k4/vine_data/vine_37/mixed_data_chonked.vf",
        8192,
        16,
        ViriFilter::Custom(filter::should_keep),
    );

    let params = (&inputs, &pp, psqt, output_buckets);

    const WARMUP_SBS: usize = SUPERBATCHES_STAGE0 / 2;
    const COOLDOWN_SBS: usize = SUPERBATCHES_STAGE0 - WARMUP_SBS;
    let stage0_mapper = inputs::make_inputs_mapper(params, wdl::ConstantWDL { value: 0.2 });
    train(
        &mut optimiser,
        TrainingSchedule {
            steps: TrainingSteps {
                batch_size: 16_384,
                batches_per_superbatch: 6104,
                start_superbatch: 1,
                end_superbatch: SUPERBATCHES_STAGE0,
            },
            lr_schedule: lr::Sequence {
                first: lr::LinearDecayLR { initial_lr: 1e-4, final_lr: 5e-3, final_superbatch: WARMUP_SBS },
                second: lr::LinearDecayLR { initial_lr: 5e-3, final_lr: 1e-4, final_superbatch: COOLDOWN_SBS },
                first_scheduler_final_superbatch: WARMUP_SBS,
            }
            .boxed(),
            log_rate: 128,
        },
        ReadMapLoader::new(reader.clone(), stage0_mapper, MAP_THREADS),
        |_, _, _| {},
        |_, _| {},
    )
    .unwrap();

    let stage1_mapper = inputs::make_inputs_mapper(params, wdl::LinearWDL { start: 0.2, end: 0.5 });
    train(
        &mut optimiser,
        TrainingSchedule {
            steps: TrainingSteps {
                batch_size: 16_384,
                batches_per_superbatch: 6104,
                start_superbatch: 1,
                end_superbatch: SUPERBATCHES_STAGE1,
            },
            lr_schedule: lr::LinearDecayLR { initial_lr: 1e-3, final_lr: 1e-6, final_superbatch: SUPERBATCHES_STAGE1 }
                .boxed(),
            log_rate: 128,
        },
        ReadMapLoader::new(reader.clone(), stage1_mapper, MAP_THREADS),
        |_, _, _| {},
        |_, _| {},
    )
    .unwrap();

    let stage2_mapper = inputs::make_inputs_mapper(params, wdl::ConstantWDL { value: 1.0 });
    train(
        &mut optimiser,
        TrainingSchedule {
            steps: TrainingSteps {
                batch_size: 16_384,
                batches_per_superbatch: 6104,
                start_superbatch: 1,
                end_superbatch: SUPERBATCHES_STAGE2,
            },
            lr_schedule: lr::LinearDecayLR { initial_lr: 1e-5, final_lr: 1e-7, final_superbatch: SUPERBATCHES_STAGE2 }
                .boxed(),
            log_rate: 128,
        },
        ReadMapLoader::new(reader, stage2_mapper, MAP_THREADS),
        |_, _, _| {},
        |_, _| {},
    )
    .unwrap();

    evaluator.load_device_weights(optimiser.weights()).unwrap();
    let evaluator_mapper = inputs::make_inputs_mapper(params, wdl::ConstantWDL { value: 0.0 });

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/P2P2PP/q2Q1R1K w kq - 0 2",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rn1qkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQka - 0 1",
        "3N4/b2R2p1/3q3r/6P1/4k1nQ/7B/8/K7 w - - 0 1",
        "k2B1Q1q/8/b7/4p3/3Pr3/1N5R/2n5/1K6 w - - 0 1",
        "1B3q2/8/r5n1/8/Rp1N1PQ1/8/4bk2/2K5 w - - 0 1",
        "8/5NR1/5q1b/8/7p/3P2B1/6Q1/1k1K1n1r w - - 0 1",
        "8/8/6r1/4B3/3Q3p/N1nq4/5RP1/b3K2k b - - 0 1",
        "3qn2Q/1R6/8/1N3b1p/4B3/1kP5/r7/5K2 b - - 0 1",
        "3rBR2/2qQ1p2/N7/2P2b2/6n1/k7/8/6K1 b - - 0 1",
        "k7/8/p1rB1q2/7Q/4R3/2N2n2/7P/6bK b - - 0 1",
        "2n2Rr1/Bk5p/N7/2Q3q1/b7/8/KP6/8 w - - 0 1",
        "8/Q6r/3qR1P1/b4p2/k7/3B4/1KN2n2/8 b - - 0 1",
        "2nR4/1qB5/2p5/7r/4bQ2/1P1N4/2K1k3/8 w - - 0 1",
        "8/2Q1B3/n3qR1r/bk1p4/1P6/8/3K4/7N w - - 0 1",
        "7r/4b3/4k1N1/2q4n/1Q2B3/R5p1/1P2K3/8 b - - 0 1",
        "2r1n1k1/NbR5/6B1/2p1P3/8/8/5K2/q6Q b - - 0 1",
        "2Q2R2/P1pn4/q1N5/1b5k/1r6/B7/6K1/8 b - - 0 1",
        "1Nr2b2/R1p5/5q2/7B/2P5/3nk3/7K/1Q6 w - - 0 1",
        "4Q3/6P1/1k3p2/4N3/2r5/K6b/1n1B2Rq/8 b - - 0 1",
        "1B5Q/1n6/2p1rN2/3R4/3P4/1K3k2/3b4/6q1 w - - 0 1",
        "3n4/3q4/5Q2/4rP2/1N2p3/2K2B2/5k2/2b4R b - - 0 1",
        "6B1/2k5/2n1R3/1q2p3/2P4Q/3K4/r5b1/3N4 w - - 0 1",
        "8/8/b6N/R3pr1n/Q7/1Pk1K3/4B3/5q2 b - - 0 1",
        "3Q2r1/4P2R/1b6/8/8/1B3K2/4p2q/1k1n1N2 w - - 0 1",
        "bR5q/2r3B1/2Q1P3/8/2n5/1N1p2K1/k7/8 w - - 0 1",
        "1q1b2r1/8/8/2p5/4N3/3k1P1K/2nB1Q2/4R3 w - - 0 1",
        "5rRq/8/1Qn5/8/K7/P1B4b/1p2N3/7k w - - 0 1",
        "1n6/8/B3q3/5R2/1KPb2N1/7Q/r4p2/2k5 w - - 0 1",
        "q3N1R1/8/1B5n/2p5/2K2P2/7r/1b1k4/7Q w - - 0 1",
        "1B6/N6q/2b5/7R/P2K4/1Q1pr3/6n1/2k5 b - - 0 1",
        "1R3q2/p3Q1n1/4N3/6r1/4K1B1/2P5/7b/4k3 w - - 0 1",
        "1k6/2RQP3/1p6/b7/1B3K2/r1n5/3Nq3/8 b - - 0 1",
        "b7/k7/5P2/n2N4/5pK1/2q5/2B2R2/r4Q2 b - - 0 1",
        "1B6/P4q2/5r2/8/1k2n2K/5b2/1NR1p3/6Q1 w - - 0 1",
        "8/3Pk3/B2r4/K5N1/b7/3n1p1Q/2R5/5q2 w - - 0 1",
        "q2Q1R2/2p4N/1b1P4/1K6/1B3r2/8/8/n2k4 w - - 0 1",
        "n1k5/5pq1/R4b2/2K5/3N4/7P/4BrQ1/8 b - - 0 1",
        "8/4Q3/B7/3KN1P1/3b4/nk3p2/8/R4r1q w - - 0 1",
        "b6n/B1k5/8/4KN1r/1Q6/7R/6Pp/5q2 b - - 0 1",
        "6k1/7r/8/bB3K1N/1R1q4/4Q3/2nP1p2/8 w - - 0 1",
        "Q6R/8/2B1q3/3N1nK1/2kb4/P7/r6p/8 w - - 0 1",
        "8/p5r1/k7/6PK/3b4/2B5/n4qQ1/3N2R1 w - - 0 1",
        "4kb2/6r1/K7/p7/6n1/2N5/2BP1qR1/7Q w - - 0 1",
        "6q1/1BN5/1K3P2/3br1np/3R4/Q7/8/5k2 w - - 0 1",
        "5n2/5q2/1NK5/k1P3r1/3p4/7Q/B6b/1R6 w - - 0 1",
        "B3r3/3p4/N2K2k1/1Q6/2R5/1bP5/1q5n/8 w - - 0 1",
        "BR2Q3/4N3/1n2K3/k7/1p1b1q2/8/5P2/7r b - - 0 1",
        "1k6/7R/5K1N/1pQ5/1n6/P4b2/1r6/6qB b - - 0 1",
        "8/3k4/3NnPK1/3QR3/3r2pB/8/4b3/q7 w - - 0 1",
        "1Q6/4q3/NB5K/1R1r4/3P4/bp1k4/6n1/8 w - - 0 1",
        "3Br3/K7/2q1N3/7n/8/4PbRQ/1p1k4/8 w - - 0 1",
        "R2r4/pK1b4/1n4NB/7P/8/3Q4/6k1/4q3 b - - 0 1",
        "3N2r1/2KP4/8/1B1p4/2b5/3RQq2/2k5/7n w - - 0 1",
        "5q2/1N1KB3/5b2/p4R2/4k3/P7/Q7/4n1r1 b - - 0 1",
        "NR6/4K3/1q3r2/3Q3P/3n2k1/8/7p/B5b1 b - - 0 1",
        "q7/1N1B1K2/1Q6/5b2/5pP1/6r1/n6k/R7 w - - 0 1",
        "2R5/2n1k1K1/5r2/3P4/2Q4p/2q5/6NB/7b w - - 0 1",
        "3n1Qr1/3p3K/8/3B4/R5b1/4P3/1qN4k/8 w - - 0 1",
        "K7/3k4/3n2b1/1P2r3/8/p2Bq3/3R4/3QN3 b - - 0 1",
        "1K6/8/3rRN2/1BP3b1/3p4/8/k2n2q1/5Q2 w - - 0 1",
        "2K5/6Bn/p4r2/2P1Q3/1qb5/8/2R5/3kN3 w - - 0 1",
        "3K4/8/2bP4/1qN5/2n3B1/3R4/4Qrp1/6k1 b - - 0 1",
        "1B2K1k1/P3b3/5q2/3R4/1pQ2r1n/8/8/6N1 b - - 0 1",
        "5K2/p4P1b/5QB1/4q3/6k1/8/4r3/R1n1N3 b - - 0 1",
        "6K1/8/b6R/N2p2P1/8/q1Q5/6r1/2Bk3n b - - 0 1",
        "7K/r2R3b/1Q6/8/2q5/1nPB2k1/N3p3/8 w - - 0 1",
    ] {
        let pos = format!("{fen} | 0 | 0.0").parse().unwrap();
        let inputs = evaluator_mapper.map(&[pos], Default::default(), 1).to_device(&device).unwrap();
        let output = evaluator.evaluate(&inputs).unwrap().get("output").unwrap();
        let [value] = output.to_host().unwrap().f32()[..] else { panic!() };
        println!("FEN: {fen}");
        println!("EVAL: {}", 400.0 * value);
    }
}
