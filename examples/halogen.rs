use bullet_lib::{
    default::{Loss, TrainerBuilder},
    game::{inputs::ChessBucketsMirroredFactorised, outputs::MaterialCount},
    nn::optimiser,
    trainer::{
        default::loader,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    Activation,
};

macro_rules! net_id {
    () => {
        "bullet_r46_768x8-1024x2-1x8"
    };
}

const NET_ID: &str = net_id!();

fn main() {
    #[rustfmt::skip]

    let inputs = ChessBucketsMirroredFactorised::new([
        0, 1, 2, 3,
        4, 4, 5, 5,
        6, 6, 6, 6,
        6, 6, 6, 6,
        7, 7, 7, 7,
        7, 7, 7, 7,
        7, 7, 7, 7,
        7, 7, 7, 7,
    ]);

    let mut trainer = TrainerBuilder::default()
        .quantisations(&[255, 64])
        .optimiser(optimiser::Ranger)
        .loss_fn(Loss::SigmoidMSE)
        .input(inputs)
        .output_buckets(MaterialCount::<8>)
        .feature_transformer(1024)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: NET_ID.to_string(),
        eval_scale: 160.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 400,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.3 },
        lr_scheduler: lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.0, final_superbatch: 400 },
        save_rate: 10,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = loader::DirectSequentialDataLoader::new(&["../../chess/data/rescored.data"]);

    trainer.set_optimiser_params(optimiser::RangerParams::default());
    //trainer.load_from_checkpoint("checkpoints/bullet_r46_768x8-1024x2-1x8-400");
    trainer.run(&schedule, &settings, &data_loader);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let eval = trainer.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", 160.0 * eval);
    }

    //trainer.save_quantised("nets/bullet_r46_768x8-1024x2-1x8-round.nn").unwrap();
}
