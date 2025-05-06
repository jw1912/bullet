use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirroredFactorised, SparseInputType},
        outputs::MaterialCount,
    },
    nn::optimiser,
    trainer::{
        default::loader,
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::ValueTrainerBuilder,
};

macro_rules! net_id {
    () => {
        "bullet_r45_768x8-1024x2-1x8"
    };
}

const NET_ID: &str = net_id!();

const HL: usize = 1024;
const OUTPUT_BUCKETS: usize = 8;

type Input = ChessBucketsMirroredFactorised;
type Output = MaterialCount<OUTPUT_BUCKETS>;

fn main() {
    #[rustfmt::skip]
    let inputs = Input::new([
        0, 1, 2, 3,
        4, 4, 5, 5,
        6, 6, 6, 6,
        6, 6, 6, 6,
        7, 7, 7, 7,
        7, 7, 7, 7,
        7, 7, 7, 7,
        7, 7, 7, 7,
    ]);
    let output = Output::default();

    let save_format = [
        SavedFormat::id("l0w").quantise::<i16>(255),
        SavedFormat::id("l0b").quantise::<i16>(255),
        SavedFormat::id("l1w").quantise::<i16>(64).transpose(),
        SavedFormat::id("l1b").quantise::<i16>(64 * 255),
    ];

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .save_format(&save_format)
        .optimiser(optimiser::Ranger)
        .loss_fn(|output, targets| output.sigmoid().squared_error(targets))
        .inputs(inputs)
        .output_buckets(output)
        .build(|builder, stm, ntm, buckets| {
            let l0 = builder.new_affine("l0", inputs.num_inputs(), HL);
            let l1 = builder.new_affine("l1", HL * 2, OUTPUT_BUCKETS);

            let stm_subnet = l0.forward(stm).screlu();
            let ntm_subnet = l0.forward(ntm).screlu();
            l1.forward(stm_subnet.concat(ntm_subnet)).select(buckets)
        });

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

    let data_loader = loader::DirectSequentialDataLoader::new(&["../../chess/data/shuffled.data"]);

    trainer.set_optimiser_params(optimiser::RangerParams::default());
    trainer.run(&schedule, &settings, &data_loader);
}
