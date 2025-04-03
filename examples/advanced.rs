use bullet_lib::{
    game::{
        inputs::{Chess768, SparseInputType},
        outputs::MaterialCount,
    },
    nn::{optimiser::AdamW, InitSettings, Shape},
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader::DirectSequentialDataLoader, ValueTrainerBuilder},
};

const HL_SIZE: usize = 512;
const OUTPUT_BUCKETS: usize = 8;

fn main() {
    let inputs = Chess768;
    let num_inputs = inputs.num_inputs();

    let save_format = [
        SavedFormat::id("pst").quantise::<i16>(255),
        SavedFormat::id("l0w").quantise::<i16>(255),
        SavedFormat::id("l0b").quantise::<i16>(255),
        SavedFormat::id("l1w").quantise::<i16>(64).transpose(),
        SavedFormat::id("l1b").quantise::<i16>(64 * 255),
        SavedFormat::id("l2w").transpose(),
        SavedFormat::id("l2b"),
        SavedFormat::id("l3w").transpose(),
        SavedFormat::id("l3b"),
    ];

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .inputs(inputs)
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        .optimiser(AdamW)
        .save_format(&save_format)
        .loss_fn(|output, targets| output.sigmoid().squared_error(targets))
        .build(|builder, stm, ntm, buckets| {
            let l0 = builder.new_affine("l0", num_inputs, HL_SIZE);
            let l1 = builder.new_affine("l1", HL_SIZE, OUTPUT_BUCKETS * 16);
            let l2 = builder.new_affine("l2", 30, OUTPUT_BUCKETS * 32);
            let l3 = builder.new_affine("l3", 32, OUTPUT_BUCKETS);
            let pst = builder.new_weights("pst", Shape::new(1, num_inputs), InitSettings::Zeroed);

            let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm).crelu().pairwise_mul();
            let mut out = stm_subnet.concat(ntm_subnet);

            out = l1.forward(out).select(buckets);

            let skip_neuron = out.slice_rows(15, 16);
            out = out.slice_rows(0, 15);

            out = out.concat(out.abs_pow(2.0));
            out = out.crelu();

            out = l2.forward(out).select(buckets).screlu();
            out = l3.forward(out).select(buckets);

            let pst_out = (pst.matmul(stm) - pst.matmul(ntm)) / 2.0;
            out + skip_neuron + pst_out
        });

    let schedule = TrainingSchedule {
        net_id: "test".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 1024,
            start_superbatch: 1,
            end_superbatch: 10,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.3, step: 60 },
        save_rate: 150,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = DirectSequentialDataLoader::new(&["data/baseline.data"]);

    trainer.run(&schedule, &settings, &data_loader);

    let eval = 400.0 * trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.0");
    println!("Eval: {eval:.3}cp");
}
