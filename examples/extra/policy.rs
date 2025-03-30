use bullet_lib::{
    default::inputs::{self, SparseInputType},
    nn::{optimiser::AdamW, Shape},
    policy::{
        loader::PolicyDataLoader,
        move_maps::{self, MoveBucket},
        PolicyLocalSettings, PolicyTrainerBuilder, PolicyTrainingSchedule,
    },
    trainer::{
        save::{Layout, QuantTarget, SavedFormat},
        schedule::{lr, TrainingSteps},
    },
};

const HL: usize = 128;

fn main() {
    let inputs = inputs::Chess768;
    let transform = move_maps::NoTransform;
    let buckets = move_maps::NoBuckets;

    let num_inputs = inputs.num_inputs();
    let num_outputs = buckets.num_buckets() * move_maps::UNIQUE_CHESS_MOVES;

    let l1_size = 2 * HL;
    let l1_shape = Shape::new(num_outputs, l1_size);

    let save_format = [
        SavedFormat::new("l0w", QuantTarget::I16(255), Layout::Normal),
        SavedFormat::new("l0b", QuantTarget::I16(255), Layout::Normal),
        SavedFormat::new("l1w", QuantTarget::I16(64), Layout::Transposed(l1_shape)),
        SavedFormat::new("l1b", QuantTarget::I16(255 * 64), Layout::Normal),
    ];

    let mut trainer = PolicyTrainerBuilder::default()
        .dual_perspective()
        .inputs(inputs)
        .optimiser(AdamW)
        .move_mapper(transform, buckets)
        .save_format(&save_format)
        .build(|builder, stm, ntm| {
            let l0 = builder.new_affine("l0", num_inputs, HL);
            let l1 = builder.new_affine("l1", l1_size, num_outputs);

            let stm_subnet = l0.forward(stm).screlu();
            let ntm_subnet = l0.forward(ntm).screlu();

            l1.forward(stm_subnet.concat(ntm_subnet))
        });

    let schedule = PolicyTrainingSchedule {
        net_id: "policy001",
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.1, step: 18 },
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 40,
        },
        save_rate: 10,
    };

    let settings = PolicyLocalSettings { data_prep_threads: 4, output_directory: "checkpoints", batch_queue_size: 64 };

    let data_loader = PolicyDataLoader::new("data/policygen6.binpack", 4096);

    trainer.run(&schedule, &settings, &data_loader);
}
