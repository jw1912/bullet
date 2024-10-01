use bullet_lib::{
    inputs, loader, lr,
    operations::{self, Activation},
    optimiser::{AdamW, AdamWParams},
    outputs, wdl, ExecutionContext, GraphBuilder, LocalSettings, Shape, Trainer, TrainingSchedule, TrainingSteps,
};

fn main() {
    let mut builder = GraphBuilder::default();

    let stm = builder.create_input("stm", Shape::new(768, 1));
    let results = builder.create_input("results", Shape::new(1, 1));

    let l1w = builder.create_weights("l1w", Shape::new(128, 768));
    let l1b = builder.create_weights("l1b", Shape::new(128, 1));

    let l2w = builder.create_weights("l2w", Shape::new(1, 128));
    let l2b = builder.create_weights("l2b", Shape::new(1, 1));

    let l1 = operations::affine(&mut builder, l1w, stm, l1b);
    let l1a = operations::activate(&mut builder, l1, Activation::ReLU);
    let predicted = operations::affine(&mut builder, l2w, l1a, l2b);

    operations::mse(&mut builder, predicted, results);

    let ctx = ExecutionContext::default();
    let graph = builder.build(ctx);

    let mut trainer = Trainer::<AdamW, inputs::Chess768>::new(
        graph,
        predicted,
        AdamWParams::default(),
        inputs::Chess768,
        outputs::Single,
    );

    let schedule = TrainingSchedule {
        net_id: "test".to_string(),
        eval_scale: 400.0,
        ft_regularisation: 0.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 240,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.3, step: 60 },
        save_rate: 150,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = loader::DirectSequentialDataLoader::new(&["data/baseline.data"]);

    trainer.train(data_loader, &schedule, &settings);
}
