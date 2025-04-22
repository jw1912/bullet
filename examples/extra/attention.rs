use bullet_lib::{
    game::inputs::ChessBucketsMirrored,
    nn::{optimiser::AdamW, NetworkBuilder, NetworkBuilderNode, Shape},
    trainer::{
        default::loader,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
        NetworkTrainer,
    },
    value::ValueTrainerBuilder,
};

fn main() {
    let mut trainer = ValueTrainerBuilder::default()
        .inputs(ChessBucketsMirrored::new([0; 32]))
        .optimiser(AdamW)
        .save_format(&[])
        .loss_fn(|outputs, targets| outputs.sigmoid().squared_error(targets))
        .build(|builder, stm| {
            // network settings
            let dim = 128;
            let token_size = 64;
            let tokens = 12;
            let smolgen_size = 256;

            // weights
            let o1 = builder.new_affine_custom("o1", dim, 1, tokens);
            let o2 = builder.new_affine("o2", tokens, 1);

            let mut attn = AttentionDescription { dim, tokens, smolgen_size, id: 0, builder };

            // inference
            let mut out = stm.to_dense().reshape(Shape::new(token_size, tokens));
            out = attn.new_block(out).relu();
            out = o1.forward(out).relu();
            out = out.reshape(Shape::new(tokens, 1));
            o2.forward(out)
        });

    println!("Params: {}", trainer.optimiser().graph.get_num_params());

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

    let data_loader = loader::DirectSequentialDataLoader::new(&["data/baseline.data"]);

    trainer.run(&schedule, &settings, &data_loader);

    let eval = 400.0 * trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.0");
    println!("Eval: {eval:.3}cp");
}

#[derive(Clone, Copy)]
struct AttentionDescription<'a> {
    dim: usize,
    tokens: usize,
    smolgen_size: usize,
    id: usize,
    builder: &'a NetworkBuilder,
}

impl<'a> AttentionDescription<'a> {
    fn new_block(&mut self, input: NetworkBuilderNode<'a>) -> NetworkBuilderNode<'a> {
        let input_rows = input.node().shape.rows();

        let AttentionDescription { dim, tokens, smolgen_size, id, builder } = *self;

        let id = |s| format!("attn{id}/{s}");
        self.id += 1;

        let pa = builder.new_affine(&id("pa"), tokens * input_rows, smolgen_size);
        let pb = builder.new_affine(&id("pb"), smolgen_size, tokens * tokens);
        let q = builder.new_affine_custom(&id("q"), input_rows, dim, tokens);
        let k = builder.new_affine_custom(&id("k"), input_rows, dim, tokens);
        let v = builder.new_affine_custom(&id("v"), input_rows, dim, tokens);

        let pa = pa.forward(input.reshape(Shape::new(tokens * input_rows, 1))).relu();
        let p = pb.forward(pa).reshape(Shape::new(tokens, tokens));

        let q = q.forward(input);
        let k = k.forward(input);
        let v = v.forward(input);

        // QKV inputs are transposed
        let qk = k.gemm(true, q, false) + p;
        v.matmul(qk)
    }
}
