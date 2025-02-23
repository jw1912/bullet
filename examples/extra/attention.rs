use bullet_lib::{
    nn::{
        optimiser::{AdamWOptimiser, AdamWParams},
        Activation, ExecutionContext, Graph, InitSettings, NetworkBuilder, Node, Shape,
    },
    trainer::{
        default::{inputs, loader, outputs, Trainer},
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
};

type InputFeatures = inputs::Chess768;

fn main() {
    let inputs = InputFeatures::default();

    let num_inputs = <InputFeatures as inputs::SparseInputType>::num_inputs(&inputs);
    let max_active = <InputFeatures as inputs::SparseInputType>::max_active(&inputs);

    let (graph, output_node) = build_network(num_inputs, max_active);

    println!("Params: {}", graph.get_num_params());

    let mut trainer = Trainer::<AdamWOptimiser, _, _>::new(
        graph,
        output_node,
        AdamWParams::default(),
        inputs,
        outputs::Single,
        Vec::new(),
        false,
    );

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

fn build_network(num_inputs: usize, max_active: usize) -> (Graph, Node) {
    let builder = NetworkBuilder::default();

    let dim = 16;

    // inputs
    let targets = builder.new_dense_input("targets", Shape::new(1, 1));
    let stm = builder.new_sparse_input("stm", Shape::new(num_inputs, 1), max_active);
    let stm = stm.to_dense().reshape(Shape::new(64, 12));

    // trainable weights
    let init = InitSettings::Normal { mean: 0.0, stdev: 1.0 / 8.0 };
    let q = builder.new_weights("q", Shape::new(dim, 64), init);
    let k = builder.new_weights("k", Shape::new(dim, 64), init);
    let v = builder.new_weights("v", Shape::new(dim, 64), init);
    let p = builder.new_weights("p", Shape::new(12, 12), init);
    let o = builder.new_affine("o", 12 * dim, 1);

    let q = q.matmul(stm);
    let k = k.matmul(stm);
    let v = v.matmul(stm);

    let qkv = (q.gemm(true, k, false) + p).gemm(false, v, true);

    let out = qkv.reshape(Shape::new(12 * dim, 1));
    let out = out.activate(Activation::ReLU);
    let out = o.forward(out);

    let pred = out.activate(Activation::Sigmoid);
    pred.mse(targets);

    // graph, output node
    let output_node = out.node();
    (builder.build(ExecutionContext::default()), output_node)
}
