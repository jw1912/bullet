use bullet_lib::{
        nn::{
        optimiser::{AdamWOptimiser, AdamWParams},
        Activation, ExecutionContext, Graph, NetworkBuilder, Node, Shape,
    }, trainer::{
        default::{inputs::SparseInputType, loader, outputs, Trainer, formats::bulletformat::ChessBoard},
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    }
};

fn main() {
    let (graph, output_node) = build_network();

    println!("Params: {}", graph.get_num_params());

    let mut trainer = Trainer::<AdamWOptimiser, _, _>::new(
        graph,
        output_node,
        AdamWParams::default(),
        Chess12x64,
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

fn build_network() -> (Graph, Node) {
    let builder = NetworkBuilder::default();

    let dim = 16;

    // inputs
    let targets = builder.new_dense_input("targets", Shape::new(1, 1));
    let stm = builder.new_sparse_input("stm", Shape::new(768, 1), 32);

    // trainable weights
    let q = builder.new_affine_custom("q", 12, dim, 64);
    let k = builder.new_affine_custom("k", 12, dim, 64);
    let v = builder.new_affine_custom("v", 12, dim, 64);
    let o = builder.new_affine("o", 64 * dim, 1);

    let l0 = builder.new_affine("l0", 768, 256);
    let l1 = builder.new_affine("l1", 256, 4096);

    let p = l0.forward(stm).activate(Activation::SCReLU);
    let p = l1.forward(p).reshape(Shape::new(64, 64));

    let stm = stm.to_dense().reshape(Shape::new(12, 64));
    let q = q.forward(stm);
    let k = k.forward(stm);
    let v = v.forward(stm);

    let qkv = (q.gemm(true, k, false) + p).gemm(false, v, true);

    let out = qkv.reshape(Shape::new(64 * dim, 1));
    let out = out.activate(Activation::ReLU);
    let out = o.forward(out);

    let pred = out.activate(Activation::Sigmoid);
    pred.mse(targets);

    // graph, output node
    let output_node = out.node();
    (builder.build(ExecutionContext::default()), output_node)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Chess12x64;
impl SparseInputType for Chess12x64 {
    type RequiredDataType = ChessBoard;

    /// The total number of inputs
    fn num_inputs(&self) -> usize {
        768
    }

    /// The maximum number of active inputs
    fn max_active(&self) -> usize {
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        for (piece, square) in pos.into_iter() {
            let c = usize::from(piece & 8 > 0);
            let pc = usize::from(piece & 7);
            let sq = usize::from(square);

            let stm = 12 * sq + [0, 6][c] + pc;
            let ntm = 12 * (sq ^ 56) + [6, 0][c] + pc;
            f(stm, ntm)
        }
    }

    /// Shorthand for the input e.g. `768x4`
    fn shorthand(&self) -> String {
        "768".to_string()
    }

    /// Description of the input type
    fn description(&self) -> String {
        "Default psqt chess inputs".to_string()
    }
}
