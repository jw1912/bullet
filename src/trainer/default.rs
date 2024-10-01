use std::collections::HashSet;

use bulletformat::BulletFormat;
use diffable::Node;

use crate::{
    inputs::InputType,
    loader::DataLoader,
    lr::LrScheduler,
    optimiser::Optimiser,
    outputs::{self, OutputBuckets},
    tensor::Shape,
    trainer::{DataPreparer, NetworkTrainer},
    wdl::WdlScheduler,
    Graph, LocalSettings, TrainingSchedule,
};

pub struct Trainer<Opt, Inp, Out = outputs::Single> {
    optimiser: Opt,
    input_getter: Inp,
    output_getter: Out,
    output_node: Node,
    perspective: bool,
    output_buckets: bool,
}

impl<Opt: Optimiser, Inp: InputType, Out: OutputBuckets<Inp::RequiredDataType>> NetworkTrainer
    for Trainer<Opt, Inp, Out>
{
    type Optimiser = Opt;
    type PreparedData = DefaultDataPreparer<Inp, Out>;

    fn load_batch(&mut self, prepared: &Self::PreparedData) -> usize {
        let batch_size = prepared.batch_size;

        let graph = self.optimiser.graph_mut();

        let input = &prepared.stm;
        graph.get_input_mut("stm").load_sparse_from_slice(input.shape, input.max_active, &input.value);

        if self.perspective {
            let input = &prepared.nstm;
            graph.get_input_mut("nstm").load_sparse_from_slice(input.shape, input.max_active, &input.value);
        }

        if self.output_buckets {
            let input = &prepared.buckets;
            graph.get_input_mut("buckets").load_sparse_from_slice(input.shape, input.max_active, &input.value);
        }

        graph.get_input_mut("targets").load_dense_from_slice(prepared.targets.shape, &prepared.targets.value);

        batch_size
    }

    fn optimiser(&self) -> &Self::Optimiser {
        &self.optimiser
    }

    fn optimiser_mut(&mut self) -> &mut Self::Optimiser {
        &mut self.optimiser
    }
}

impl<Opt: Optimiser, Inp: InputType, Out: OutputBuckets<Inp::RequiredDataType>> Trainer<Opt, Inp, Out> {
    pub fn new(graph: Graph, output_node: Node, params: Opt::Params, input_getter: Inp, output_getter: Out) -> Self {
        let inputs = graph.input_ids();
        let inputs = inputs.iter().map(String::as_str).collect::<HashSet<_>>();

        assert!(inputs.contains("stm"), "Graph does not contain stm inputs!");
        assert!(inputs.contains("targets"), "Graph does not contain targets!");

        let perspective = inputs.contains("nstm");
        let output_buckets = inputs.contains("buckets");
        let expected = 2 + usize::from(perspective) + usize::from(output_buckets);

        if inputs.len() != expected {
            println!("WARNING: The network graph contains an unexpected number of inputs!")
        };

        Self {
            optimiser: Opt::new(graph, params),
            input_getter,
            output_getter,
            output_node,
            perspective,
            output_buckets,
        }
    }

    pub fn train<D: DataLoader<Inp::RequiredDataType>, LR: LrScheduler, WDL: WdlScheduler>(
        &mut self,
        data_loader: D,
        schedule: &TrainingSchedule<LR, WDL>,
        settings: &LocalSettings,
    ) {
        let preparer = DefaultDataLoader::new(self.input_getter, self.output_getter, schedule.eval_scale, data_loader);

        self.train_custom(&preparer, schedule, settings, |_, _, _, _| {});
    }

    pub fn eval(&mut self, fen: &str) -> f32
    where
        Inp::RequiredDataType: std::str::FromStr<Err = String>,
    {
        let pos = format!("{fen} | 0 | 0.0").parse::<Inp::RequiredDataType>().unwrap();

        let prepared = DefaultDataPreparer::prepare(self.input_getter, self.output_getter, &[pos], 1, 1.0, 1.0);

        self.load_batch(&prepared);
        self.optimiser.graph_mut().forward();

        let eval = self.optimiser.graph().get_node(self.output_node);

        let mut val = vec![0.0; eval.values.dense().allocated_size()];
        eval.values.dense().write_to_slice(&mut val);
        val[0]
    }
}

#[derive(Clone)]
pub(crate) struct DefaultDataLoader<I, O, D> {
    input_getter: I,
    output_getter: O,
    scale: f32,
    loader: D,
}

impl<I, O, D> DefaultDataLoader<I, O, D> {
    pub fn new(input_getter: I, output_getter: O, scale: f32, loader: D) -> Self {
        Self { input_getter, output_getter, scale, loader }
    }
}

impl<I, O, D> DataPreparer for DefaultDataLoader<I, O, D>
where
    I: InputType,
    O: OutputBuckets<I::RequiredDataType>,
    D: DataLoader<I::RequiredDataType>,
{
    type DataType = I::RequiredDataType;
    type PreparedData = DefaultDataPreparer<I, O>;

    fn get_data_file_paths(&self) -> &[String] {
        self.loader.data_file_paths()
    }

    fn try_count_positions(&self) -> Option<u64> {
        self.loader.count_positions()
    }

    fn load_and_map_batches<F: FnMut(&[Self::DataType]) -> bool>(&self, batch_size: usize, f: F) {
        self.loader.map_batches(batch_size, f);
    }

    fn prepare(&self, data: &[Self::DataType], threads: usize, blend: f32) -> Self::PreparedData {
        DefaultDataPreparer::prepare(self.input_getter, self.output_getter, data, threads, blend, self.scale)
    }
}

pub struct DenseInput {
    pub shape: Shape,
    pub value: Vec<f32>,
}

#[derive(Clone)]
pub struct SparseInput {
    pub shape: Shape,
    pub value: Vec<i32>,
    pub max_active: usize,
}

impl Default for SparseInput {
    fn default() -> Self {
        Self { shape: Shape::new(0, 0), value: Vec::new(), max_active: 0 }
    }
}

/// A batch of data, in the correct format for the GPU.
pub struct DefaultDataPreparer<I, O> {
    input_getter: I,
    output_getter: O,
    pub batch_size: usize,
    pub stm: SparseInput,
    pub nstm: SparseInput,
    pub buckets: SparseInput,
    pub targets: DenseInput,
}

impl<I: InputType, O: OutputBuckets<I::RequiredDataType>> DefaultDataPreparer<I, O> {
    pub fn prepare(
        input_getter: I,
        output_getter: O,
        data: &[I::RequiredDataType],
        threads: usize,
        blend: f32,
        scale: f32,
    ) -> Self {
        let rscale = 1.0 / scale;
        let batch_size = data.len();
        let max_active = input_getter.max_active_inputs();
        let chunk_size = (batch_size + threads - 1) / threads;

        let input_size = input_getter.size();

        let mut prep = Self {
            input_getter,
            output_getter,
            batch_size,
            stm: SparseInput {
                shape: Shape::new(input_size, batch_size),
                max_active,
                value: vec![0; max_active * batch_size],
            },
            nstm: SparseInput {
                shape: Shape::new(input_size, batch_size),
                max_active,
                value: vec![0; max_active * batch_size],
            },
            buckets: SparseInput { shape: Shape::new(1, batch_size), max_active: 1, value: vec![0; batch_size] },
            targets: DenseInput { shape: Shape::new(1, batch_size), value: vec![0.0; batch_size] },
        };

        std::thread::scope(|s| {
            data.chunks(chunk_size)
                .zip(prep.stm.value.chunks_mut(max_active * chunk_size))
                .zip(prep.nstm.value.chunks_mut(max_active * chunk_size))
                .zip(prep.buckets.value.chunks_mut(chunk_size))
                .zip(prep.targets.value.chunks_mut(chunk_size))
                .for_each(|((((data_chunk, stm_chunk), nstm_chunk), buckets_chunk), results_chunk)| {
                    let inp = &prep.input_getter;
                    let out = &prep.output_getter;
                    s.spawn(move || {
                        let chunk_len = data_chunk.len();

                        for i in 0..chunk_len {
                            let pos = &data_chunk[i];
                            let mut j = 0;
                            let offset = max_active * i;

                            for (our, opp) in inp.feature_iter(pos) {
                                stm_chunk[offset + j] = our as i32;
                                nstm_chunk[offset + j] = opp as i32;
                                j += 1;
                            }

                            if j < max_active {
                                stm_chunk[offset + j] = -1;
                                nstm_chunk[offset + j] = -1;
                            }

                            assert!(j <= max_active, "More inputs provided than the specified maximum!");

                            results_chunk[i] = pos.blended_result(blend, rscale);
                            buckets_chunk[i] = i32::from(out.bucket(pos));
                        }
                    });
                });
        });

        prep
    }
}
