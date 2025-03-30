mod builder;
mod loader;
mod move_maps;
mod preparer;

use bullet_core::optimiser::{Optimiser, OptimiserState};
use bulletformat::ChessBoard;
use loader::PolicyDataLoader;
use move_maps::{ChessMoveMapper, MoveBucket, SquareTransform};
use preparer::{PolicyDataPreparer, PolicyPreparedData};

use crate::{
    default::inputs::SparseInputType, lr::LrScheduler, trainer::{logger, save::SavedFormat, NetworkTrainer}, wdl::WdlScheduler, ExecutionContext, LocalSettings, TrainingSchedule
};

pub use builder::PolicyTrainerBuilder;

pub struct PolicyTrainer<Opt: OptimiserState<ExecutionContext>, Inp, T, B> {
    optimiser: Optimiser<ExecutionContext, Opt>,
    input_getter: Inp,
    move_mapper: ChessMoveMapper<T, B>,
    saved_format: Option<Vec<SavedFormat>>,
}

impl<Opt, Inp, T, B> PolicyTrainer<Opt, Inp, T, B>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType<RequiredDataType = ChessBoard>,
    T: SquareTransform,
    B: MoveBucket,
{
    pub fn run(
        &mut self,
        schedule: &TrainingSchedule<impl LrScheduler, impl WdlScheduler>,
        settings: &LocalSettings,
        data_loader: &PolicyDataLoader,
    ) {
        self.run_with_callback(schedule, settings, data_loader, |_, _, _, _| {});
    }

    pub fn run_with_callback<LR: LrScheduler, WDL: WdlScheduler>(
        &mut self,
        schedule: &TrainingSchedule<LR, WDL>,
        settings: &LocalSettings,
        data_loader: &PolicyDataLoader,
        mut callback: impl FnMut(usize, &Self, &TrainingSchedule<LR, WDL>, &LocalSettings),
    ) {
        let test_loader = settings.test_set.map(|test| PolicyDataLoader::new(test.path, 1));
        let (preparer, test_preparer) = self.training_preamble(schedule, settings, data_loader, &test_loader);

        self.train_custom(&preparer, &test_preparer, schedule, settings, |superbatch, trainer, schedule, settings| {
            callback(superbatch, trainer, schedule, settings)
        });
    }

    pub fn training_preamble<LR: LrScheduler, WDL: WdlScheduler>(
        &self,
        schedule: &TrainingSchedule<LR, WDL>,
        settings: &LocalSettings,
        data_loader: &PolicyDataLoader,
        test_loader: &Option<PolicyDataLoader>,
    ) -> Paired<PolicyDataPreparer<Inp, T, B>> {
        logger::clear_colours();
        println!("{}", logger::ansi("Beginning Training", "34;1"));

        schedule.display();
        settings.display();

        let preparer = PolicyDataPreparer::new(
            data_loader.clone(),
            self.input_getter.clone(),
            self.move_mapper,
        );

        let test_preparer = test_loader.as_ref().map(|loader| {
            PolicyDataPreparer::new(
                loader.clone(),
                self.input_getter.clone(),
                self.move_mapper,
            )
        });

        (preparer, test_preparer)
    }
}

type Paired<T> = (T, Option<T>);

impl<Opt, Inp, T, B> NetworkTrainer for PolicyTrainer<Opt, Inp, T, B>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType<RequiredDataType = ChessBoard>,
    T: SquareTransform,
    B: MoveBucket,
{
    type OptimiserState = Opt;
    type PreparedData = PolicyPreparedData<Inp>;

    fn load_batch(&mut self, prepared: &Self::PreparedData) -> usize {
        let graph = &mut self.optimiser.graph;

        let batch_size = prepared.batch_size;
        let expected_inputs = prepared.input_getter.num_inputs();

        unsafe {
            let input = &prepared.stm;
            let mut stm = graph.get_input_mut("stm");

            assert_eq!(stm.values.single_size(), expected_inputs);

            stm.load_sparse_from_slice(input.max_active, Some(batch_size), &input.value).unwrap();

            drop(stm);
            let input_ids = graph.input_ids();

            if input_ids.contains(&"nstm".to_string()) {
                let input = &prepared.ntm;
                let ntm = &mut *graph.get_input_mut("nstm");

                assert_eq!(ntm.values.single_size(), expected_inputs);

                ntm.load_sparse_from_slice(input.max_active, Some(batch_size), &input.value).unwrap();
            }
        }

        let mask = &prepared.mask;
        unsafe {
            graph.get_input_mut("mask").load_sparse_from_slice(mask.max_active, Some(batch_size), &mask.value).unwrap();
        }

        let dist = &prepared.dist;
        graph.get_input_mut("dist").load_dense_from_slice(Some(batch_size), &dist.value).unwrap();

        batch_size
    }

    fn optimiser(&self) -> &Optimiser<ExecutionContext, Self::OptimiserState> {
        &self.optimiser
    }

    fn optimiser_mut(&mut self) -> &mut Optimiser<ExecutionContext, Self::OptimiserState> {
        &mut self.optimiser
    }

    fn save_to_checkpoint(&self, path: &str) {
        std::fs::create_dir(path).unwrap_or(());

        let optimiser_path = format!("{path}/optimiser_state");
        std::fs::create_dir(optimiser_path.as_str()).unwrap_or(());
        self.optimiser().write_to_checkpoint(&optimiser_path).unwrap();

        if let Some(fmt) = &self.saved_format {
            if let Err(e) = self.save_weights_portion(&format!("{path}/saved.bin"), fmt) {
                println!("Failed to write raw network weights:");
                println!("{e}");
            }
        }
    }
}
