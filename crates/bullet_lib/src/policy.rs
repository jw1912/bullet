mod builder;
pub mod loader;
pub mod move_maps;
mod preparer;

use bullet_core::optimiser::{Optimiser, OptimiserState};
use bulletformat::ChessBoard;
use loader::PolicyDataLoader;
use move_maps::{ChessMoveMapper, MoveBucket, SquareTransform};
use preparer::{PolicyDataPreparer, PolicyPreparedData};

use crate::{
    game::inputs::SparseInputType,
    nn::ExecutionContext,
    trainer::{
        logger,
        save::SavedFormat,
        schedule::{lr::LrScheduler, wdl::ConstantWDL, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
        NetworkTrainer,
    },
};

#[derive(Clone)]
pub struct PolicyTrainingSchedule<'a, LR> {
    pub net_id: &'a str,
    pub steps: TrainingSteps,
    pub lr_scheduler: LR,
    pub save_rate: usize,
}

#[derive(Clone, Copy)]
pub struct PolicyLocalSettings<'a> {
    pub data_prep_threads: usize,
    /// Directory to write checkpoints to.
    pub output_directory: &'a str,
    /// Number of batches that the dataloader can prepare and put in a queue before
    /// they are processed in training.
    pub batch_queue_size: usize,
}

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
    pub fn profile_all_nodes(&mut self) {
        self.optimiser.graph.profile_all_nodes();
    }

    pub fn report_profiles(&self) {
        self.optimiser.graph.report_profiles();
    }

    pub fn run(
        &mut self,
        schedule: &PolicyTrainingSchedule<impl LrScheduler>,
        settings: &PolicyLocalSettings,
        data_loader: &PolicyDataLoader,
    ) {
        self.run_with_callback(schedule, settings, data_loader, |_, _, _, _| {});
    }

    pub fn run_with_callback<LR: LrScheduler>(
        &mut self,
        schedule: &PolicyTrainingSchedule<LR>,
        settings: &PolicyLocalSettings,
        data_loader: &PolicyDataLoader,
        mut callback: impl FnMut(usize, &Self, &TrainingSchedule<LR, ConstantWDL>, &LocalSettings),
    ) {
        let PolicyTrainingSchedule { net_id, steps, lr_scheduler, save_rate } = schedule.clone();
        let schedule = TrainingSchedule {
            net_id: net_id.to_string(),
            eval_scale: 1.0,
            steps,
            wdl_scheduler: ConstantWDL { value: 0.0 },
            lr_scheduler,
            save_rate,
        };

        let PolicyLocalSettings { data_prep_threads, output_directory, batch_queue_size } = *settings;
        let settings = LocalSettings { threads: data_prep_threads, test_set: None, output_directory, batch_queue_size };

        let preparer = self.training_preamble(&schedule, &settings, data_loader);
        let test_preparer = None::<PolicyDataPreparer<Inp, T, B>>;

        self.train_custom(
            &preparer,
            &test_preparer,
            &schedule,
            &settings,
            |superbatch, trainer, schedule, settings| callback(superbatch, trainer, schedule, settings),
        );
    }

    fn training_preamble<LR: LrScheduler>(
        &self,
        schedule: &TrainingSchedule<LR, ConstantWDL>,
        settings: &LocalSettings,
        data_loader: &PolicyDataLoader,
    ) -> PolicyDataPreparer<Inp, T, B> {
        logger::clear_colours();
        println!("{}", logger::ansi("Beginning Training", "34;1"));

        schedule.display();
        settings.display();

        PolicyDataPreparer::new(data_loader.clone(), self.input_getter.clone(), self.move_mapper)
    }
}

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
