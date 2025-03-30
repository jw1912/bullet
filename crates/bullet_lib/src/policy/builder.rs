use std::marker::PhantomData;

use bullet_core::{
    graph::{builder::Shape, ir::args::GraphIRCompileArgs},
    optimiser::Optimiser,
};
use bulletformat::ChessBoard;

use crate::{
    game::inputs::SparseInputType,
    nn::{optimiser::OptimiserType, NetworkBuilder, NetworkBuilderNode},
    trainer::save::SavedFormat,
    ExecutionContext,
};

use super::{
    move_maps::{ChessMoveMapper, MoveBucket, SquareTransform, MAX_MOVES},
    PolicyTrainer,
};

pub struct PolicyTrainerBuilder<O, I, T, B, P> {
    input_getter: Option<I>,
    move_mapper: Option<ChessMoveMapper<T, B>>,
    saved_format: Option<Vec<SavedFormat>>,
    optimiser: Option<O>,
    compile_args: Option<GraphIRCompileArgs>,
    perspective: PhantomData<P>,
}

impl<O, I, T, B> Default for PolicyTrainerBuilder<O, I, T, B, SinglePerspective> {
    fn default() -> Self {
        Self {
            input_getter: None,
            move_mapper: None,
            saved_format: None,
            optimiser: None,
            compile_args: None,
            perspective: PhantomData,
        }
    }
}

impl<O, I, T, B, P> PolicyTrainerBuilder<O, I, T, B, P>
where
    I: SparseInputType<RequiredDataType = ChessBoard>,
    T: SquareTransform,
    B: MoveBucket,
    O: OptimiserType,
{
    pub fn inputs(mut self, inputs: I) -> Self {
        assert!(self.input_getter.is_none(), "Inputs already set!");
        self.input_getter = Some(inputs);
        self
    }

    pub fn optimiser(mut self, optimiser: O) -> Self {
        assert!(self.optimiser.is_none(), "Optimiser already set!");
        self.optimiser = Some(optimiser);
        self
    }

    pub fn move_mapper(mut self, transform: T, bucket: B) -> Self {
        assert!(self.move_mapper.is_none(), "Move mapper already set!");
        self.move_mapper = Some(ChessMoveMapper { transform, bucket });
        self
    }

    pub fn compile_args(mut self, args: GraphIRCompileArgs) -> Self {
        self.compile_args = Some(args);
        self
    }

    pub fn save_format(mut self, fmt: &[SavedFormat]) -> Self {
        assert!(self.saved_format.is_none(), "Save format already set!");
        self.saved_format = Some(fmt.to_vec());
        self
    }

    fn build_internal<F>(self, f: F) -> PolicyTrainer<O::Optimiser, I, T, B>
    where
        F: for<'a> Fn(usize, usize, &'a NetworkBuilder) -> NetworkBuilderNode<'a>,
    {
        let input_getter = self.input_getter.expect("Need to set inputs!");
        let move_mapper = self.move_mapper.expect("Need to set move mapper!");

        let inputs = input_getter.num_inputs();
        let nnz = input_getter.max_active();
        let outputs = move_mapper.num_move_indices();

        let mut builder = NetworkBuilder::default();

        if let Some(args) = self.compile_args {
            builder.set_compile_args(args);
        }

        let mask = builder.new_sparse_input("mask", Shape::new(outputs, 1), MAX_MOVES);
        let dist = builder.new_dense_input("dist", Shape::new(MAX_MOVES, 1));

        let out = f(inputs, nnz, &builder);
        out.masked_softmax_crossentropy_loss(dist, mask);
        let graph = builder.build(ExecutionContext::default());

        PolicyTrainer {
            optimiser: Optimiser::new(graph, Default::default()).unwrap(),
            input_getter,
            move_mapper,
            saved_format: self.saved_format,
        }
    }
}

pub struct SinglePerspective;
pub struct DualPerspective;

impl<O, I, T, B> PolicyTrainerBuilder<O, I, T, B, SinglePerspective>
where
    I: SparseInputType<RequiredDataType = ChessBoard>,
    T: SquareTransform,
    B: MoveBucket,
    O: OptimiserType,
{
    pub fn single_perspective(self) -> Self {
        self
    }

    pub fn dual_perspective(self) -> PolicyTrainerBuilder<O, I, T, B, DualPerspective> {
        PolicyTrainerBuilder {
            input_getter: self.input_getter,
            move_mapper: self.move_mapper,
            saved_format: self.saved_format,
            optimiser: self.optimiser,
            compile_args: self.compile_args,
            perspective: PhantomData,
        }
    }

    pub fn build<F>(self, f: F) -> PolicyTrainer<O::Optimiser, I, T, B>
    where
        F: for<'a> Fn(&'a NetworkBuilder, NetworkBuilderNode<'a>) -> NetworkBuilderNode<'a>,
    {
        self.build_internal(|inputs, nnz, builder| {
            let stm = builder.new_sparse_input("stm", Shape::new(inputs, 1), nnz);
            f(builder, stm)
        })
    }
}

impl<O, I, T, B> PolicyTrainerBuilder<O, I, T, B, DualPerspective>
where
    I: SparseInputType<RequiredDataType = ChessBoard>,
    T: SquareTransform,
    B: MoveBucket,
    O: OptimiserType,
{
    pub fn build<F>(self, f: F) -> PolicyTrainer<O::Optimiser, I, T, B>
    where
        F: for<'a> Fn(&'a NetworkBuilder, NetworkBuilderNode<'a>, NetworkBuilderNode<'a>) -> NetworkBuilderNode<'a>,
    {
        self.build_internal(|inputs, nnz, builder| {
            let stm = builder.new_sparse_input("stm", Shape::new(inputs, 1), nnz);
            let ntm = builder.new_sparse_input("nstm", Shape::new(inputs, 1), nnz);
            f(builder, stm, ntm)
        })
    }
}
