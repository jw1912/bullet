use bullet_core::{graph::builder::Shape, optimiser::Optimiser};
use bulletformat::ChessBoard;

use crate::{default::inputs::SparseInputType, nn::{optimiser::OptimiserType, NetworkBuilder, NetworkBuilderNode}, trainer::save::SavedFormat, ExecutionContext};

use super::{move_maps::{ChessMoveMapper, MoveBucket, SquareTransform, MAX_MOVES}, PolicyTrainer};

#[derive(Default)]
pub struct PolicyTrainerBuilder<O, I, T, B> {
    input_getter: Option<I>,
    move_mapper: Option<ChessMoveMapper<T, B>>,
    saved_format: Option<Vec<SavedFormat>>,
    optimiser: Option<O>,
}

impl<O, I, T, B> PolicyTrainerBuilder<O, I, T, B>
where I: SparseInputType<RequiredDataType = ChessBoard>, T: SquareTransform, B: MoveBucket, O: OptimiserType
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

    pub fn move_mapper(mut self, transform: T, bucket: B) {
        assert!(self.move_mapper.is_none(), "Move mapper already set!");
        self.move_mapper = Some(ChessMoveMapper { transform, bucket });
    }

    pub fn save_format(mut self, fmt: &[SavedFormat]) -> Self {
        assert!(self.saved_format.is_none(), "Save format already set!");
        self.saved_format = Some(fmt.to_vec());
        self
    }

    pub fn build_single_perspective<F>(self, f: F) -> PolicyTrainer<O::Optimiser, I, T, B>
    where F: for<'a> Fn(&'a NetworkBuilder, NetworkBuilderNode<'a>) -> NetworkBuilderNode<'a>
    {
        self.build(|inputs, nnz, builder| {
            let stm = builder.new_sparse_input("stm", Shape::new(inputs, 1), nnz);
            f(builder, stm)
        })
    }

    pub fn build_dual_perspective<F>(self, f: F) -> PolicyTrainer<O::Optimiser, I, T, B>
    where F: for<'a> Fn(&'a NetworkBuilder, NetworkBuilderNode<'a>, NetworkBuilderNode<'a>) -> NetworkBuilderNode<'a>
    {
        self.build(|inputs, nnz, builder| {
            let stm = builder.new_sparse_input("stm", Shape::new(inputs, 1), nnz);
            let ntm = builder.new_sparse_input("nstm", Shape::new(inputs, 1), nnz);

            f(builder, stm, ntm)
        })
    }

    fn build<F>(self, f: F) -> PolicyTrainer<O::Optimiser, I, T, B>
    where F: for<'a> Fn(usize, usize, &'a NetworkBuilder) -> NetworkBuilderNode<'a>
    {
        let input_getter = self.input_getter.expect("Need to set inputs!");
        let move_mapper = self.move_mapper.expect("Need to set move mapper!");

        let builder = NetworkBuilder::default();
        let mask = builder.new_sparse_input("mask", Shape::new(move_mapper.num_move_indices(), 1), MAX_MOVES);
        let dist = builder.new_dense_input("dist", Shape::new(MAX_MOVES, 1));

        let inputs = input_getter.num_inputs();
        let nnz = input_getter.max_active();

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
