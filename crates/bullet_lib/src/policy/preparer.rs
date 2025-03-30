use bulletformat::ChessBoard;

use crate::{
    default::loader::DataLoader as Blah,
    game::{formats::montyformat::chess::Move, inputs::SparseInputType},
    trainer::DataPreparer,
};

use super::{
    loader::{DecompressedData, PolicyDataLoader},
    move_maps::{ChessMoveMapper, MoveBucket, SquareTransform, MAX_MOVES},
};

#[derive(Clone)]
pub struct PolicyDataPreparer<Inp, T, B> {
    loader: PolicyDataLoader,
    input_getter: Inp,
    move_mapper: ChessMoveMapper<T, B>,
}

impl<Inp, T, B> PolicyDataPreparer<Inp, T, B> {
    pub fn new(loader: PolicyDataLoader, input_getter: Inp, move_mapper: ChessMoveMapper<T, B>) -> Self {
        Self { loader, input_getter, move_mapper }
    }
}

impl<Inp, T, B> DataPreparer for PolicyDataPreparer<Inp, T, B>
where
    Inp: SparseInputType<RequiredDataType = ChessBoard>,
    T: SquareTransform,
    B: MoveBucket,
{
    type DataType = DecompressedData;
    type PreparedData = PolicyPreparedData<Inp>;

    fn get_data_file_paths(&self) -> &[String] {
        self.loader.data_file_paths()
    }

    fn try_count_positions(&self) -> Option<u64> {
        self.loader.count_positions()
    }

    fn load_and_map_batches<F: FnMut(&[Self::DataType]) -> bool>(&self, start_batch: usize, batch_size: usize, f: F) {
        self.loader.map_batches(start_batch, batch_size, f);
    }

    fn prepare(&self, data: &[Self::DataType], threads: usize, _: f32) -> Self::PreparedData {
        PolicyPreparedData::new(data, self.input_getter.clone(), self.move_mapper, threads)
    }
}

pub struct DenseInput {
    pub value: Vec<f32>,
}

#[derive(Clone)]
pub struct SparseInput {
    pub value: Vec<i32>,
    pub max_active: usize,
}

pub struct PolicyPreparedData<I> {
    pub input_getter: I,
    pub batch_size: usize,
    pub stm: SparseInput,
    pub ntm: SparseInput,
    pub mask: SparseInput,
    pub dist: DenseInput,
}

impl<I: SparseInputType<RequiredDataType = ChessBoard>> PolicyPreparedData<I> {
    pub fn new(
        data: &[DecompressedData],
        input_getter: I,
        move_mapper: ChessMoveMapper<impl SquareTransform, impl MoveBucket>,
        threads: usize,
    ) -> Self {
        let batch_size = data.len();
        let chunk_size = batch_size.div_ceil(threads);

        let input_size = input_getter.num_inputs();
        let max_active = input_getter.max_active();
        let num_move_indices = move_mapper.num_move_indices();

        let mut prep = Self {
            input_getter: input_getter.clone(),
            batch_size,
            stm: SparseInput { max_active, value: vec![0; max_active * batch_size] },
            ntm: SparseInput { max_active, value: vec![0; max_active * batch_size] },
            mask: SparseInput { max_active: MAX_MOVES, value: vec![0; MAX_MOVES * batch_size] },
            dist: DenseInput { value: vec![0.0; MAX_MOVES * batch_size] },
        };

        std::thread::scope(|s| {
            for ((((data_chunk, stm_chunk), ntm_chunk), mask_chunk), dist_chunk) in data
                .chunks(chunk_size)
                .zip(prep.stm.value.chunks_mut(max_active * chunk_size))
                .zip(prep.ntm.value.chunks_mut(max_active * chunk_size))
                .zip(prep.mask.value.chunks_mut(MAX_MOVES * chunk_size))
                .zip(prep.dist.value.chunks_mut(MAX_MOVES * chunk_size))
            {
                let input_getter = input_getter.clone();
                s.spawn(move || {
                    for (i, point) in data_chunk.iter().enumerate() {
                        let mask_offset = MAX_MOVES * i;
                        let dist_offset = MAX_MOVES * i;
                        let sparse_offset = max_active * i;

                        let board = ChessBoard::from_raw(point.pos.bbs(), point.pos.stm(), 0, 0.5).unwrap();

                        let mut j = 0;

                        input_getter.map_features(&board, |our, opp| {
                            assert!(our < input_size && opp < input_size, "Input feature index exceeded input size!");

                            stm_chunk[sparse_offset + j] = our as i32;
                            ntm_chunk[sparse_offset + j] = opp as i32;

                            j += 1;
                        });

                        for j in j..max_active {
                            stm_chunk[sparse_offset + j] = -1;
                            ntm_chunk[sparse_offset + j] = -1;
                        }

                        assert!(j <= max_active, "More inputs provided than the specified maximum!");

                        let mut total = 0;
                        let mut distinct = 0;

                        for &(mov, visits) in &point.moves[..point.num] {
                            let idx = move_mapper.map(&point.pos, Move::from(mov));
                            assert!(idx < num_move_indices, "{idx} >= {num_move_indices}");
                            total += visits;

                            mask_chunk[mask_offset + distinct] = idx as i32;
                            dist_chunk[dist_offset + distinct] = f32::from(visits);
                            distinct += 1;
                        }

                        if distinct < MAX_MOVES {
                            mask_chunk[mask_offset + distinct] = -1;
                        }

                        let total = f32::from(total);

                        for idx in 0..distinct {
                            dist_chunk[dist_offset + idx] /= total;
                        }
                    }
                });
            }
        });

        prep
    }
}
