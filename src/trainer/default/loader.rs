use bulletformat::BulletFormat;

use super::{inputs::InputType, outputs::OutputBuckets};

use crate::{loader::DataLoader, tensor::Shape, trainer::DataPreparer};

#[derive(Clone)]
pub(crate) struct DefaultDataLoader<I, O, D> {
    input_getter: I,
    output_getter: O,
    wdl: bool,
    scale: f32,
    loader: D,
}

impl<I, O, D> DefaultDataLoader<I, O, D> {
    pub fn new(input_getter: I, output_getter: O, wdl: bool, scale: f32, loader: D) -> Self {
        Self { input_getter, output_getter, wdl, scale, loader }
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
        DefaultDataPreparer::prepare(self.input_getter, self.output_getter, self.wdl, data, threads, blend, self.scale)
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
        wdl: bool,
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

        let output_size = if wdl { 3 } else { 1 };

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
            buckets: SparseInput {
                shape: Shape::new(O::BUCKETS, batch_size),
                max_active: 1,
                value: vec![0; batch_size],
            },
            targets: DenseInput {
                shape: Shape::new(output_size, batch_size),
                value: vec![0.0; output_size * batch_size],
            },
        };

        std::thread::scope(|s| {
            data.chunks(chunk_size)
                .zip(prep.stm.value.chunks_mut(max_active * chunk_size))
                .zip(prep.nstm.value.chunks_mut(max_active * chunk_size))
                .zip(prep.buckets.value.chunks_mut(chunk_size))
                .zip(prep.targets.value.chunks_mut(output_size * chunk_size))
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

                            buckets_chunk[i] = i32::from(out.bucket(pos));

                            if wdl {
                                results_chunk[output_size * i + pos.result_idx()] = 1.0;
                            } else {
                                results_chunk[i] = pos.blended_result(blend, rscale);
                            }
                        }
                    });
                });
        });

        prep
    }
}
