use bulletformat::BulletFormat;

use crate::{inputs::InputType, outputs::OutputBuckets};

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Feat {
    our: i32,
    opp: i32,
}

impl Feat {
    pub fn new(our: i32, opp: i32) -> Self {
        Self { our, opp }
    }

    pub fn our(&self) -> i32 {
        self.our
    }

    pub fn opp(&self) -> i32 {
        self.opp
    }
}

pub struct GpuDataLoader<I: InputType, O: OutputBuckets<I::RequiredDataType>> {
    inputs: Vec<Feat>,
    results: Vec<f32>,
    buckets: Vec<u8>,
    input_getter: I,
    output_getter: O,
}

impl<I, O: OutputBuckets<I::RequiredDataType>> GpuDataLoader<I, O>
where
    I: InputType + Send + Sync,
    I::RequiredDataType: Send + Sync + Copy,
{
    pub fn new(input_getter: I, output_getter: O) -> Self {
        Self { inputs: Vec::new(), results: Vec::new(), buckets: Vec::new(), input_getter, output_getter }
    }

    pub fn inputs(&self) -> &Vec<Feat> {
        &self.inputs
    }

    pub fn results(&self) -> &Vec<f32> {
        &self.results
    }

    pub fn buckets(&self) -> &Vec<u8> {
        &self.buckets
    }

    pub fn load(&mut self, data: &[I::RequiredDataType], threads: usize, blend: f32, rscale: f32) {
        let batch_size = data.len();
        let max_features = self.input_getter.max_active_inputs();
        let chunk_size = (batch_size + threads - 1) / threads;

        self.inputs = vec![Feat { our: 0, opp: 0 }; max_features * batch_size];
        self.results = vec![0.0; batch_size];
        self.buckets = vec![0; batch_size];

        std::thread::scope(move |s| {
            data.chunks(chunk_size)
                .zip(self.inputs.chunks_mut(max_features * chunk_size))
                .zip(self.results.chunks_mut(chunk_size))
                .zip(self.buckets.chunks_mut(chunk_size))
                .for_each(|(((data_chunk, input_chunk), results_chunk), buckets_chunk)| {
                    let inp = &self.input_getter;
                    let out = &self.output_getter;
                    s.spawn(move || {
                        let chunk_len = data_chunk.len();

                        for i in 0..chunk_len {
                            let pos = &data_chunk[i];
                            let mut j = 0;
                            let offset = max_features * i;

                            for (our, opp) in inp.feature_iter(pos) {
                                input_chunk[offset + j] = Feat::new(our as i32, opp as i32);
                                j += 1;
                            }

                            if j < max_features {
                                input_chunk[offset + j] = Feat::new(-1, -1);
                            }

                            results_chunk[i] = pos.blended_result(blend, rscale);
                            buckets_chunk[i] = out.bucket(pos);
                        }
                    });
                });
        });
    }
}
