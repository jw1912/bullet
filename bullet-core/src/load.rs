use bulletformat::BulletFormat;

use crate::inputs::InputType;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Feat {
    our: u16,
    opp: u16,
}

impl Feat {
    pub fn new(our: u16, opp: u16) -> Self {
        Self { our, opp }
    }
}

pub struct GpuDataLoader<I: InputType> {
    inputs: Vec<Feat>,
    results: Vec<f32>,
    input_getter: I,
}

impl<I> GpuDataLoader<I>
where
    I: InputType + Send + Sync,
    I::RequiredDataType: Send + Sync + Copy
{
    pub fn new(input_getter: I) -> Self {
        Self { inputs: Vec::new(), results: Vec::new(), input_getter }
    }

    pub fn inputs(&self) -> &Vec<Feat> {
        &self.inputs
    }

    pub fn results(&self) -> &Vec<f32> {
        &self.results
    }

    pub fn load(&mut self, data: &[I::RequiredDataType], threads: usize, blend: f32, rscale: f32) {
        let batch_size = data.len();
        let chunk_size = (batch_size + threads - 1) / threads;

        self.inputs = vec![Feat { our: 0, opp: 0 }; 32 * batch_size];
        self.results = vec![0.0; batch_size];

        std::thread::scope(move |s| {
            data.chunks(chunk_size)
                .zip(self.inputs.chunks_mut(32 * chunk_size))
                .zip(self.results.chunks_mut(chunk_size))
                .for_each(|((data_chunk, input_chunk), results_chunk)| {
                    let inp = &self.input_getter;
                    s.spawn(move || {
                        let chunk_len = data_chunk.len();

                        for i in 0..chunk_len {
                            let pos = &data_chunk[i];
                            let mut j = 0;
                            let offset = 32 * i;

                            for feat in pos.into_iter() {
                                let (our, opp) = inp.get_feature_indices(feat);
                                input_chunk[offset + j] = Feat { our: our as u16, opp: opp as u16 };
                                j += 1;
                            }

                            if j < I::RequiredDataType::MAX_FEATURES {
                                input_chunk[offset + j] = Feat { our: u16::MAX, opp: u16::MAX };
                            }

                            results_chunk[i] = pos.blended_result(blend, rscale);
                        }
                    });
                });
        });
    }
}
