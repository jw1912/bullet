use bullet_trainer::{
    model::{ModelDefinition, ModelInputs, ModelInputsMapper, ModelWeights},
    optimiser::adam::{AdamW, AdamWParams},
    reader::{FixedSizeData, FixedSizeDataReader, ReadMapLoader},
    run::{DefaultDevice, TrainingSchedule, TrainingSteps, train},
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AtaxxBoard {
    pub bbs: [u64; 3],
    pub score: i16,
    pub result: u8,
    pub origstm: u8,
    pub fullm: u16,
    pub halfm: u8,
    pub extra: u8,
}

unsafe impl FixedSizeData for AtaxxBoard {}

impl AtaxxBoard {
    pub fn map_pieces(&self, mut f: impl FnMut(i32, i32)) {
        for (stage, mut occ) in self.bbs.iter().cloned().enumerate() {
            while occ > 0 {
                f(stage as i32, occ.trailing_zeros() as i32);
                occ &= occ - 1;
            }
        }
    }
}

fn main() {
    let hl_size = 256;
    let num_inputs = 147;
    let nnz = 49;

    let inputs = ModelInputs::default()
        .add_sparse("stm", (num_inputs, 1), nnz)
        .add_sparse("ntm", (num_inputs, 1), nnz)
        .add_dense("target", (1, 1));

    let mapper = ModelInputsMapper::build(&inputs, move |pnt: &AtaxxBoard, _, ((stm, ntm), target)| {
        let mut i = 0;

        pnt.map_pieces(|pc, sq| {
            stm[i] = 49 * pc + sq;
            ntm[i] = 49 * [1, 0, 2][pc as usize] + sq;
            i += 1;
        });

        if i < nnz {
            stm[i] = -1;
            ntm[i] = -1;
        }

        let lambda = 0.2;
        let score = 1.0 / (1.0 + (-f32::from(pnt.score) / 400.0).exp());
        let result = f32::from(pnt.result) / 2.0;
        target[0] = lambda * result + (1.0 - lambda) * score;
    });

    let defn = ModelDefinition::build(&inputs, |builder, ((stm, ntm), target)| {
        let l0 = builder.new_affine("l0", num_inputs, hl_size);
        let l1 = builder.new_affine("l1", 2 * hl_size, 1);

        let stm_hidden = l0.forward(stm).screlu();
        let ntm_hidden = l0.forward(ntm).screlu();
        let hidden_layer = stm_hidden.concat(ntm_hidden);
        let output = l1.forward(hidden_layer);

        let loss = output.sigmoid().squared_error(target).reduce_sum_batch();

        (Some(loss), vec![("output".to_string(), output)])
    });

    let weights = ModelWeights::new(&defn, 198273612);
    let device = DefaultDevice::new(0).unwrap();
    let params = AdamWParams { decay: 0.01, beta1: 0.9, beta2: 0.999, min_weight: -1.98, max_weight: 1.98 };

    let mut optimiser = AdamW::new(defn, weights, device, params).unwrap();

    let reader = FixedSizeDataReader::new(&["data/ataxx.data"]);
    let loader = ReadMapLoader::new(reader, mapper, 4);

    let schedule = TrainingSchedule {
        steps: TrainingSteps { batch_size: 16_384, batches_per_superbatch: 1, start_superbatch: 1, end_superbatch: 10 },
        log_rate: 128,
        lr_schedule: Box::new(|_| 0.001),
    };

    train(&mut optimiser, schedule, loader, |_, _, _| {}, |_, _| {}).unwrap();
}
