use bullet_trainer::{
    model::{ModelDefinition, ModelInputs, ModelInputsMapper, ModelWeights},
    optimiser::adam::{AdamW, AdamWParams},
    reader::{FixedSizeData, FixedSizeDataReader, ReadMapLoader},
    run::{DefaultDevice, TrainingSchedule, TrainingSteps, train},
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChessBoard {
    pub occ: u64,
    pub pcs: [u8; 16],
    pub score: i16,
    pub result: u8,
    pub ksq: u8,
    pub opp_ksq: u8,
    pub extra: [u8; 3],
}

unsafe impl FixedSizeData for ChessBoard {}

impl ChessBoard {
    pub fn map_pieces(&self, mut f: impl FnMut(u8, u8)) {
        let mut occ = self.occ;
        let mut idx = 0;

        while occ > 0 {
            let square = occ.trailing_zeros() as u8;
            let piece = (self.pcs[idx / 2] >> (4 * (idx % 2))) & 0b1111;

            occ &= occ - 1;
            idx += 1;

            f(piece, square)
        }
    }
}

fn main() {
    let num_inputs = 768;
    let nnz = 32;
    let inputs = ModelInputs::default()
        .add_sparse("stm", (num_inputs, 1), nnz)
        .add_sparse("ntm", (num_inputs, 1), nnz)
        .add_dense("target", (1, 1));

    let mapper = ModelInputsMapper::build(&inputs, move |pnt: &ChessBoard, _, ((stm, ntm), target)| {
        let mut i = 0;

        pnt.map_pieces(|pc, sq| {
            let c = usize::from(pc & 8 > 0);
            let pc = 64 * i32::from(pc & 7);
            let sq = i32::from(sq);

            stm[i] = [0, 384][c] + pc + sq;
            ntm[i] = [384, 0][c] + pc + (sq ^ 56);
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
        let l0 = builder.new_affine("l0", 768, 32);
        let l1 = builder.new_affine("l1", 2 * 32, 1);

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

    let reader = FixedSizeDataReader::new(&["examples/tests/batch.bf"]);
    let loader = ReadMapLoader::new(reader, mapper, 4);

    let schedule = TrainingSchedule {
        steps: TrainingSteps { batch_size: 16_384, batches_per_superbatch: 1, start_superbatch: 1, end_superbatch: 10 },
        log_rate: 128,
        lr_schedule: Box::new(|_| 0.001),
    };

    train(&mut optimiser, schedule, loader, |_, _, _| {}, |_, _| {}).unwrap();
}
