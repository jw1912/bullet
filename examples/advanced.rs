use bullet_nn::{
    games::chess::ChessDataPoint,
    train::inputs::{DenseInput, SparseInput, TrainerInputsBuilder},
};

fn main() {
    let num_inputs = 768;
    let nnz = 32;
    let wdl = 0.75;

    let _inputs = TrainerInputsBuilder::default()
        .add_input(SparseInput::new((num_inputs, 1), nnz))
        .add_input(SparseInput::new((num_inputs, 1), nnz))
        .add_input(DenseInput::new((1, 1)))
        .build(move |datapoint: &ChessDataPoint, _batch, ((stm, ntm), target)| {
            let mut i = 0;

            datapoint.map_pieces(|c, pc, sq| {
                stm[i] = ([0, 384][c] + 64 * pc + sq) as i32;
                ntm[i] = ([384, 0][c] + 64 * pc + (sq ^ 56)) as i32;
                i += 1;
            });

            for j in i..nnz {
                stm[j] = -1;
                ntm[j] = -1;
            }

            let score = 1.0 / (1.0 + f32::from(-datapoint.score).exp());

            target[0] = wdl * datapoint.result() + (1.0 - wdl) * score;
        });
}
