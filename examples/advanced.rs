use bullet_lib::{
    game::inputs::{Chess768, SparseInputType},
    value::loader::LoadableDataType,
};
use bullet_trainer::model::{DenseInput, ModelInputsBuilder, SparseInput};
use bulletformat::ChessBoard;

fn main() {
    let num_inputs = 768;
    let nnz = 32;
    let wdl = 0.75;

    let feats = Chess768;

    let _inputs = ModelInputsBuilder::default()
        .add_input("stm", SparseInput::new((num_inputs, 1), nnz))
        .add_input("ntm", SparseInput::new((num_inputs, 1), nnz))
        .add_input("target", DenseInput::new((1, 1)))
        .build(move |datapoint: &ChessBoard, _, ((stm, ntm), target)| {
            let mut i = 0;

            feats.map_features(datapoint, |sfeat, nfeat| {
                if sfeat.max(nfeat) >= feats.num_inputs() {
                    panic!("{sfeat} or {nfeat} >= {}", feats.num_inputs());
                }

                stm[i] = sfeat as i32;
                ntm[i] = nfeat as i32;
                i += 1;
            });

            for j in i..nnz {
                stm[j] = -1;
                ntm[j] = -1;
            }

            let score = 1.0 / (1.0 + f32::from(-datapoint.score).exp());
            let result = (f32::from(datapoint.result() as u8) - 1.0) / 2.0;

            target[0] = wdl * result + (1.0 - wdl) * score;
        });
}
