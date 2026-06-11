use bullet_lib::{
    game::inputs::{Chess768, SparseInputType},
    value::loader::LoadableDataType,
};
use bullet_trainer::model::ModelInputsBuilder;
use bulletformat::ChessBoard;

fn main() {
    let wdl = 0.75;

    let feats = Chess768;
    let num_inputs = feats.num_inputs();
    let nnz = feats.max_active();

    let _inputs = ModelInputsBuilder::default()
        .add_sparse_input("stm", (num_inputs, 1), nnz)
        .add_sparse_input("ntm", (num_inputs, 1), nnz)
        .add_dense_input("target", (1, 1))
        .build(move |datapoint: &ChessBoard, _, ((stm, ntm), target)| {
            let mut i = 0;

            feats.map_features(datapoint, |sfeat, nfeat| {
                assert!(sfeat.max(nfeat) < num_inputs);
                stm[i] = sfeat as i32;
                ntm[i] = nfeat as i32;
                i += 1;
            });

            assert!(i <= nnz);

            for j in i..nnz {
                stm[j] = -1;
                ntm[j] = -1;
            }

            let score = 1.0 / (1.0 + f32::from(-datapoint.score).exp());
            let result = (f32::from(datapoint.result() as u8) - 1.0) / 2.0;

            target[0] = wdl * result + (1.0 - wdl) * score;
        });
}
