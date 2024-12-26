use bulletformat::ChessBoard;

use super::SparseInputType;

#[derive(Clone, Copy, Debug, Default)]
pub struct Chess768;
impl SparseInputType for Chess768 {
    type RequiredDataType = ChessBoard;

    /// The total number of inputs
    fn num_inputs(&self) -> usize {
        768
    }

    /// The maximum number of active inputs
    fn max_active(&self) -> usize {
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        for (piece, square) in pos.into_iter() {
            let c = usize::from(piece & 8 > 0);
            let pc = 64 * usize::from(piece & 7);
            let sq = usize::from(square);

            let stm = [0, 384][c] + pc + sq;
            let ntm = [384, 0][c] + pc + (sq ^ 56);
            f(stm, ntm)
        }
    }

    /// Shorthand for the input e.g. `768x4`
    fn shorthand(&self) -> String {
        "768".to_string()
    }

    /// Description of the input type
    fn description(&self) -> String {
        "Default psqt chess inputs".to_string()
    }
}
