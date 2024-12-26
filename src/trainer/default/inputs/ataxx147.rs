use bulletformat::AtaxxBoard;

use super::SparseInputType;

#[derive(Clone, Copy, Debug, Default)]
pub struct Ataxx147;
impl SparseInputType for Ataxx147 {
    type RequiredDataType = AtaxxBoard;

    fn num_inputs(&self) -> usize {
        147
    }

    fn max_active(&self) -> usize {
        49
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        for (piece, square) in <AtaxxBoard as std::iter::IntoIterator>::into_iter(*pos) {
            let pc = usize::from(piece);
            let sq = usize::from(square);

            let stm = 49 * pc + sq;
            let ntm = if pc == 2 { stm } else { 49 * (1 - pc) + sq };
            f(stm, ntm)
        }
    }

    fn shorthand(&self) -> String {
        "147".to_string()
    }

    fn description(&self) -> String {
        "Default ataxx psqt inputs".to_string()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Ataxx98;
impl SparseInputType for Ataxx98 {
    type RequiredDataType = AtaxxBoard;

    fn num_inputs(&self) -> usize {
        98
    }

    fn max_active(&self) -> usize {
        49
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        Ataxx147.map_features(pos, |stm, ntm| {
            if stm < 98 {
                f(stm, ntm)
            }
        });
    }

    fn shorthand(&self) -> String {
        "98".to_string()
    }

    fn description(&self) -> String {
        "Default ataxx psqt inputs without gaps".to_string()
    }
}
