use crate::util::sigmoid;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Position {
    occ: u64,
    pcs: [u8; 16],
    stm_enp: u8,
    hfm: u8,
    fmc: u16,
    score: i16,
    result: u8,
    extra: u8,
}

impl Position {
    #[allow(clippy::too_many_arguments)]
    pub fn new(occ: u64, pcs: [u8; 16], stm_enp: u8, hfm: u8, fmc: u16, score: i16, result: u8, extra: u8) -> Self {
        Self { occ, pcs, stm_enp, hfm, fmc, score, result, extra }
    }

    pub fn score(&self) -> i16 {
        self.score
    }

    pub fn result(&self) -> f32 {
        f32::from(self.result) / 2.
    }

    pub fn result_idx(&self) -> usize {
        usize::from(self.result)
    }

    pub fn blended_result(&self, blend: f32, stm: usize, scale: f32) -> f32 {
        let (wdl, score) = if stm == 1 {
            (1.0 - self.result(), -self.score)
        } else {
            (self.result(), self.score)
        };
        blend * wdl + (1. - blend) * sigmoid(f32::from(score), scale)
    }

    pub fn stm(&self) -> usize {
        usize::from(self.stm_enp >> 7)
    }
}

#[derive(Default)]
pub struct Features {
    features: [(usize, usize); 32],
    len: usize,
}

impl Features {
    pub fn push(&mut self, wfeat: usize, bfeat: usize) {
        self.features[self.len] = (wfeat, bfeat);
        self.len += 1;
    }
}

impl IntoIterator for Features {
    type Item = (usize, usize);
    type IntoIter = std::iter::Take<std::array::IntoIter<(usize, usize), 32>>;
    fn into_iter(self) -> Self::IntoIter {
        self.features.into_iter().take(self.len)
    }
}

impl IntoIterator for Position {
    type Item = (u8, u8, u8);
    type IntoIter = BoardIter;
    fn into_iter(self) -> Self::IntoIter {
        BoardIter {
            board: self,
            idx: 0,
        }
    }
}

pub struct BoardIter {
    board: Position,
    idx: usize,
}

impl Iterator for BoardIter {
    type Item = (u8, u8, u8);
    fn next(&mut self) -> Option<Self::Item> {
        if self.board.occ == 0 {
            return None;
        }

        let square = self.board.occ.trailing_zeros() as u8;
        let coloured_piece = (self.board.pcs[self.idx / 2] >> (4 * (self.idx & 1))) & 0b1111;

        let mut piece = coloured_piece & 0b111;
        if piece == 6 {
            piece = 3;
        }

        let colour = coloured_piece >> 3;

        self.board.occ &= self.board.occ - 1;
        self.idx += 1;

        Some((colour, piece, square))
    }
}
