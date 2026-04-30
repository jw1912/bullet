#[derive(Clone, Copy)]
pub struct ChessDataPoint {
    pub stm_bb: u64,
    pub pcs_bb: [u64; 6],
    pub score: i16,
    pub rook_files: [u8; 2],
    pub castle_rights: u8,
    pub halfmove_clock: u8,
    pub actual_stm: bool,
    result: u8,
}

impl ChessDataPoint {
    pub fn map_pieces(&self, mut f: impl FnMut(usize, usize, usize)) {
        for (pc, &pc_bb) in self.pcs_bb.iter().enumerate() {
            let mut stm = pc_bb & self.stm_bb;
            while stm > 0 {
                let sq = stm.trailing_zeros();
                stm &= stm - 1;

                f(0, pc, sq as usize);
            }

            let mut ntm = pc_bb & !self.stm_bb;
            while ntm > 0 {
                let sq = ntm.trailing_zeros();
                ntm &= ntm - 1;

                f(1, pc, sq as usize);
            }
        }
    }

    pub fn result(&self) -> f32 {
        f32::from(self.result) / 2.0
    }
}
