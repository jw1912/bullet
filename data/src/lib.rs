#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PackedPosition {
    occ: u64,
    pcs: [u8; 16],
    pub stm: bool,
    pub res: i8,
    pub score: i16,
}

impl PackedPosition {
    pub fn from_fen(fen: &str) -> Self {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        let board_str = parts[0];
        let stm_str = parts[1];

        let mut pos = Self::default();

        let mut idx = 0;
        for (i, row) in board_str.split('/').rev().enumerate() {
            let mut col = 0;
            for ch in row.chars() {
                if ('1'..='8').contains(&ch) {
                    col += ch.to_digit(10).expect("hard coded") as usize;
                } else if let Some(piece) = "PNBRQKpnbrqk".chars().position(|el| el == ch) {
                    let square = 8 * i + col;
                    pos.occ |= 1 << square;
                    pos.pcs[idx / 2] |= (piece as u8) << (4 * (idx & 1));
                    idx += 1;
                    col += 1;
                }
            }
        }

        pos.stm = stm_str == "b";

        pos.res = match parts[6] {
            "\"1-0\";" | " [1.0]" => 1,
            "\"0-1\";" | " [0.0]" => -1,
            _ => 0,
        };

        if let Some(score) = parts.get(7) {
            pos.score = score.parse::<i16>().unwrap_or(0)
        }

        pos
    }
}

impl IntoIterator for PackedPosition {
    type IntoIter = BoardIter;
    type Item = (u8, u8);
    fn into_iter(self) -> Self::IntoIter {
        BoardIter {
            board: self,
            idx: 0,
        }
    }
}

pub struct BoardIter {
    board: PackedPosition,
    idx: usize,
}

impl Iterator for BoardIter {
    type Item = (u8, u8);
    fn next(&mut self) -> Option<Self::Item> {
        if self.board.occ == 0 {
            return None;
        }

        let square = self.board.occ.trailing_zeros() as u8;
        let piece = (self.board.pcs[self.idx / 2] >> (4 * (self.idx & 1))) & 0b1111;

        self.board.occ &= self.board.occ - 1;
        self.idx += 1;

        Some((piece, square))
    }
}
