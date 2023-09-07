use crate::util::sigmoid;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MarlinFormat {
    occ: u64,
    pcs: [u8; 16],
    stm_enp: u8,
    hfm: u8,
    fmc: u16,
    score: i16,
    result: u8,
    extra: u8,
}

impl MarlinFormat {
    pub fn occ(&self) -> u64 {
        self.occ
    }

    pub fn pcs(&self) -> [u8; 16] {
        self.pcs
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

    pub fn blended_result(&self, blend: f32, scale: f32) -> f32 {
        let (wdl, score) = if self.stm() == 1 {
            (1.0 - self.result(), -self.score)
        } else {
            (self.result(), self.score)
        };
        blend * wdl + (1. - blend) * sigmoid(f32::from(score), scale)
    }

    pub fn stm(&self) -> usize {
        usize::from(self.stm_enp >> 7)
    }

    pub fn from_epd(fen: &str) -> Result<Self, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        let board_str = parts[0];
        let stm_str = parts[1];

        let mut occ = 0;
        let mut pcs = [0; 16];

        let mut idx = 0;
        for (i, row) in board_str.split('/').rev().enumerate() {
            let mut col = 0;
            for ch in row.chars() {
                if ('1'..='8').contains(&ch) {
                    col += ch.to_digit(10).expect("hard coded") as usize;
                } else if let Some(piece) = "PNBRQKpnbrqk".chars().position(|el| el == ch) {
                    let square = 8 * i + col;
                    occ |= 1 << square;
                    let pc = (piece % 6) | (piece / 6) << 3;
                    pcs[idx / 2] |= (pc as u8) << (4 * (idx & 1));
                    idx += 1;
                    col += 1;
                }
            }
        }

        // don't currently worry about en passant square
        let stm_enp = u8::from(stm_str == "b") << 7;

        let hfm = parts[4].parse().unwrap_or(0);

        let fmc = parts[5].parse().unwrap_or(1);

        let score = parts[6].parse::<i16>().unwrap_or(0);

        let result = match parts[7] {
            "[1.0]" => 2,
            "[0.5]" => 1,
            "[0.0]" => 0,
            _ => {
                println!("{fen}");
                return Err(String::from("Bad game result!"));
            }
        };

        Ok(Self {occ, pcs, stm_enp, hfm, fmc, score, result, extra: 0})
    }

}

impl IntoIterator for MarlinFormat {
    type Item = (u8, u8, u8);
    type IntoIter = MarlinFormatIter;
    fn into_iter(self) -> Self::IntoIter {
        MarlinFormatIter {
            board: self,
            idx: 0,
        }
    }
}

pub struct MarlinFormatIter {
    board: MarlinFormat,
    idx: usize,
}

impl Iterator for MarlinFormatIter {
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