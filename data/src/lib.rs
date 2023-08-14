#[derive(Clone, Copy, Default)]
pub struct PackedPosition {
    pub board: Board,
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
        let mut builder = BoardBuilder::default();

        let mut row = 7i16;
        let mut col = 0i16;
        for ch in board_str.chars() {
            if ch == '/' {
                row -= 1;
                col = 0;
            } else if ch == ' ' {
                break;
            } else if ('1'..='8').contains(&ch) {
                col += ch.to_digit(10).expect("hard coded") as i16;
            } else if let Some(piece) = "PNBRQKpnbrqk".chars().position(|el| el == ch) {
                let square = 8 * row + col;
                builder.add(piece as u8, square as u8);
                col += 1;
            }
        }

        pos.board = builder.get_board();

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

#[derive(Clone, Copy, Default)]
pub struct Board {
    occ: u64,
    pcs: [u8; 16],
}

impl IntoIterator for Board {
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
    board: Board,
    idx: usize,
}

impl Iterator for BoardIter {
    type Item = (u8, u8);
    fn next(&mut self) -> Option<Self::Item> {
        if self.board.occ == 0 {
            return None;
        }

        let square = self.board.occ.trailing_zeros() as u8;
        let piece = self.board.pcs[self.idx / 2] >> (4 * (self.idx & 1));

        self.board.occ &= self.board.occ - 1;
        self.idx += 1;

        Some((piece, square))
    }
}

#[derive(Default)]
struct BoardBuilder {
    internal: [(u8, u8); 32],
    len: usize,
}

impl BoardBuilder {
    fn add(&mut self, piece: u8, square: u8) {
        self.internal[self.len] = (piece, square);
        self.len += 1;
    }

    fn pop(&mut self) -> Option<(u8, u8)> {
        if self.len == 0 {
            return None;
        }

        self.len -= 1;
        Some(self.internal[self.len])
    }

    fn get_board(&mut self) -> Board {
        let mut board = Board::default();
        let mut idx = 0;

        while let Some((piece, square)) = self.pop() {
            board.occ |= 1 << square;
            board.pcs[idx / 2] |= piece << (4 * (idx & 1));
            idx += 1;
        }

        board
    }
}
