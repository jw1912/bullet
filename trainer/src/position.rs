use std::str::FromStr;
use data::PackedPosition;

#[derive(Clone, Copy, Debug, Default)]
pub struct Position {
    pub active: [u16; 32],
    pub num: usize,
    pub result: f64,
}

impl From<PackedPosition> for Position {
    fn from(value: PackedPosition) -> Self {
        let mut pos = Self::default();

        for (piece, square) in value {
            pos.active[pos.num] = 64 * piece as u16 + square as u16;
            pos.num += 1;
        }

        pos.result = f64::from(value.res + 1) / 2.;

        pos
    }
}

impl FromStr for Position {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut pos = Position::default();
        let mut row = 7;
        let mut col = 0;
        for ch in s.chars() {
            if ch == '/' {
                row -= 1;
                col = 0;
            } else if ch == ' ' {
                break;
            } else if ('1'..='8').contains(&ch) {
                col += ch.to_digit(10).expect("hard coded") as u16;
            } else if let Some(idx) = "PNBRQKpnbrqk".chars().position(|el| el == ch) {
                let sq = 8 * row + col;
                pos.active[pos.num] = 64 * idx as u16 + sq;
                pos.num += 1;
                col += 1;
            }
        }

        pos.result = match &s[(s.len() - 6)..] {
            "\"1-0\";" | " [1.0]" => 1.0,
            "\"0-1\";" | " [0.0]" => 0.0,
            _ => 0.5,
        };

        Ok(pos)
    }
}

#[test]
fn t() {
    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 [1.0] -80";

    let packed = PackedPosition::from_fen(fen);

    let mut position = Position::from(packed);
    position.active.sort();

    println!("{position:?}");

    position = fen.parse().unwrap();
    position.active.sort();

    println!("{position:?}");
}
