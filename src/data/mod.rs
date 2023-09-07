pub mod chess;
pub mod marlinformat;

pub use chess::ChessBoard;

trait DataType {
    type FeatureType;
}

#[derive(Default)]
pub struct Features {
    features: [(usize, usize); 32],
    len: usize,
    consumed: usize
}

impl Features {
    pub fn push(&mut self, wfeat: usize, bfeat: usize) {
        self.features[self.len] = (wfeat, bfeat);
        self.len += 1;
    }
}

impl Iterator for Features {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        if self.consumed == self.len {
            return None;
        }

        let ret = self.features[self.consumed];

        self.consumed += 1;

        Some(ret)
    }
}

#[cfg(test)]
mod test {
    use super::{*, marlinformat::MarlinFormat};

    #[test]
    fn working_conversion() {
        let board = ChessBoard::from_epd("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 5 [1.0]").unwrap();
        let mf = MarlinFormat::from_epd("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 5 [1.0]").unwrap();
        let mf_board = ChessBoard::from_marlinformat(&mf);

        println!("{mf:?}");
        println!("{board:?}");
        println!("{mf_board:?}");

        assert_eq!(board, mf_board);
    }
}