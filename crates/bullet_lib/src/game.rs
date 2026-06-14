/// Contains the `InputType` trait for implementing custom input types,
/// as well as several premade input formats that are commonly used.
pub mod inputs;
/// Contains the `OutputBuckets` trait for implementing custom output bucket types,
/// as well as several premade output buckets that are commonly used.
pub mod outputs;
/// Implementations of readers for different data formats
pub mod readers;

/// Contains data formats
pub mod formats {
    pub use bulletformat;
    pub use montyformat;
    pub use sfbinpack;
    pub use viriformat;

    use crate::value::loader::GameResult;

    #[derive(Clone, Copy)]
    pub struct ChessDatapoint {
        pub bbs: [u64; 7],
        pub stm: bool,
        pub score: i16,
        pub result: GameResult,
        pub fullm: u16,
        pub halfm: u8,
    }

    impl ChessDatapoint {
        pub fn map_pieces(&self, mut f: impl FnMut(u8, u8)) {
            let white = self.bbs[0];
            let black = self.bbs.iter().fold(0, |x, y| x ^ y);

            let side = usize::from(self.stm);
            let mut stm = [white, black][side];
            let mut ntm = [black, white][side];

            if self.stm {
                stm = stm.swap_bytes();
                ntm = ntm.swap_bytes();
            }

            for (pc, mut bb) in self.bbs.iter().skip(1).cloned().enumerate() {
                if self.stm {
                    bb = bb.swap_bytes();
                }

                let mut stm_bb = bb & stm;
                while stm_bb > 0 {
                    let sq = stm_bb.trailing_zeros();
                    f(pc as u8, sq as u8);
                    stm_bb &= stm_bb - 1;
                }

                let mut ntm_bb = bb & ntm;
                while ntm_bb > 0 {
                    let sq = ntm_bb.trailing_zeros();
                    f(6 + pc as u8, sq as u8);
                    ntm_bb &= ntm_bb - 1;
                }
            }
        }

        pub fn score(&self) -> i16 {
            if self.stm { -self.score } else { self.score }
        }

        pub fn result(&self) -> f32 {
            let result = match self.result {
                GameResult::Loss => 0.0,
                GameResult::Draw => 0.5,
                GameResult::Win => 1.0,
            };

            if self.stm { 1.0 - result } else { result }
        }
    }
}
