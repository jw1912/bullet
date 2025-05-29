#![deprecated(note = "Merged kings can be done in postprocessing!")]

use bulletformat::ChessBoard;

use super::{get_num_buckets, Chess768, Factorised, Factorises, SparseInputType};

pub type ChessBucketsMergedKingsFactorised = Factorised<ChessBucketsMergedKings, Chess768>;
impl ChessBucketsMergedKingsFactorised {
    pub fn new(buckets: [usize; 64]) -> Self {
        Self::from_parts(ChessBucketsMergedKings::new(buckets), Chess768)
    }
}

pub type ChessBucketsMergedKingsMirroredFactorised = Factorised<ChessBucketsMergedKingsMirrored, Chess768>;
impl ChessBucketsMergedKingsMirroredFactorised {
    pub fn new(buckets: [usize; 32]) -> Self {
        Self::from_parts(ChessBucketsMergedKingsMirrored::new(buckets), Chess768)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ChessBucketsMergedKings {
    buckets: [usize; 64],
    num_buckets: usize,
}

impl ChessBucketsMergedKings {
    pub fn new(buckets: [usize; 64]) -> Self {
        for (sq, bucket) in buckets.iter().enumerate() {
            for (sq2, bucket2) in buckets.iter().enumerate() {
                if bucket == bucket2 {
                    let rank_diff = (sq / 8).abs_diff(sq2 / 8);
                    let file_diff = (sq % 8).abs_diff(sq2 % 8);
                    if rank_diff > 1 || file_diff > 1 {
                        panic!("Invalid bucket layout in ChessBucketsMergedKings!");
                    }
                }
            }
        }

        Self { buckets, num_buckets: get_num_buckets(&buckets) }
    }
}

impl SparseInputType for ChessBucketsMergedKings {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        704 * self.num_buckets
    }

    fn max_active(&self) -> usize {
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let our_bucket = 704 * self.buckets[usize::from(pos.our_ksq())];
        let opp_bucket = 704 * self.buckets[usize::from(pos.opp_ksq())];

        for (piece, square) in pos.into_iter() {
            let c = usize::from(piece & 8 > 0);
            let pc = 64 * usize::from(piece & 7);
            let sq = usize::from(square);

            let offsets = [0, if piece & 7 != 5 { 384 } else { 0 }];
            let stm = offsets[c] + pc + sq;
            let ntm = offsets[1 - c] + pc + (sq ^ 56);
            f(our_bucket + stm, opp_bucket + ntm)
        }
    }

    fn shorthand(&self) -> String {
        format!("704x{}", self.num_buckets)
    }

    fn description(&self) -> String {
        "King bucketed psqt chess inputs, with merged kings".to_string()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ChessBucketsMergedKingsMirrored {
    wrapped: ChessBucketsMergedKings,
}

impl ChessBucketsMergedKingsMirrored {
    pub fn new(buckets: [usize; 32]) -> Self {
        for (sq, bucket) in buckets.iter().enumerate() {
            for (sq2, bucket2) in buckets.iter().enumerate() {
                if bucket == bucket2 {
                    let rank_diff = (sq / 4).abs_diff(sq2 / 4);
                    let file_diff = (sq % 4).abs_diff(sq2 % 4);
                    if rank_diff > 1 || file_diff > 1 {
                        panic!("Invalid bucket layout in ChessBucketsMergedKingsMirrored!");
                    }
                }
            }
        }

        let mut expanded = [0; 64];
        for (idx, elem) in expanded.iter_mut().enumerate() {
            *elem = buckets[(idx / 8) * 4 + [0, 1, 2, 3, 3, 2, 1, 0][idx % 8]];
        }

        Self { wrapped: ChessBucketsMergedKings { buckets: expanded, num_buckets: get_num_buckets(&expanded) } }
    }
}

impl SparseInputType for ChessBucketsMergedKingsMirrored {
    type RequiredDataType = ChessBoard;

    /// The total number of inputs
    fn num_inputs(&self) -> usize {
        self.wrapped.num_inputs()
    }

    /// The maximum number of active inputs
    fn max_active(&self) -> usize {
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let get_flip = |ksq| if ksq % 8 > 3 { 7 } else { 0 };
        let stm_flip = get_flip(pos.our_ksq());
        let ntm_flip = get_flip(pos.opp_ksq());

        self.wrapped.map_features(pos, |stm, ntm| f(stm ^ stm_flip, ntm ^ ntm_flip));
    }

    /// Shorthand for the input e.g. `704x32`
    fn shorthand(&self) -> String {
        format!("{}hm", self.wrapped.shorthand())
    }

    /// Description of the input type
    fn description(&self) -> String {
        "Horizontally mirrored, king bucketed psqt chess inputs, with merged kings".to_string()
    }
}

impl Factorises<ChessBucketsMergedKings> for Chess768 {
    fn derive_feature(&self, inputs: &ChessBucketsMergedKings, feat: usize) -> Option<usize> {
        let mut feature = feat % 704;
        let pc = feature / 64;

        if pc == 5 {
            let bucket = feat / 704;
            let sq = feat % 64;

            if inputs.buckets[sq] != bucket {
                feature += 384;
            }
        };

        Some(feature)
    }
}

impl Factorises<ChessBucketsMergedKingsMirrored> for Chess768 {
    fn derive_feature(&self, inputs: &ChessBucketsMergedKingsMirrored, feat: usize) -> Option<usize> {
        let mut feature = feat % 704;
        let pc = feature / 64;

        if pc == 5 {
            let bucket = feat / 704;
            let sq = feat % 64;

            if sq % 8 > 3 || inputs.wrapped.buckets[sq] != bucket {
                feature += 384;
            }
        };

        Some(feature)
    }
}
