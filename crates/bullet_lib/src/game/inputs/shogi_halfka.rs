//! ShogiHalfKA_hm - 将棋用 HalfKA_hm 特徴量
//!
//! Half-Mirror King + All pieces (coalesced) 特徴量。
//!
//! - キングバケット: 45バケット (Half-Mirror: 9段 × 5筋)
//! - 入力次元: 73,305 (45 × 1629)
//! - 最大アクティブ特徴: 40

use super::SparseInputType;
use crate::shogi::{HALFKA_HM_DIMENSIONS, MAX_ACTIVE_FEATURES, PackedSfenValue, ShogiBoard};

/// ShogiHalfKA_hm 特徴量
///
/// YaneuraOu / nnue-pytorch 互換の HalfKA_hm 特徴量。
/// coalesce 済みモデル専用（Factorization の重みは Base 側に畳み込み済み）。
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiHalfKA_hm;

impl SparseInputType for ShogiHalfKA_hm {
    /// 学習データの型: PackedSfenValue (40バイト)
    type RequiredDataType = PackedSfenValue;

    /// 特徴量の総次元数: 45 × 1629 = 73,305
    fn num_inputs(&self) -> usize {
        HALFKA_HM_DIMENSIONS
    }

    /// 同時にアクティブになる最大特徴数: 40
    ///
    /// 将棋の合法局面では駒の総数は40個固定:
    /// - 盤上駒（王含む）+ 手駒 = 40
    fn max_active(&self) -> usize {
        MAX_ACTIVE_FEATURES
    }

    /// 特徴量インデックスを列挙
    ///
    /// PackedSfenValue をデコードして ShogiBoard を作成し、
    /// 各駒について (stm_index, nstm_index) を生成。
    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        board.map_features(f);
    }

    /// 短縮名
    fn shorthand(&self) -> String {
        "shogi-73305x45hm".to_string()
    }

    /// 説明
    fn description(&self) -> String {
        "Shogi HalfKA_hm: 45 king buckets (half-mirrored), 1629 piece inputs".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensions() {
        let input = ShogiHalfKA_hm;
        assert_eq!(input.num_inputs(), 73_305);
        assert_eq!(input.max_active(), 40);
    }

    #[test]
    fn test_shorthand() {
        let input = ShogiHalfKA_hm;
        assert_eq!(input.shorthand(), "shogi-73305x45hm");
    }
}
