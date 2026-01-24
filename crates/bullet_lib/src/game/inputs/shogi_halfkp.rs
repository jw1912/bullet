//! ShogiHalfKP - 将棋用 HalfKP 特徴量
//!
//! King + Piece 特徴量（王は特徴量に含めない）。
//! nnue-pytorch の HalfKP 実装に準拠。
//!
//! - キングバケット: 81バケット (全マス)
//! - 入力次元: 125,388 (81 × 1548)
//! - 最大アクティブ特徴: 38 (王2枚を除く)

use super::SparseInputType;
use crate::shogi::{
    BonaPiece, PackedSfenValue, ShogiBoard,
    bona_piece::FE_OLD_END,
    types::{BOARD_PIECE_TYPES, Color, HAND_PIECE_TYPES, Piece},
};

// =============================================================================
// 定数
// =============================================================================

/// キングバケット数 (全81マス)
pub const NUM_KING_SQ: usize = 81;

/// 駒入力数 (fe_end = 1548、王を除く)
pub const FE_END: usize = FE_OLD_END; // 1548

/// HalfKP の総入力次元
pub const HALFKP_DIMENSIONS: usize = NUM_KING_SQ * FE_END; // 125,388

/// 最大アクティブ特徴数 (王2枚を除く38駒)
pub const MAX_ACTIVE_FEATURES: usize = 38;

// =============================================================================
// ShogiHalfKP 特徴量型
// =============================================================================

/// ShogiHalfKP 特徴量
///
/// nnue-pytorch / YaneuraOu 互換の HalfKP 特徴量。
/// 王は特徴量に含めない（HalfKA_hm との違い）。
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiHalfKP;

impl SparseInputType for ShogiHalfKP {
    /// 学習データの型: PackedSfenValue (40バイト)
    type RequiredDataType = PackedSfenValue;

    /// 特徴量の総次元数: 81 × 1548 = 125,388
    fn num_inputs(&self) -> usize {
        HALFKP_DIMENSIONS
    }

    /// 同時にアクティブになる最大特徴数: 38
    ///
    /// 将棋の合法局面では王を除く駒は最大38個:
    /// - 盤上駒（王除く）+ 手駒 = 38
    fn max_active(&self) -> usize {
        MAX_ACTIVE_FEATURES
    }

    /// 特徴量インデックスを列挙
    ///
    /// PackedSfenValue をデコードして ShogiBoard を作成し、
    /// 各駒について (stm_index, nstm_index) を生成。
    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        map_halfkp_features(&board, f);
    }

    /// 短縮名
    fn shorthand(&self) -> String {
        "shogi-125388x81".to_string()
    }

    /// 説明
    fn description(&self) -> String {
        "Shogi HalfKP: 81 king squares, 1548 piece inputs (no kings in features)".to_string()
    }
}

// =============================================================================
// HalfKP 特徴量計算
// =============================================================================

/// HalfKP 特徴量インデックスを列挙
///
/// stm (side-to-move) 視点と nstm (not-side-to-move) 視点の両方を返す。
/// 片玉・詰将棋データ（玉位置が SQ_NB=81）の場合は何もしない。
fn map_halfkp_features<F: FnMut(usize, usize)>(board: &ShogiBoard, mut f: F) {
    // STM と NSTM の視点
    let stm = board.side_to_move;
    let nstm = stm.opponent();

    // 玉位置の妥当性チェック（SQ_NB=81 は「玉なし」を意味する）
    let stm_king_sq = board.king_square(stm);
    let nstm_king_sq = board.king_square(nstm);
    if !stm_king_sq.is_valid() || !nstm_king_sq.is_valid() {
        // 片玉/詰将棋データはスキップ
        return;
    }

    // 視点に応じた玉位置（後手視点では反転）
    let stm_ksq = if stm == Color::Black { stm_king_sq.index() } else { stm_king_sq.inverse().index() };

    let nstm_ksq = if nstm == Color::Black { nstm_king_sq.index() } else { nstm_king_sq.inverse().index() };

    // 盤上の駒（王以外）
    for &pt in &BOARD_PIECE_TYPES {
        for color in [Color::Black, Color::White] {
            for sq in board.pieces(color, pt) {
                let piece = Piece::new(color, pt);

                // STM 視点での BonaPiece
                let stm_bp = BonaPiece::from_piece_square(piece, sq, stm);
                if stm_bp == BonaPiece::ZERO {
                    continue;
                }
                let stm_idx = halfkp_index(stm_ksq, stm_bp.value() as usize);

                // NSTM 視点での BonaPiece
                let nstm_bp = BonaPiece::from_piece_square(piece, sq, nstm);
                let nstm_idx = halfkp_index(nstm_ksq, nstm_bp.value() as usize);

                f(stm_idx, nstm_idx);
            }
        }
    }

    // 注意: HalfKP では王は特徴量に含めない（HalfKA_hm との違い）

    // 手駒の特徴量
    for owner in [Color::Black, Color::White] {
        for &pt in &HAND_PIECE_TYPES {
            let count = board.hand(owner).count(pt);
            if count == 0 {
                continue;
            }

            // 各枚数分の特徴量を追加
            for i in 1..=count {
                // STM 視点
                let stm_bp = BonaPiece::from_hand_piece(stm, owner, pt, i);
                if stm_bp == BonaPiece::ZERO {
                    continue;
                }
                let stm_idx = halfkp_index(stm_ksq, stm_bp.value() as usize);

                // NSTM 視点
                let nstm_bp = BonaPiece::from_hand_piece(nstm, owner, pt, i);
                let nstm_idx = halfkp_index(nstm_ksq, nstm_bp.value() as usize);

                f(stm_idx, nstm_idx);
            }
        }
    }
}

/// HalfKP の特徴インデックスを計算
///
/// feature_index = king_sq * FE_END + bonapiece
#[inline]
fn halfkp_index(king_sq: usize, bonapiece: usize) -> usize {
    king_sq * FE_END + bonapiece
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use crate::shogi::{PieceType, Square};

    use super::*;

    #[test]
    fn test_dimensions() {
        let input = ShogiHalfKP;
        assert_eq!(input.num_inputs(), 125_388);
        assert_eq!(input.max_active(), 38);
    }

    #[test]
    fn test_shorthand() {
        let input = ShogiHalfKP;
        assert_eq!(input.shorthand(), "shogi-125388x81");
    }

    #[test]
    fn test_halfkp_index() {
        // king_sq=0, bp=0 → 0
        assert_eq!(halfkp_index(0, 0), 0);

        // king_sq=1, bp=0 → 1548
        assert_eq!(halfkp_index(1, 0), FE_END);

        // king_sq=80, bp=1547 → 80*1548 + 1547 = 125,387
        assert_eq!(halfkp_index(80, 1547), 125_387);
    }

    #[test]
    fn test_map_features_count() {
        // ダミーの局面を作成（手動で設定）
        let mut board = ShogiBoard::default();
        board.side_to_move = Color::Black;

        // 玉を配置
        board.black_king_sq = Square::new(4, 8); // 5九
        board.white_king_sq = Square::new(4, 0); // 5一
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        // 歩を9枚ずつ配置
        for file in 0..9 {
            board.board[Square::new(file, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
            board.board[Square::new(file, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
        }

        let mut count = 0;
        map_halfkp_features(&board, |_, _| count += 1);

        // 歩18枚（王は含まない）= 18
        assert_eq!(count, 18);
    }

    #[test]
    fn test_map_features_no_kings() {
        // HalfKP では王は特徴量に含めないことを確認
        let mut board = ShogiBoard::default();
        board.side_to_move = Color::Black;

        // 玉のみを配置
        board.black_king_sq = Square::new(4, 8); // 5九
        board.white_king_sq = Square::new(4, 0); // 5一
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        let mut count = 0;
        map_halfkp_features(&board, |_, _| count += 1);

        // 王は特徴量に含めないので 0
        assert_eq!(count, 0);
    }

    #[test]
    fn test_map_features_sq_nb_guard() {
        // 片玉データ（玉位置が SQ_NB=81）のテスト
        let mut board = ShogiBoard::default();
        board.side_to_move = Color::Black;

        // 先手玉を正常位置、後手玉を SQ_NB(81) に設定
        board.black_king_sq = Square::new(4, 8); // 5九
        board.white_king_sq = Square::NONE; // SQ_NB (81)
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);

        let mut count = 0;
        map_halfkp_features(&board, |_, _| count += 1);

        // 片玉データはスキップされるため、カウントは 0
        assert_eq!(count, 0);
    }

    #[test]
    fn test_feature_indices_in_range() {
        // 特徴インデックスが範囲内であることを確認
        let mut board = ShogiBoard::default();
        board.side_to_move = Color::Black;

        // 玉を配置
        board.black_king_sq = Square::new(4, 8); // 5九
        board.white_king_sq = Square::new(4, 0); // 5一
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        // 歩を配置
        for file in 0..9 {
            board.board[Square::new(file, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
            board.board[Square::new(file, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
        }

        let max_valid_index = HALFKP_DIMENSIONS - 1;
        map_halfkp_features(&board, |stm_idx, nstm_idx| {
            assert!(stm_idx <= max_valid_index, "STM index {} exceeds max {}", stm_idx, max_valid_index);
            assert!(nstm_idx <= max_valid_index, "NSTM index {} exceeds max {}", nstm_idx, max_valid_index);
        });
    }
}
