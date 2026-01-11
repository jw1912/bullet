//! ShogiHalfKA_hm - 将棋用 HalfKA_hm 特徴量
//!
//! Half-Mirror King + All pieces (coalesced) 特徴量。
//!
//! - キングバケット: 45バケット (Half-Mirror: 9段 × 5筋)
//! - 入力次元: 73,305 (45 × 1629)
//! - 最大アクティブ特徴: 40

use super::SparseInputType;
use crate::shogi::{
    BonaPiece, PackedSfenValue, ShogiBoard,
    bona_piece::{E_KING, F_KING, FE_HAND_END},
    types::{BOARD_PIECE_TYPES, Color, HAND_PIECE_TYPES, Piece, Square},
};

// =============================================================================
// 定数
// =============================================================================

/// キングバケット数 (Half-Mirror: 9段 × 5筋)
pub const NUM_KING_BUCKETS: usize = 45;

/// 駒入力数 (BonaPiece の最大値)
pub const PIECE_INPUTS: usize = 1629;

/// HalfKA_hm の総入力次元
pub const HALFKA_HM_DIMENSIONS: usize = NUM_KING_BUCKETS * PIECE_INPUTS; // 73,305

/// 最大アクティブ特徴数 (盤上駒 + 手駒 = 40)
pub const MAX_ACTIVE_FEATURES: usize = 40;

// =============================================================================
// ShogiHalfKA_hm 特徴量型
// =============================================================================

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
        map_halfka_features(&board, f);
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

// =============================================================================
// HalfKA 特徴量計算
// =============================================================================

/// HalfKA_hm 特徴量インデックスを列挙
///
/// stm (side-to-move) 視点と nstm (not-side-to-move) 視点の両方を返す。
/// 片玉・詰将棋データ（玉位置が SQ_NB=81）の場合は何もしない。
fn map_halfka_features<F: FnMut(usize, usize)>(board: &ShogiBoard, mut f: F) {
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

    // STM 視点でのキングバケット計算
    let stm_kb = king_bucket(stm_king_sq, stm);
    let stm_hm = is_hm_mirror(stm_king_sq, stm);

    // NSTM 視点でのキングバケット計算
    let nstm_kb = king_bucket(nstm_king_sq, nstm);
    let nstm_hm = is_hm_mirror(nstm_king_sq, nstm);

    // 盤上の駒（王以外）
    for &pt in &BOARD_PIECE_TYPES {
        for color in [Color::Black, Color::White] {
            for sq in board.pieces(color, pt) {
                // STM 視点での BonaPiece
                let piece = Piece::new(color, pt);
                let stm_bp = BonaPiece::from_piece_square(piece, sq, stm);
                let stm_packed = pack_bonapiece(stm_bp, stm_hm);
                let stm_idx = halfka_index(stm_kb, stm_packed);

                // NSTM 視点での BonaPiece
                let nstm_bp = BonaPiece::from_piece_square(piece, sq, nstm);
                let nstm_packed = pack_bonapiece(nstm_bp, nstm_hm);
                let nstm_idx = halfka_index(nstm_kb, nstm_packed);

                f(stm_idx, nstm_idx);
            }
        }
    }

    // 両方の玉の特徴量
    // STM 視点での自玉と敵玉
    {
        // 自玉 (STM視点)
        let stm_king_sq_idx = if stm == Color::Black { stm_king_sq.index() } else { stm_king_sq.inverse().index() };
        let stm_friend_king_bp = king_bonapiece(stm_king_sq_idx, true);
        let stm_friend_packed = pack_bonapiece(stm_friend_king_bp, stm_hm);
        let stm_friend_idx = halfka_index(stm_kb, stm_friend_packed);

        // 敵玉 (STM視点)
        let nstm_king_sq_for_stm =
            if stm == Color::Black { nstm_king_sq.index() } else { nstm_king_sq.inverse().index() };
        let stm_enemy_king_bp = king_bonapiece(nstm_king_sq_for_stm, false);
        let stm_enemy_packed = pack_bonapiece(stm_enemy_king_bp, stm_hm);
        let stm_enemy_idx = halfka_index(stm_kb, stm_enemy_packed);

        // NSTM 視点での自玉と敵玉
        let nstm_king_sq_idx = if nstm == Color::Black { nstm_king_sq.index() } else { nstm_king_sq.inverse().index() };
        let nstm_friend_king_bp = king_bonapiece(nstm_king_sq_idx, true);
        let nstm_friend_packed = pack_bonapiece(nstm_friend_king_bp, nstm_hm);
        let nstm_friend_idx = halfka_index(nstm_kb, nstm_friend_packed);

        let stm_king_sq_for_nstm =
            if nstm == Color::Black { stm_king_sq.index() } else { stm_king_sq.inverse().index() };
        let nstm_enemy_king_bp = king_bonapiece(stm_king_sq_for_nstm, false);
        let nstm_enemy_packed = pack_bonapiece(nstm_enemy_king_bp, nstm_hm);
        let nstm_enemy_idx = halfka_index(nstm_kb, nstm_enemy_packed);

        // 自玉の特徴量
        f(stm_friend_idx, nstm_friend_idx);
        // 敵玉の特徴量
        f(stm_enemy_idx, nstm_enemy_idx);
    }

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
                if stm_bp != BonaPiece::ZERO {
                    let stm_packed = pack_bonapiece(stm_bp, stm_hm);
                    let stm_idx = halfka_index(stm_kb, stm_packed);

                    // NSTM 視点
                    let nstm_bp = BonaPiece::from_hand_piece(nstm, owner, pt, i);
                    let nstm_packed = pack_bonapiece(nstm_bp, nstm_hm);
                    let nstm_idx = halfka_index(nstm_kb, nstm_packed);

                    f(stm_idx, nstm_idx);
                }
            }
        }
    }
}

// =============================================================================
// キングバケット計算
// =============================================================================

/// キングバケットを計算 (Half-Mirror)
///
/// 玉位置を 45 バケット (9段 × 5筋) に圧縮。
/// ファイル 5-8 (0-indexed) は 0-3 にミラーリング。
#[inline]
fn king_bucket(ksq: Square, perspective: Color) -> usize {
    // 視点に応じてマスを変換
    let sq = if perspective == Color::Black { ksq } else { ksq.inverse() };

    let file = sq.file() as usize; // 0..8
    let rank = sq.rank() as usize; // 0..8

    // Half-mirror: file >= 5 なら反転 (5,6,7,8 → 3,2,1,0)
    let file_m = if file >= 5 { 8 - file } else { file }; // 0..4

    rank * 5 + file_m // 0..44
}

/// Half-Mirror が必要かどうかを判定
///
/// 玉のファイルが 5 以上 (6筋-9筋) の場合に true。
#[inline]
fn is_hm_mirror(ksq: Square, perspective: Color) -> bool {
    let sq = if perspective == Color::Black { ksq } else { ksq.inverse() };
    sq.file() as usize >= 5
}

// =============================================================================
// BonaPiece パッキング
// =============================================================================

/// BonaPiece を HalfKA_hm 用にパック
///
/// 1. 手駒 (<90): そのまま
/// 2. 盤上駒 (>=90): hm_mirror が必要な場合はマス目を反転
/// 3. 敵王 (>=e_king): -81 して f_king 平面に揃える
#[inline]
fn pack_bonapiece(bp: BonaPiece, hm_mirror: bool) -> usize {
    let mut pp = bp.value() as usize;

    // 手駒はミラー不要
    if hm_mirror && pp >= FE_HAND_END {
        // 盤上駒: layout is fe_hand_end + piece_index*81 + sq
        let rel = pp - FE_HAND_END;
        let piece_index = rel / 81;
        let sq = rel % 81;

        // マス目をミラー（ファイルのみ: 1筋 ↔ 9筋）
        let file = sq / 9;
        let rank = sq % 9;
        let mirrored_file = 8 - file;
        let mirrored_sq = mirrored_file * 9 + rank;

        pp = FE_HAND_END + piece_index * 81 + mirrored_sq;
    }

    // 敵王を先手王平面にパック
    if pp >= E_KING as usize {
        pp -= 81;
    }

    pp // 0..1628
}

/// 王の BonaPiece を生成
///
/// HalfKA_hm では両方の王を特徴量に含める。
#[inline]
fn king_bonapiece(sq_index: usize, is_friend: bool) -> BonaPiece {
    let base = if is_friend { F_KING } else { E_KING };
    BonaPiece::new((base as usize + sq_index) as u16)
}

/// HalfKA_hm の特徴インデックスを計算
#[inline]
fn halfka_index(kb: usize, packed_bp: usize) -> usize {
    kb * PIECE_INPUTS + packed_bp
}

// =============================================================================
// テスト
// =============================================================================

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

    #[test]
    fn test_king_bucket_black_perspective() {
        // 5九 (file=4, rank=8): bucket = 8*5 + 4 = 44
        let sq_59 = Square::new(4, 8);
        assert_eq!(king_bucket(sq_59, Color::Black), 44);

        // 1九 (file=0, rank=8): bucket = 8*5 + 0 = 40
        let sq_19 = Square::new(0, 8);
        assert_eq!(king_bucket(sq_19, Color::Black), 40);

        // 9九 (file=8, mirror to 0, rank=8): bucket = 8*5 + 0 = 40
        let sq_99 = Square::new(8, 8);
        assert_eq!(king_bucket(sq_99, Color::Black), 40);

        // 6九 (file=5, mirror to 3, rank=8): bucket = 8*5 + 3 = 43
        let sq_69 = Square::new(5, 8);
        assert_eq!(king_bucket(sq_69, Color::Black), 43);

        // 5一 (file=4, rank=0): bucket = 0*5 + 4 = 4
        let sq_51 = Square::new(4, 0);
        assert_eq!(king_bucket(sq_51, Color::Black), 4);

        // 1一 (file=0, rank=0): bucket = 0*5 + 0 = 0
        let sq_11 = Square::new(0, 0);
        assert_eq!(king_bucket(sq_11, Color::Black), 0);
    }

    #[test]
    fn test_is_hm_mirror() {
        // ファイル1-5 (index 0-4): ミラー不要
        assert!(!is_hm_mirror(Square::new(0, 0), Color::Black));
        assert!(!is_hm_mirror(Square::new(4, 8), Color::Black));

        // ファイル6-9 (index 5-8): ミラー必要
        assert!(is_hm_mirror(Square::new(5, 0), Color::Black));
        assert!(is_hm_mirror(Square::new(8, 8), Color::Black));
    }

    #[test]
    fn test_pack_bonapiece_hand_no_mirror() {
        // 手駒はミラーしない
        let bp = BonaPiece::new(50); // 手駒領域内
        assert_eq!(pack_bonapiece(bp, true), 50);
        assert_eq!(pack_bonapiece(bp, false), 50);
    }

    #[test]
    fn test_pack_bonapiece_board_mirror() {
        // 盤上駒のミラー
        // f_pawn (90) + sq の場合

        // sq=0 (1一): file=0, rank=0
        // ミラー後: file=8 (9筋), rank=0 → sq = 8*9+0 = 72
        let sq = 0;
        let bp = BonaPiece::new((90 + sq) as u16);
        assert_eq!(pack_bonapiece(bp, false), 90 + sq);
        assert_eq!(pack_bonapiece(bp, true), 90 + 72);

        // sq=9 (2一): file=1, rank=0
        // ミラー後: file=7 (8筋), rank=0 → sq = 7*9+0 = 63
        let sq = 9;
        let bp = BonaPiece::new((90 + sq) as u16);
        assert_eq!(pack_bonapiece(bp, false), 90 + sq);
        assert_eq!(pack_bonapiece(bp, true), 90 + 63);
    }

    #[test]
    fn test_pack_bonapiece_enemy_king() {
        // 敵王のパック: e_king - 81
        let bp = BonaPiece::new(E_KING);
        assert_eq!(pack_bonapiece(bp, false), E_KING as usize - 81);
    }

    #[test]
    fn test_halfka_index() {
        // kb=0, bp=0 → index=0
        assert_eq!(halfka_index(0, 0), 0);

        // kb=1, bp=0 → index=1629
        assert_eq!(halfka_index(1, 0), PIECE_INPUTS);

        // kb=44, bp=0 → index=44*1629=71676
        assert_eq!(halfka_index(44, 0), 44 * PIECE_INPUTS);
    }

    #[test]
    fn test_king_bonapiece() {
        // 自玉 (sq_index=0)
        let bp = king_bonapiece(0, true);
        assert_eq!(bp.value(), F_KING);

        // 敵玉 (sq_index=0)
        let bp = king_bonapiece(0, false);
        assert_eq!(bp.value(), E_KING);
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
        map_halfka_features(&board, |_, _| count += 1);

        // 歩18枚 + 両玉2 = 20（これは簡易テスト）
        assert!(count >= 20);
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
        map_halfka_features(&board, |_, _| count += 1);

        // 片玉データはスキップされるため、カウントは 0
        assert_eq!(count, 0);
    }
}
