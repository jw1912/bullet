//! HalfKA_hm 特徴量計算用ヘルパー関数
//!
//! Half-Mirror & King-Bucket 処理を行う関数群。

use super::PIECE_INPUTS;
use super::bona_piece::{BonaPiece, E_KING, F_KING, FE_HAND_END};
use super::types::{Color, Square};

// =============================================================================
// キングバケット計算
// =============================================================================

/// キングバケットを計算 (Half-Mirror)
///
/// 玉位置を 45 バケット (9段 × 5筋) に圧縮。
/// ファイル 5-8 (0-indexed) は 0-3 にミラーリング。
///
/// # Arguments
/// - `ksq`: 玉のマス
/// - `perspective`: 視点
///
/// # Returns
/// バケット番号 (0-44)
#[inline]
pub fn king_bucket(ksq: Square, perspective: Color) -> usize {
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
pub fn is_hm_mirror(ksq: Square, perspective: Color) -> bool {
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
///
/// # Returns
/// パック後の値 (0..1628)
#[inline]
pub fn pack_bonapiece(bp: BonaPiece, hm_mirror: bool) -> usize {
    let mut pp = bp.value() as usize;

    // 手駒はミラー不要
    if hm_mirror && pp >= FE_HAND_END {
        // 盤上駒: layout is fe_hand_end + piece_index*81 + sq
        let rel = pp - FE_HAND_END;
        let piece_index = rel / 81;
        let sq = rel % 81;

        // マス目をミラー（ファイルのみ: 1筋 ↔ 9筋）
        // Square index: sq = file * 9 + rank (file: 0-8, rank: 0-8)
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
/// - is_friend=true: F_KING + sq_index (自玉)
/// - is_friend=false: E_KING + sq_index (敵玉)
#[inline]
pub fn king_bonapiece(sq_index: usize, is_friend: bool) -> BonaPiece {
    let base = if is_friend { F_KING } else { E_KING };
    BonaPiece::new((base as usize + sq_index) as u16)
}

// =============================================================================
// 特徴量インデックス計算
// =============================================================================

/// HalfKA_hm の特徴インデックスを計算
///
/// index = king_bucket * PIECE_INPUTS + packed_bonapiece
#[inline]
pub fn halfka_index(kb: usize, packed_bp: usize) -> usize {
    kb * PIECE_INPUTS + packed_bp
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // sq = file * 9 + rank

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
}
