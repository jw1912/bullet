//! BonaPiece - 駒の種類と位置を一意に表現するインデックス
//!
//! YaneuraOu の NNUE 実装で用いられる BonaPiece に準拠した定義。
//!
//! ## YaneuraOu BonaPiece定義 (DISTINGUISH_GOLDS無効時)
//!
//! ### 手駒 (1〜89)
//! - f_hand_pawn = 1, e_hand_pawn = 20 (各18枚分)
//! - f_hand_lance = 39, e_hand_lance = 44 (各4枚分)
//! - f_hand_knight = 49, e_hand_knight = 54 (各4枚分)
//! - f_hand_silver = 59, e_hand_silver = 64 (各4枚分)
//! - f_hand_gold = 69, e_hand_gold = 74 (各4枚分)
//! - f_hand_bishop = 79, e_hand_bishop = 82 (各2枚分)
//! - f_hand_rook = 85, e_hand_rook = 88 (各2枚分)
//! - fe_hand_end = 90
//!
//! ### 盤上駒 (90〜1547)
//! - f_pawn = 90, e_pawn = 171
//! - f_lance = 252, e_lance = 333
//! - f_knight = 414, e_knight = 495
//! - f_silver = 576, e_silver = 657
//! - f_gold = 738, e_gold = 819
//! - f_bishop = 900, e_bishop = 981
//! - f_horse = 1062, e_horse = 1143
//! - f_rook = 1224, e_rook = 1305
//! - f_dragon = 1386, e_dragon = 1467
//! - fe_end = 1548
//!
//! ### 王 (1548〜1710)
//! - f_king = 1548, e_king = 1629

use super::types::{Color, Piece, PieceType, Square};

// =============================================================================
// 定数定義
// =============================================================================

/// 手駒領域の終端
pub const FE_HAND_END: usize = 90;

// 先手の手駒ベースオフセット
pub const F_HAND_PAWN: u16 = 1;
pub const F_HAND_LANCE: u16 = 39;
pub const F_HAND_KNIGHT: u16 = 49;
pub const F_HAND_SILVER: u16 = 59;
pub const F_HAND_GOLD: u16 = 69;
pub const F_HAND_BISHOP: u16 = 79;
pub const F_HAND_ROOK: u16 = 85;

// 後手の手駒ベースオフセット
pub const E_HAND_PAWN: u16 = 20;
pub const E_HAND_LANCE: u16 = 44;
pub const E_HAND_KNIGHT: u16 = 54;
pub const E_HAND_SILVER: u16 = 64;
pub const E_HAND_GOLD: u16 = 74;
pub const E_HAND_BISHOP: u16 = 82;
pub const E_HAND_ROOK: u16 = 88;

// 盤上駒ベースオフセット
pub const F_PAWN: u16 = 90;
pub const E_PAWN: u16 = 171;
pub const F_LANCE: u16 = 252;
pub const E_LANCE: u16 = 333;
pub const F_KNIGHT: u16 = 414;
pub const E_KNIGHT: u16 = 495;
pub const F_SILVER: u16 = 576;
pub const E_SILVER: u16 = 657;
pub const F_GOLD: u16 = 738;
pub const E_GOLD: u16 = 819;
pub const F_BISHOP: u16 = 900;
pub const E_BISHOP: u16 = 981;
pub const F_HORSE: u16 = 1062;
pub const E_HORSE: u16 = 1143;
pub const F_ROOK: u16 = 1224;
pub const E_ROOK: u16 = 1305;
pub const F_DRAGON: u16 = 1386;
pub const E_DRAGON: u16 = 1467;

/// 盤上駒領域の終端
pub const FE_OLD_END: usize = 1548;

/// 先手王の開始位置
pub const F_KING: u16 = 1548;

/// 後手王の開始位置
pub const E_KING: u16 = 1629;

/// 駒種・is_friend に対する base offset テーブル（盤上駒用）
/// `[piece_type as usize][is_friend as usize]` -> base offset
/// is_friend: 0=enemy, 1=friend
pub const PIECE_BASE: [[u16; 2]; 15] = [
    // index 0: 未使用（ダミー）
    [0, 0],
    // PieceType::Pawn = 1
    [E_PAWN, F_PAWN],
    // PieceType::Lance = 2
    [E_LANCE, F_LANCE],
    // PieceType::Knight = 3
    [E_KNIGHT, F_KNIGHT],
    // PieceType::Silver = 4
    [E_SILVER, F_SILVER],
    // PieceType::Bishop = 5
    [E_BISHOP, F_BISHOP],
    // PieceType::Rook = 6
    [E_ROOK, F_ROOK],
    // PieceType::Gold = 7
    [E_GOLD, F_GOLD],
    // PieceType::King = 8 (使用しない)
    [0, 0],
    // PieceType::ProPawn = 9 (Gold と同じ扱い)
    [E_GOLD, F_GOLD],
    // PieceType::ProLance = 10
    [E_GOLD, F_GOLD],
    // PieceType::ProKnight = 11
    [E_GOLD, F_GOLD],
    // PieceType::ProSilver = 12
    [E_GOLD, F_GOLD],
    // PieceType::Horse = 13
    [E_HORSE, F_HORSE],
    // PieceType::Dragon = 14
    [E_DRAGON, F_DRAGON],
];

// =============================================================================
// BonaPiece
// =============================================================================

/// BonaPiece - 駒の種類と位置を一意に表現するインデックス
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct BonaPiece(pub u16);

impl BonaPiece {
    /// ゼロ（無効値）
    pub const ZERO: BonaPiece = BonaPiece(0);

    /// 新しいBonaPieceを作成
    #[inline]
    pub const fn new(value: u16) -> Self {
        Self(value)
    }

    /// 値を取得
    #[inline]
    pub const fn value(self) -> u16 {
        self.0
    }

    /// 盤上の駒から BonaPiece を計算
    ///
    /// # Arguments
    /// - `piece`: 駒
    /// - `sq`: マス
    /// - `perspective`: 視点
    ///
    /// # Returns
    /// BonaPiece。玉の場合は king_bonapiece を使用すること。
    pub fn from_piece_square(piece: Piece, sq: Square, perspective: Color) -> BonaPiece {
        if piece.is_none() {
            return BonaPiece::ZERO;
        }

        let pt = piece.piece_type;
        let pc_color = piece.color;

        // 玉は専用の関数を使う
        if pt == PieceType::King {
            return BonaPiece::ZERO;
        }

        // 視点に応じてマスを変換
        let sq_index = if perspective == Color::Black { sq.index() } else { sq.inverse().index() };

        // 駒の色が視点と同じかどうか
        let is_friend = pc_color == perspective;

        // PIECE_BASE テーブルからベースオフセットを取得
        let base = PIECE_BASE[pt as usize][is_friend as usize];

        BonaPiece::new(base + sq_index as u16)
    }

    /// 手駒から BonaPiece を計算
    ///
    /// # Arguments
    /// - `perspective`: 視点
    /// - `owner`: 持ち駒の所有者
    /// - `pt`: 駒種
    /// - `count`: 枚数 (1から始まる)
    ///
    /// count=1 のとき base が返る（1枚目のBonaPiece）。
    pub fn from_hand_piece(perspective: Color, owner: Color, pt: PieceType, count: u8) -> BonaPiece {
        if count == 0 {
            return BonaPiece::ZERO;
        }

        let is_friend = owner == perspective;

        let base = match pt {
            PieceType::Pawn => {
                if is_friend {
                    F_HAND_PAWN
                } else {
                    E_HAND_PAWN
                }
            }
            PieceType::Lance => {
                if is_friend {
                    F_HAND_LANCE
                } else {
                    E_HAND_LANCE
                }
            }
            PieceType::Knight => {
                if is_friend {
                    F_HAND_KNIGHT
                } else {
                    E_HAND_KNIGHT
                }
            }
            PieceType::Silver => {
                if is_friend {
                    F_HAND_SILVER
                } else {
                    E_HAND_SILVER
                }
            }
            PieceType::Gold => {
                if is_friend {
                    F_HAND_GOLD
                } else {
                    E_HAND_GOLD
                }
            }
            PieceType::Bishop => {
                if is_friend {
                    F_HAND_BISHOP
                } else {
                    E_HAND_BISHOP
                }
            }
            PieceType::Rook => {
                if is_friend {
                    F_HAND_ROOK
                } else {
                    E_HAND_ROOK
                }
            }
            _ => return BonaPiece::ZERO,
        };

        // count に応じてオフセット (count=1 のとき base)
        BonaPiece::new(base + count as u16 - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bona_piece_zero() {
        assert_eq!(BonaPiece::ZERO.value(), 0);
    }

    #[test]
    fn test_bona_piece_from_piece_square() {
        let sq = Square::new(6, 6); // 7七
        let piece = Piece::new(Color::Black, PieceType::Pawn);

        let bp = BonaPiece::from_piece_square(piece, sq, Color::Black);
        assert_ne!(bp, BonaPiece::ZERO);
        // f_pawn (90) + sq_index (6*9+6=60) = 150
        assert_eq!(bp.value(), F_PAWN + 60);
    }

    #[test]
    fn test_bona_piece_from_hand() {
        // 先手視点で先手の歩1枚目
        let bp = BonaPiece::from_hand_piece(Color::Black, Color::Black, PieceType::Pawn, 1);
        assert_eq!(bp.value(), F_HAND_PAWN);

        // 先手視点で先手の歩2枚目
        let bp = BonaPiece::from_hand_piece(Color::Black, Color::Black, PieceType::Pawn, 2);
        assert_eq!(bp.value(), F_HAND_PAWN + 1);

        // 先手視点で後手の歩1枚目
        let bp = BonaPiece::from_hand_piece(Color::Black, Color::White, PieceType::Pawn, 1);
        assert_eq!(bp.value(), E_HAND_PAWN);
    }

    #[test]
    fn test_piece_base_consistency() {
        // 歩
        assert_eq!(PIECE_BASE[PieceType::Pawn as usize][1], F_PAWN);
        assert_eq!(PIECE_BASE[PieceType::Pawn as usize][0], E_PAWN);

        // 成金は金と同じ扱い
        assert_eq!(PIECE_BASE[PieceType::ProPawn as usize][1], F_GOLD);
        assert_eq!(PIECE_BASE[PieceType::ProPawn as usize][0], E_GOLD);
    }
}
