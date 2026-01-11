//! 将棋固有のモジュール
//!
//! bullet を将棋 NNUE 学習に対応させるための型・関数定義。
//!
//! - `types`: 基本型 (Color, PieceType, Square, Piece)
//! - `packed_sfen`: PackedSfen/PackedSfenValue デコーダ
//! - `bona_piece`: BonaPiece 定義
//! - `halfka`: HalfKA_hm 特徴量計算用ヘルパー

pub mod bona_piece;
pub mod halfka;
pub mod packed_sfen;
pub mod types;

pub use bona_piece::BonaPiece;
pub use halfka::{halfka_index, is_hm_mirror, king_bonapiece, king_bucket, pack_bonapiece};
pub use packed_sfen::{BitStream, PackedSfen, PackedSfenValue, ShogiBoard};
pub use types::{Color, Hand, Piece, PieceType, Square};

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
