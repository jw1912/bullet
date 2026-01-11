//! PackedSfen / PackedSfenValue デコーダ
//!
//! YaneuraOu の教師データ形式を読み込むためのモジュール。
//! PackedSfenValue (40バイト) から局面を復元し、特徴量計算に使用する。
//!
//! YaneuraOu-latest 互換（駒箱対応）。

use super::types::{Color, Hand, Piece, PieceType, Square};

// =============================================================================
// Huffman 符号テーブル
// =============================================================================

/// Huffman 符号テーブル（YaneuraOu sfen_packer.cpp 準拠）
///
/// インデックス: 0=NO_PIECE, 1=PAWN, 2=LANCE, 3=KNIGHT, 4=SILVER, 5=BISHOP, 6=ROOK, 7=GOLD
const HUFFMAN_TABLE: [(u32, u8); 8] = [
    (0x00, 1), // NO_PIECE: 0
    (0x01, 2), // PAWN:     01
    (0x03, 4), // LANCE:    0011
    (0x0b, 4), // KNIGHT:   1011
    (0x07, 4), // SILVER:   0111
    (0x1f, 6), // BISHOP:   011111
    (0x3f, 6), // ROOK:     111111
    (0x0f, 5), // GOLD:     01111
];

/// 駒種インデックス（Huffman テーブル用）
const HUFFMAN_PAWN: usize = 1;
const HUFFMAN_LANCE: usize = 2;
const HUFFMAN_KNIGHT: usize = 3;
const HUFFMAN_SILVER: usize = 4;
const HUFFMAN_BISHOP: usize = 5;
const HUFFMAN_ROOK: usize = 6;
const HUFFMAN_GOLD: usize = 7;

// =============================================================================
// ビットストリーム
// =============================================================================

/// LSB-first ビットストリーム
pub struct BitStream<'a> {
    data: &'a [u8],
    bit_cursor: usize,
    /// ビット単位の上限（これ以上は読み出し不可）
    bit_limit: usize,
}

impl<'a> BitStream<'a> {
    /// 新しいビットストリームを作成
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, bit_cursor: 0, bit_limit: data.len() * 8 }
    }

    /// 現在のカーソル位置（ビット単位）
    #[inline]
    pub fn cursor(&self) -> usize {
        self.bit_cursor
    }

    /// 読み出し可能かどうか
    #[inline]
    pub fn can_read(&self) -> bool {
        self.bit_cursor < self.bit_limit
    }

    /// 1ビット読み出し（境界チェック付き）
    ///
    /// 境界を超えた場合は false を返す（安全なデフォルト）
    #[inline]
    pub fn read_bit(&mut self) -> bool {
        if self.bit_cursor >= self.bit_limit {
            return false;
        }
        let byte_pos = self.bit_cursor / 8;
        let bit_pos = self.bit_cursor % 8;
        let bit = (self.data[byte_pos] >> bit_pos) & 1;
        self.bit_cursor += 1;
        bit != 0
    }

    /// nビット読み出し（最大32ビット）
    #[inline]
    pub fn read_bits(&mut self, n: u8) -> u32 {
        let mut result = 0u32;
        for i in 0..n {
            if self.read_bit() {
                result |= 1 << i;
            }
        }
        result
    }
}

// =============================================================================
// PackedSfen (32バイト)
// =============================================================================

/// Huffman符号化された局面データ (32バイト)
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PackedSfen {
    pub data: [u8; 32],
}

impl PackedSfen {
    /// バイト配列への参照を取得
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.data
    }

    /// バイト配列への可変参照を取得
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 32] {
        &mut self.data
    }
}

// =============================================================================
// PackedSfenValue (40バイト)
// =============================================================================

/// YaneuraOu の教師データ形式 (40バイト)
///
/// これが `SparseInputType::RequiredDataType` として使用される。
///
/// メモリレイアウト:
/// - bytes 0-31:  PackedSfen (32 bytes)
/// - bytes 32-33: score (i16, little-endian)
/// - bytes 34-35: move16 (u16, little-endian)
/// - bytes 36-37: game_ply (u16, little-endian)
/// - byte 38:     game_result (i8)
/// - byte 39:     padding (u8)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PackedSfenValue {
    data: [u8; 40],
}

impl Default for PackedSfenValue {
    fn default() -> Self {
        Self { data: [0u8; 40] }
    }
}

// Safety: PackedSfenValue is a plain byte array
unsafe impl Send for PackedSfenValue {}
unsafe impl Sync for PackedSfenValue {}

impl PackedSfenValue {
    /// PackedSfen 部分への参照を取得
    pub fn sfen(&self) -> &PackedSfen {
        // Safety: PackedSfen は [u8; 32] と同じレイアウト
        unsafe { &*(self.data.as_ptr() as *const PackedSfen) }
    }

    /// 評価値を取得（手番側視点）
    pub fn score(&self) -> i16 {
        i16::from_le_bytes([self.data[32], self.data[33]])
    }

    /// 指し手を取得
    pub fn move16(&self) -> u16 {
        u16::from_le_bytes([self.data[34], self.data[35]])
    }

    /// 手数を取得
    pub fn game_ply(&self) -> u16 {
        u16::from_le_bytes([self.data[36], self.data[37]])
    }

    /// 勝敗結果を取得
    /// 1=手番側の勝ち, 0=引き分け, -1=手番側の負け
    pub fn game_result(&self) -> i8 {
        self.data[38] as i8
    }

    /// バイトスライスへの参照を取得
    pub fn as_bytes(&self) -> &[u8; 40] {
        &self.data
    }

    /// バイトスライスへの可変参照を取得
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 40] {
        &mut self.data
    }

    /// 局面をデコードして ShogiBoard を返す
    pub fn decode(&self) -> ShogiBoard {
        ShogiBoard::from_packed_sfen(self)
    }
}

// =============================================================================
// ShogiBoard - デコード済み局面
// =============================================================================

/// デコード済みの将棋局面
///
/// PackedSfenValue からデコードした結果を保持。
/// `map_features` で使用する。
#[derive(Clone)]
pub struct ShogiBoard {
    /// 盤面 (81マス)
    pub board: [Piece; 81],
    /// 先手の持ち駒
    pub black_hand: Hand,
    /// 後手の持ち駒
    pub white_hand: Hand,
    /// 手番
    pub side_to_move: Color,
    /// 先手玉の位置
    pub black_king_sq: Square,
    /// 後手玉の位置
    pub white_king_sq: Square,
    /// 評価値
    pub score: i16,
    /// 勝敗結果
    pub result: i8,
    /// 手数
    pub ply: u16,
}

impl Default for ShogiBoard {
    fn default() -> Self {
        Self {
            board: [Piece::NONE; 81],
            black_hand: Hand::EMPTY,
            white_hand: Hand::EMPTY,
            side_to_move: Color::Black,
            black_king_sq: Square::NONE,
            white_king_sq: Square::NONE,
            score: 0,
            result: 0,
            ply: 0,
        }
    }
}

impl ShogiBoard {
    /// PackedSfenValue からデコード
    pub fn from_packed_sfen(psv: &PackedSfenValue) -> Self {
        let mut board =
            ShogiBoard { score: psv.score(), result: psv.game_result(), ply: psv.game_ply(), ..Default::default() };

        let mut stream = BitStream::new(&psv.sfen().data);

        // 1. 手番 (1 bit)
        board.side_to_move = if stream.read_bit() { Color::White } else { Color::Black };

        // 2. 玉の位置 (7 bit × 2)
        let black_king_idx = stream.read_bits(7) as u8;
        let white_king_idx = stream.read_bits(7) as u8;

        board.black_king_sq = Square(black_king_idx);
        board.white_king_sq = Square(white_king_idx);

        // 玉を盤面に配置
        if black_king_idx < 81 {
            board.board[black_king_idx as usize] = Piece::new(Color::Black, PieceType::King);
        }
        if white_king_idx < 81 {
            board.board[white_king_idx as usize] = Piece::new(Color::White, PieceType::King);
        }

        // 3. 盤上の駒 (Huffman符号)
        for sq_idx in 0..81u8 {
            // 玉位置はスキップ
            if sq_idx == black_king_idx || sq_idx == white_king_idx {
                continue;
            }

            let piece = decode_board_piece(&mut stream);
            board.board[sq_idx as usize] = piece;
        }

        // 4. 持ち駒・駒箱 (Huffman符号、256bitまで)
        while stream.cursor() < 256 {
            let (piece, is_piecebox) = decode_hand_piece(&mut stream);

            // 駒箱の駒は無視（駒落ち対応）
            if is_piecebox {
                continue;
            }

            // 持ち駒に追加
            let pt = piece.piece_type;
            match piece.color {
                Color::Black => board.black_hand.add(pt, 1),
                Color::White => board.white_hand.add(pt, 1),
            }
        }

        board
    }

    /// 指定マスの駒を取得
    #[inline]
    pub fn piece_on(&self, sq: Square) -> Piece {
        self.board[sq.index()]
    }

    /// 指定色の玉位置を取得
    #[inline]
    pub fn king_square(&self, color: Color) -> Square {
        match color {
            Color::Black => self.black_king_sq,
            Color::White => self.white_king_sq,
        }
    }

    /// 指定色の持ち駒を取得
    #[inline]
    pub fn hand(&self, color: Color) -> &Hand {
        match color {
            Color::Black => &self.black_hand,
            Color::White => &self.white_hand,
        }
    }

    /// 盤上の指定色・駒種の駒を列挙
    pub fn pieces(&self, color: Color, pt: PieceType) -> impl Iterator<Item = Square> + '_ {
        self.board
            .iter()
            .enumerate()
            .filter(move |(_, p)| p.color == color && p.piece_type == pt)
            .map(|(i, _)| Square(i as u8))
    }
}

// =============================================================================
// Huffman デコード
// =============================================================================

/// Huffman 符号インデックスから PieceType に変換
fn huffman_index_to_piece_type(idx: usize) -> PieceType {
    match idx {
        HUFFMAN_PAWN => PieceType::Pawn,
        HUFFMAN_LANCE => PieceType::Lance,
        HUFFMAN_KNIGHT => PieceType::Knight,
        HUFFMAN_SILVER => PieceType::Silver,
        HUFFMAN_BISHOP => PieceType::Bishop,
        HUFFMAN_ROOK => PieceType::Rook,
        HUFFMAN_GOLD => PieceType::Gold,
        _ => PieceType::None,
    }
}

/// 盤上の駒をデコード
///
/// YaneuraOu sfen_packer.cpp の read_board_piece_from_stream() に準拠。
/// 形式: Huffman符号 + 成りbit(金以外) + 先後bit
fn decode_board_piece(stream: &mut BitStream) -> Piece {
    // Huffman 符号をデコードして駒種を取得
    let mut code = 0u32;
    let mut bits = 0u8;

    loop {
        code |= (stream.read_bit() as u32) << bits;
        bits += 1;

        // テーブルから一致するエントリを探す
        for (idx, &(pattern, len)) in HUFFMAN_TABLE.iter().enumerate() {
            if bits == len && code == pattern {
                if idx == 0 {
                    // NO_PIECE
                    return Piece::NONE;
                }

                let base_pt = huffman_index_to_piece_type(idx);

                // 成りフラグ（金以外）
                let promoted = if idx != HUFFMAN_GOLD { stream.read_bit() } else { false };

                // 先後フラグ
                let color = if stream.read_bit() { Color::White } else { Color::Black };

                let pt = if promoted { base_pt.promote() } else { base_pt };
                return Piece::new(color, pt);
            }
        }

        // 最大6ビット
        if bits > 6 {
            return Piece::NONE;
        }
    }
}

/// 手駒をデコード
///
/// YaneuraOu sfen_packer.cpp の read_hand_piece_from_stream() に準拠。
/// 形式: Huffman符号(bit0を除く) + 成りbit(金以外、駒箱判定用) + 先後bit
///
/// 戻り値: (駒, 駒箱フラグ)
/// - 駒箱フラグが true の場合、その駒は持ち駒ではなく駒箱の駒
fn decode_hand_piece(stream: &mut BitStream) -> (Piece, bool) {
    // 手駒の Huffman 符号は盤上より1ビット少ない（bit0 を省略）
    let mut code = 0u32;
    let mut bits = 0u8;

    loop {
        code |= (stream.read_bit() as u32) << bits;
        bits += 1;

        for (idx, &(pattern, len)) in HUFFMAN_TABLE.iter().enumerate() {
            // 手駒: pattern >> 1, len - 1
            if idx == 0 {
                continue; // NO_PIECE は手駒にない
            }

            let hand_pattern = pattern >> 1;
            let hand_len = len - 1;

            if bits == hand_len && code == hand_pattern {
                let base_pt = huffman_index_to_piece_type(idx);

                // 成りフラグ（金以外）
                // これが true なら駒箱の駒
                let is_piecebox = if idx != HUFFMAN_GOLD { stream.read_bit() } else { false };

                // 先後フラグ
                let color = if stream.read_bit() { Color::White } else { Color::Black };

                let piece = Piece::new(color, base_pt);
                return (piece, is_piecebox);
            }
        }

        if bits > 5 {
            // エラー: 不正な符号
            return (Piece::NONE, false);
        }
    }
}

// =============================================================================
// DirectSequentialDataLoader 対応
// =============================================================================

/// PackedSfenValue は固定長で transmute 可能
///
/// # Safety
/// - `#[repr(C)]` でメモリレイアウトが固定
/// - 40バイト固定長
/// - 任意のビットパターンでもパニックしない
unsafe impl crate::value::loader::CanBeDirectlySequentiallyLoaded for PackedSfenValue {}

/// LoadableDataType 実装
impl crate::value::loader::LoadableDataType for PackedSfenValue {
    fn score(&self) -> i16 {
        PackedSfenValue::score(self)
    }

    fn result(&self) -> crate::value::loader::GameResult {
        match self.game_result() {
            r if r > 0 => crate::value::loader::GameResult::Win,
            r if r < 0 => crate::value::loader::GameResult::Loss,
            _ => crate::value::loader::GameResult::Draw,
        }
    }
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_sfen_value_size() {
        assert_eq!(std::mem::size_of::<PackedSfenValue>(), 40);
    }

    #[test]
    fn test_packed_sfen_size() {
        assert_eq!(std::mem::size_of::<PackedSfen>(), 32);
    }

    #[test]
    fn test_bitstream_read_bits() {
        let data = [0b10101010, 0b11001100];
        let mut stream = BitStream::new(&data);

        // 0b10101010 の bit0-3 = 0,1,0,1 → result に bit 順に格納 → 0b1010 = 10
        assert_eq!(stream.read_bits(4), 0b1010);

        // 0b10101010 の bit4-7 = 0,1,0,1 → 0b1010 = 10
        assert_eq!(stream.read_bits(4), 0b1010);

        // 0b11001100 の bit0-3 = 0,0,1,1 → 0b1100 = 12
        assert_eq!(stream.read_bits(4), 0b1100);

        // 0b11001100 の bit4-7 = 0,0,1,1 → 0b1100 = 12
        assert_eq!(stream.read_bits(4), 0b1100);
    }

    #[test]
    fn test_bitstream_cursor() {
        let data = [0u8; 32];
        let mut stream = BitStream::new(&data);

        assert_eq!(stream.cursor(), 0);
        stream.read_bit();
        assert_eq!(stream.cursor(), 1);
        stream.read_bits(7);
        assert_eq!(stream.cursor(), 8);
        stream.read_bits(7);
        assert_eq!(stream.cursor(), 15);
    }

    #[test]
    fn test_packed_sfen_value_accessors() {
        let mut psv = PackedSfenValue::default();
        // score = 0x1234 at bytes 32-33
        psv.data[32] = 0x34;
        psv.data[33] = 0x12;
        // game_ply = 0x0056 at bytes 36-37
        psv.data[36] = 0x56;
        psv.data[37] = 0x00;
        // game_result = -1 at byte 38
        psv.data[38] = 0xFF; // -1 as i8

        assert_eq!(psv.score(), 0x1234);
        assert_eq!(psv.game_ply(), 0x0056);
        assert_eq!(psv.game_result(), -1);
    }

    #[test]
    fn test_huffman_decode_empty() {
        // 空マス: 0
        let data = [0b00000000u8; 32];
        let mut stream = BitStream::new(&data);

        let piece = decode_board_piece(&mut stream);
        assert_eq!(piece, Piece::NONE);
        assert_eq!(stream.cursor(), 1);
    }

    #[test]
    fn test_huffman_decode_pawn() {
        // 先手歩: 01 (PAWN) + 0 (不成) + 0 (先手) = 0001
        // LSB first: bit0=1, bit1=0, bit2=0, bit3=0 → 0b0001
        let data = [0b00000001u8, 0u8, 0u8, 0u8];
        let mut stream = BitStream::new(&data);

        let piece = decode_board_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::Pawn);
        assert_eq!(piece.color, Color::Black);
    }

    #[test]
    fn test_huffman_decode_promoted_pawn() {
        // 後手と金: 01 (PAWN) + 1 (成り) + 1 (後手) = 1101
        // LSB first: bit0=1, bit1=0, bit2=1, bit3=1 → 0b1101
        let data = [0b00001101u8, 0u8, 0u8, 0u8];
        let mut stream = BitStream::new(&data);

        let piece = decode_board_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::ProPawn);
        assert_eq!(piece.color, Color::White);
    }

    #[test]
    fn test_huffman_decode_gold() {
        // 先手金: 01111 (GOLD) + 0 (先手) = 001111
        // LSB first: 1,1,1,1,0,0 → 0b001111
        let data = [0b00001111u8, 0u8, 0u8, 0u8];
        let mut stream = BitStream::new(&data);

        let piece = decode_board_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::Gold);
        assert_eq!(piece.color, Color::Black);
    }

    #[test]
    fn test_bitstream_oob_protection() {
        // 2バイト = 16ビットのデータ
        let data = [0xFF, 0xFF];
        let mut stream = BitStream::new(&data);

        // 16ビット読み出し可能
        for _ in 0..16 {
            assert!(stream.can_read());
            stream.read_bit();
        }

        // 17ビット目以降は読み出し不可（false を返す）
        assert!(!stream.can_read());
        assert!(!stream.read_bit()); // OOB だが panic しない
        assert!(!stream.read_bit());
    }

    #[test]
    fn test_decode_hand_piece_pawn() {
        // 手駒の歩: 盤上歩の Huffman 符号 "01" から bit0 を除いた "0" + 成りbit(0) + 先後bit(0)
        // = 00 (2ビット)
        // LSB first: bit0=0, bit1=0 → 0b00
        let data = [0b00000000u8; 32];
        let mut stream = BitStream::new(&data);

        let (piece, is_piecebox) = decode_hand_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::Pawn);
        assert_eq!(piece.color, Color::Black);
        assert!(!is_piecebox);
    }

    #[test]
    fn test_decode_hand_piece_gold() {
        // 手駒の金: 盤上金の Huffman 符号 "01111" から bit0 を除いた "0111" + 先後bit(0)
        // 0111 (code) = bit0=1, bit1=1, bit2=1, bit3=0
        // 先後bit = bit4 = 0 (Black)
        // バイト: 0b000_0_0111 = 0b00000111 = 7
        let data = [0b00000111u8; 32];
        let mut stream = BitStream::new(&data);

        let (piece, is_piecebox) = decode_hand_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::Gold);
        assert_eq!(piece.color, Color::Black);
        assert!(!is_piecebox);
    }

    #[test]
    fn test_decode_hand_piece_piecebox_pawn() {
        // 駒箱の歩: "0" (歩の手駒符号) + 成りbit(1=駒箱) + 先後bit(0)
        // 0 (code) = bit0 = 0
        // 成りbit = bit1 = 1 (駒箱)
        // 先後bit = bit2 = 0 (Black)
        // バイト: 0b00000_0_1_0 = 0b00000010 = 2
        let data = [0b00000010u8; 32];
        let mut stream = BitStream::new(&data);

        let (piece, is_piecebox) = decode_hand_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::Pawn);
        assert_eq!(piece.color, Color::Black);
        assert!(is_piecebox); // 駒箱フラグが立っている
    }

    #[test]
    fn test_decode_hand_piece_piecebox_rook() {
        // 駒箱の飛: 盤上飛の Huffman "111111" から bit0 除去 → "11111" + 成りbit(1=駒箱) + 先後bit(0)
        // 11111 (code) = bit0-4 = 1,1,1,1,1
        // 成りbit = bit5 = 1 (駒箱)
        // 先後bit = bit6 = 0 (Black)
        // バイト: 0b0_0_1_11111 = 0b00111111 = 63
        let data = [0b00111111u8; 32];
        let mut stream = BitStream::new(&data);

        let (piece, is_piecebox) = decode_hand_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::Rook);
        assert_eq!(piece.color, Color::Black);
        assert!(is_piecebox);
    }

    #[test]
    fn test_hand_decode_cursor_256_boundary() {
        // 256ビット境界のテスト
        // 手駒ループは cursor < 256 でループするので、
        // 256ビットちょうどで終了することを確認

        let data = [0u8; 32];
        // 全て空マス符号 "0" で埋める（1ビット×256=256ビット）
        // これにより cursor が 256 に到達して終了

        let mut stream = BitStream::new(&data);

        // 256ビット読み出し
        for _ in 0..256 {
            stream.read_bit();
        }
        assert_eq!(stream.cursor(), 256);
        assert!(!stream.can_read()); // これ以上読めない

        // OOB アクセスしても panic しない
        assert!(!stream.read_bit());
    }
}
