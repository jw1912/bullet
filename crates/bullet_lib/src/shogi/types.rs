//! 将棋の基本型定義
//!
//! Color, PieceType, Square, Piece, Hand など。

/// 手番 (先手/後手)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Color {
    #[default]
    Black = 0, // 先手
    White = 1, // 後手
}

impl Color {
    /// 相手の手番を取得
    #[inline]
    pub const fn opponent(self) -> Self {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }
}

/// 駒種 (生駒・成駒)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum PieceType {
    #[default]
    None = 0,
    Pawn = 1,       // 歩
    Lance = 2,      // 香
    Knight = 3,     // 桂
    Silver = 4,     // 銀
    Bishop = 5,     // 角
    Rook = 6,       // 飛
    Gold = 7,       // 金
    King = 8,       // 玉
    ProPawn = 9,    // と
    ProLance = 10,  // 成香
    ProKnight = 11, // 成桂
    ProSilver = 12, // 成銀
    Horse = 13,     // 馬
    Dragon = 14,    // 龍
}

impl PieceType {
    /// u8 から PieceType に変換
    #[inline]
    pub const fn from_u8(value: u8) -> Self {
        match value {
            0 => PieceType::None,
            1 => PieceType::Pawn,
            2 => PieceType::Lance,
            3 => PieceType::Knight,
            4 => PieceType::Silver,
            5 => PieceType::Bishop,
            6 => PieceType::Rook,
            7 => PieceType::Gold,
            8 => PieceType::King,
            9 => PieceType::ProPawn,
            10 => PieceType::ProLance,
            11 => PieceType::ProKnight,
            12 => PieceType::ProSilver,
            13 => PieceType::Horse,
            14 => PieceType::Dragon,
            _ => PieceType::None,
        }
    }

    /// 成駒かどうか
    #[inline]
    pub const fn is_promoted(self) -> bool {
        matches!(
            self,
            PieceType::ProPawn
                | PieceType::ProLance
                | PieceType::ProKnight
                | PieceType::ProSilver
                | PieceType::Horse
                | PieceType::Dragon
        )
    }

    /// 手駒にできる駒種かどうか
    #[inline]
    pub const fn can_be_in_hand(self) -> bool {
        matches!(
            self,
            PieceType::Pawn
                | PieceType::Lance
                | PieceType::Knight
                | PieceType::Silver
                | PieceType::Gold
                | PieceType::Bishop
                | PieceType::Rook
        )
    }

    /// 成駒に変換
    ///
    /// 金・玉は成れないので自身を返す。
    #[inline]
    pub const fn promote(self) -> Self {
        match self {
            PieceType::Pawn => PieceType::ProPawn,
            PieceType::Lance => PieceType::ProLance,
            PieceType::Knight => PieceType::ProKnight,
            PieceType::Silver => PieceType::ProSilver,
            PieceType::Bishop => PieceType::Horse,
            PieceType::Rook => PieceType::Dragon,
            // 金・玉・成駒は成れない
            _ => self,
        }
    }

    /// 生駒に変換（成りを戻す）
    #[inline]
    pub const fn unpromote(self) -> Self {
        match self {
            PieceType::ProPawn => PieceType::Pawn,
            PieceType::ProLance => PieceType::Lance,
            PieceType::ProKnight => PieceType::Knight,
            PieceType::ProSilver => PieceType::Silver,
            PieceType::Horse => PieceType::Bishop,
            PieceType::Dragon => PieceType::Rook,
            _ => self,
        }
    }
}

/// マス (0-80)
///
/// 将棋盤のマスを表す。
/// インデックス: file * 9 + rank (file: 0-8, rank: 0-8)
/// file=0 は 1筋、rank=0 は 1段。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct Square(pub u8);

impl Square {
    /// 無効なマス
    pub const NONE: Square = Square(81);

    /// 新しい Square を作成
    #[inline]
    pub const fn new(file: u8, rank: u8) -> Self {
        debug_assert!(file < 9 && rank < 9);
        Square(file * 9 + rank)
    }

    /// インデックスから Square を作成
    #[inline]
    pub const fn from_index(index: usize) -> Self {
        debug_assert!(index < 81);
        Square(index as u8)
    }

    /// インデックスを取得
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// 筋 (0-8, 0=1筋)
    #[inline]
    pub const fn file(self) -> u8 {
        self.0 / 9
    }

    /// 段 (0-8, 0=1段)
    #[inline]
    pub const fn rank(self) -> u8 {
        self.0 % 9
    }

    /// 180度回転 (先後反転)
    #[inline]
    pub const fn inverse(self) -> Self {
        Square(80 - self.0)
    }

    /// 筋のみを反転 (Half-Mirror用)
    #[inline]
    pub const fn mirror_file(self) -> Self {
        let file = self.file();
        let rank = self.rank();
        let mirrored_file = 8 - file;
        Square::new(mirrored_file, rank)
    }

    /// 有効なマスかどうか
    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 < 81
    }
}

/// 駒 (色 + 駒種)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Piece {
    /// 駒種
    pub piece_type: PieceType,
    /// 色 (所有者)
    pub color: Color,
}

impl Piece {
    /// 空マス
    pub const NONE: Piece = Piece { piece_type: PieceType::None, color: Color::Black };

    /// 新しい Piece を作成
    #[inline]
    pub const fn new(color: Color, piece_type: PieceType) -> Self {
        Piece { piece_type, color }
    }

    /// 空マスかどうか
    #[inline]
    pub const fn is_none(self) -> bool {
        matches!(self.piece_type, PieceType::None)
    }

    /// 空マスでないかどうか
    #[inline]
    pub const fn is_some(self) -> bool {
        !self.is_none()
    }
}

/// 持ち駒
#[derive(Debug, Clone, Copy, Default)]
pub struct Hand {
    /// 各駒種の枚数 (pawn, lance, knight, silver, gold, bishop, rook)
    pub counts: [u8; 7],
}

impl Hand {
    /// 空の持ち駒
    pub const EMPTY: Hand = Hand { counts: [0; 7] };

    /// 駒種のインデックス
    const fn piece_index(pt: PieceType) -> Option<usize> {
        match pt {
            PieceType::Pawn => Some(0),
            PieceType::Lance => Some(1),
            PieceType::Knight => Some(2),
            PieceType::Silver => Some(3),
            PieceType::Gold => Some(4),
            PieceType::Bishop => Some(5),
            PieceType::Rook => Some(6),
            _ => None,
        }
    }

    /// 指定駒種の枚数を取得
    #[inline]
    pub fn count(&self, pt: PieceType) -> u8 {
        Self::piece_index(pt).map_or(0, |i| self.counts[i])
    }

    /// 指定駒種の枚数を設定
    #[inline]
    pub fn set(&mut self, pt: PieceType, count: u8) {
        if let Some(i) = Self::piece_index(pt) {
            self.counts[i] = count;
        }
    }

    /// 指定駒種の枚数を追加
    #[inline]
    pub fn add(&mut self, pt: PieceType, count: u8) {
        if let Some(i) = Self::piece_index(pt) {
            self.counts[i] += count;
        }
    }

    /// 歩の枚数
    #[inline]
    pub fn pawn(&self) -> u8 {
        self.counts[0]
    }
    /// 香の枚数
    #[inline]
    pub fn lance(&self) -> u8 {
        self.counts[1]
    }
    /// 桂の枚数
    #[inline]
    pub fn knight(&self) -> u8 {
        self.counts[2]
    }
    /// 銀の枚数
    #[inline]
    pub fn silver(&self) -> u8 {
        self.counts[3]
    }
    /// 金の枚数
    #[inline]
    pub fn gold(&self) -> u8 {
        self.counts[4]
    }
    /// 角の枚数
    #[inline]
    pub fn bishop(&self) -> u8 {
        self.counts[5]
    }
    /// 飛の枚数
    #[inline]
    pub fn rook(&self) -> u8 {
        self.counts[6]
    }

    /// 歩の枚数を設定
    #[inline]
    pub fn set_pawn(&mut self, count: u8) {
        self.counts[0] = count;
    }
    /// 香の枚数を設定
    #[inline]
    pub fn set_lance(&mut self, count: u8) {
        self.counts[1] = count;
    }
    /// 桂の枚数を設定
    #[inline]
    pub fn set_knight(&mut self, count: u8) {
        self.counts[2] = count;
    }
    /// 銀の枚数を設定
    #[inline]
    pub fn set_silver(&mut self, count: u8) {
        self.counts[3] = count;
    }
    /// 金の枚数を設定
    #[inline]
    pub fn set_gold(&mut self, count: u8) {
        self.counts[4] = count;
    }
    /// 角の枚数を設定
    #[inline]
    pub fn set_bishop(&mut self, count: u8) {
        self.counts[5] = count;
    }
    /// 飛の枚数を設定
    #[inline]
    pub fn set_rook(&mut self, count: u8) {
        self.counts[6] = count;
    }
}

/// 手駒として持てる駒種の配列
pub const HAND_PIECE_TYPES: [PieceType; 7] = [
    PieceType::Pawn,
    PieceType::Lance,
    PieceType::Knight,
    PieceType::Silver,
    PieceType::Gold,
    PieceType::Bishop,
    PieceType::Rook,
];

/// 盤上の駒種（King除外）
pub const BOARD_PIECE_TYPES: [PieceType; 13] = [
    PieceType::Pawn,
    PieceType::Lance,
    PieceType::Knight,
    PieceType::Silver,
    PieceType::Gold,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::ProPawn,
    PieceType::ProLance,
    PieceType::ProKnight,
    PieceType::ProSilver,
    PieceType::Horse,
    PieceType::Dragon,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_basics() {
        let sq = Square::new(4, 8); // 5九
        assert_eq!(sq.file(), 4);
        assert_eq!(sq.rank(), 8);
        assert_eq!(sq.index(), 44);
    }

    #[test]
    fn test_square_inverse() {
        let sq = Square::new(0, 0); // 1一
        let inv = sq.inverse();
        assert_eq!(inv, Square::new(8, 8)); // 9九
    }

    #[test]
    fn test_square_mirror_file() {
        let sq = Square::new(0, 4); // 1五
        let mir = sq.mirror_file();
        assert_eq!(mir, Square::new(8, 4)); // 9五
    }

    #[test]
    fn test_color_opponent() {
        assert_eq!(Color::Black.opponent(), Color::White);
        assert_eq!(Color::White.opponent(), Color::Black);
    }

    #[test]
    fn test_hand() {
        let mut hand = Hand::EMPTY;
        hand.set_pawn(5);
        hand.set_lance(2);
        assert_eq!(hand.pawn(), 5);
        assert_eq!(hand.lance(), 2);
        assert_eq!(hand.knight(), 0);
    }
}
