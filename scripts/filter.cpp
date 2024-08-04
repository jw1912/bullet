#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <ostream>

enum Square {
  // clang-format off
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,

    N_SQUARES,
  // clang-format on
};

enum MoveFlag {
  QUIET,
  PAWN_DOUBLE_MOVE,
  A_SIDE_CASTLE,
  H_SIDE_CASTLE,
  CAPTURE,
  EN_PASSANT = 5,

  DONT_USE_1 = 6,
  DONT_USE_2 = 7,

  KNIGHT_PROMOTION = 8,
  BISHOP_PROMOTION,
  ROOK_PROMOTION,
  QUEEN_PROMOTION,
  KNIGHT_PROMOTION_CAPTURE,
  BISHOP_PROMOTION_CAPTURE,
  ROOK_PROMOTION_CAPTURE,
  QUEEN_PROMOTION_CAPTURE
};

class Move {
public:
  Move() = default;
  Move(Square from, Square to, MoveFlag flag);
  explicit Move(uint16_t data);

  Square GetFrom() const;
  Square GetTo() const;
  MoveFlag GetFlag() const;

  bool IsPromotion() const;
  bool IsCapture() const;
  bool IsCastle() const;

  constexpr bool operator==(const Move &rhs) const {
    return (data == rhs.data);
  }

  constexpr bool operator!=(const Move &rhs) const {
    return (data != rhs.data);
  }

  static const Move Uninitialized;

  friend std::ostream &operator<<(std::ostream &os, Move m);

private:
  void SetFrom(Square from);
  void SetTo(Square to);
  void SetFlag(MoveFlag flag);

  uint16_t data;
};

static_assert(std::is_trivial_v<Move>);

inline const Move Move::Uninitialized = Move(
    static_cast<Square>(0), static_cast<Square>(0), static_cast<MoveFlag>(0));

constexpr int CAPTURE_MASK = 1 << 14;   // 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
constexpr int PROMOTION_MASK = 1 << 15; // 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
constexpr int FROM_MASK = 0b111111;     // 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1
constexpr int TO_MASK = 0b111111 << 6;  // 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
constexpr int FLAG_MASK = 0b1111 << 12; // 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0

Move::Move(Square from, Square to, MoveFlag flag) : data(0) {
  assert(from < 64);
  assert(to < 64);

  SetFrom(from);
  SetTo(to);
  SetFlag(flag);
}

Move::Move(uint16_t data_) : data(data_) {}

Square Move::GetFrom() const { return static_cast<Square>(data & FROM_MASK); }

Square Move::GetTo() const {
  return static_cast<Square>((data & TO_MASK) >> 6);
}

MoveFlag Move::GetFlag() const {
  return static_cast<MoveFlag>((data & FLAG_MASK) >> 12);
}

bool Move::IsPromotion() const { return ((data & PROMOTION_MASK) != 0); }

bool Move::IsCapture() const { return ((data & CAPTURE_MASK) != 0); }

bool Move::IsCastle() const {
  return GetFlag() == A_SIDE_CASTLE || GetFlag() == H_SIDE_CASTLE;
}

std::ostream &operator<<(std::ostream &os, Move m) {
  // unused
}

void Move::SetFrom(Square from) {
  data &= ~FROM_MASK;
  data |= from;
}

void Move::SetTo(Square to) {
  data &= ~TO_MASK;
  data |= to << 6;
}

void Move::SetFlag(MoveFlag flag) {
  data &= ~FLAG_MASK;
  data |= flag << 12;
}

struct training_data {
  uint64_t occ;
  std::array<uint8_t, 16> pcs;
  int16_t score;
  int8_t result;
  uint8_t ksq;
  uint8_t opp_ksq;

  uint8_t padding[1];
  Move best_move;
};

int main(int argc, char **argv) {
  std::ifstream input(argv[1], std::ios::binary);
  std::ofstream output(argv[2], std::ios::binary);

  training_data data;
  size_t input_data_count = 0;
  size_t output_data_count = 0;
  while (input.read((char *)(&data), sizeof(training_data))) {
    input_data_count++;
    if (!data.best_move.IsCapture() && !data.best_move.IsPromotion()) {
      output_data_count++;
      output.write((char *)(&data), sizeof(training_data));
    }
    if (input_data_count % (1024 * 1024) == 0) {
      std::cout << "Read " << input_data_count << " points. Filtered "
                << output_data_count << " points\n";
    }
  }
  std::cout << "Read " << input_data_count << " points. Filtered "
            << output_data_count << " points\n";
  std::cout << "Complete" << std::endl;
}
