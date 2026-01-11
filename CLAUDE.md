# CLAUDE.md for bullet-shogi

bullet (https://github.com/jw1912/bullet) の将棋対応フォーク。

## プロジェクト概要

- **目的**: 将棋NNUE学習のための高速トレーナー
- **ベース**: bullet (チェス用NNUE学習ライブラリ)
- **言語**: Rust + CUDA

## 関連リポジトリ

```
~/development/
├── shogi/                              # 本体（将棋エンジン）
│   └── packages/rust-core/
│       ├── crates/engine-core/src/nnue/  # NNUE推論実装
│       └── memo/
│           ├── bullet/                 # bulletオリジナル（参照用）
│           └── nnue-pytorch/           # PackedSfen実装の参照先
└── bullet-shogi/                       # このリポジトリ
```

## ドキュメント

```
docs/
├── bullet-shogi-implementation.md      # メイン実装ガイド ★まずこれを読む
├── bullet-shogi-packed-sfen-spec.md    # PackedSfen Huffman符号化仕様
├── bullet-shogi-data-structures.md     # ShogiBoard データ構造設計
└── (bulletオリジナルのドキュメント)
```

## 実装タスク

| # | タスク | ファイル |
|---|--------|----------|
| 1 | ShogiHalfKA_hm | `crates/bullet_lib/src/game/inputs/shogi_halfka.rs` |
| 2 | ShogiBoard | `crates/bullet_lib/src/game/inputs/shogi_board.rs` |
| 3 | DataLoader | `crates/bullet_lib/src/value/loader/packed_sfen.rs` |
| 4 | 学習スクリプト | `examples/shogi_simple.rs` |
| 5 | 出力変換 | `examples/export_nnue.rs` |

## 参照すべきファイル

### PackedSfen デコード実装 ★重要

```
~/development/shogi/packages/rust-core/memo/nnue-pytorch/training_data_loader.cpp
```

主要な関数:
- `Position::set_from_packed_sfen()` - Huffman復号
- `HuffmanedPiece` - Huffman符号テーブル

### rust-core の NNUE 実装

```
~/development/shogi/packages/rust-core/crates/engine-core/src/nnue/
├── features/half_ka_hm.rs    # HalfKA_hm 特徴量（移植元）
├── bona_piece_halfka.rs      # BonaPiece 計算
├── bona_piece.rs             # BonaPiece 定義
└── leb128.rs                 # LEB128 圧縮
```

## ビルド

```bash
# CUDA バックエンド（デフォルト）
cargo build --release

# CPU バックエンド（GPUなし環境）
cargo build --release --features cpu --no-default-features

# 学習実行（将棋版完成後）
cargo run --release --example shogi_simple
```

## 開発方針

### コーディング規約

- Rust 2024 edition
- `cargo fmt` でフォーマット
- `cargo clippy -- -D warnings` で警告チェック

### ブランチ

- `main`: bulletオリジナル（upstream追従用）
- `shogi-support`: 将棋対応開発ブランチ

### コミット

将棋固有の変更は `shogi-support` ブランチで行う。
upstream の更新を取り込む際は `main` にマージしてから rebase。

```bash
# upstream の更新を取り込む
git fetch upstream
git checkout main
git merge upstream/main
git checkout shogi-support
git rebase main
```

## テスト

```bash
# 全テスト
cargo test

# 将棋固有テスト（実装後）
cargo test shogi
```

## 言語設定

ユーザーへの返答は日本語で行うこと。
