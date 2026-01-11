/*
bullet quantised.bin -> やねうら王形式 (.nnue) 変換ツール

使用方法:
    cargo run --release --example export_shogi_nnue -- <input> <output>

例:
    cargo run --release --example export_shogi_nnue -- \
        checkpoints/shogi-halfka-hm-10/quantised.bin \
        eval/shogi.nnue
*/

use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

// =============================================================================
// ネットワーク構成（shogi_simple.rs と同じ）
// =============================================================================

const L1_SIZE: usize = 256;
const L2_SIZE: usize = 32;
const L3_SIZE: usize = 32;
const INPUT_SIZE: usize = 73305;

// =============================================================================
// LEB128 エンコード
// =============================================================================

/// LEB128 (signed) エンコード
fn write_leb128(writer: &mut impl Write, value: i32) -> std::io::Result<()> {
    let mut val = value;
    loop {
        let byte = (val & 0x7F) as u8;
        val >>= 7;

        let more = !((val == 0 && (byte & 0x40) == 0) || (val == -1 && (byte & 0x40) != 0));

        if more {
            writer.write_all(&[byte | 0x80])?;
        } else {
            writer.write_all(&[byte])?;
            break;
        }
    }
    Ok(())
}

// =============================================================================
// bullet quantised.bin の構造
// =============================================================================

/// bullet の quantised.bin を読み込み
fn read_quantised_bin(path: &str) -> std::io::Result<BulletParams> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut params = BulletParams::default();

    // bullet の SavedFormat 順に読み込み
    // l0w: L1_SIZE x INPUT_SIZE (column-major)
    let l0w_size = L1_SIZE * INPUT_SIZE;
    params.l0_weights.resize(l0w_size, 0);
    read_i16_array(&mut reader, &mut params.l0_weights)?;

    // l0b: L1_SIZE
    params.l0_bias.resize(L1_SIZE, 0);
    read_i16_array(&mut reader, &mut params.l0_bias)?;

    // l1w: L2_SIZE x (2*L1_SIZE) (column-major)
    let l1w_size = L2_SIZE * 2 * L1_SIZE;
    params.l1_weights.resize(l1w_size, 0);
    read_i16_array(&mut reader, &mut params.l1_weights)?;

    // l1b: L2_SIZE
    params.l1_bias.resize(L2_SIZE, 0);
    read_i16_array(&mut reader, &mut params.l1_bias)?;

    // l2w: L3_SIZE x L2_SIZE (column-major)
    let l2w_size = L3_SIZE * L2_SIZE;
    params.l2_weights.resize(l2w_size, 0);
    read_i16_array(&mut reader, &mut params.l2_weights)?;

    // l2b: L3_SIZE
    params.l2_bias.resize(L3_SIZE, 0);
    read_i16_array(&mut reader, &mut params.l2_bias)?;

    // outw: 1 x L3_SIZE
    params.out_weights.resize(L3_SIZE, 0);
    read_i16_array(&mut reader, &mut params.out_weights)?;

    // outb: 1
    params.out_bias.resize(1, 0);
    read_i16_array(&mut reader, &mut params.out_bias)?;

    Ok(params)
}

/// i16 配列を読み込み
fn read_i16_array(reader: &mut impl Read, arr: &mut [i16]) -> std::io::Result<()> {
    let mut buf = vec![0u8; arr.len() * 2];
    reader.read_exact(&mut buf)?;
    for (i, chunk) in buf.chunks_exact(2).enumerate() {
        arr[i] = i16::from_le_bytes([chunk[0], chunk[1]]);
    }
    Ok(())
}

#[derive(Default)]
struct BulletParams {
    l0_weights: Vec<i16>,
    l0_bias: Vec<i16>,
    l1_weights: Vec<i16>,
    l1_bias: Vec<i16>,
    l2_weights: Vec<i16>,
    l2_bias: Vec<i16>,
    out_weights: Vec<i16>,
    out_bias: Vec<i16>,
}

// =============================================================================
// やねうら王形式 (.nnue) の書き出し
// =============================================================================

/// やねうら王形式で書き出し
fn write_nnue(path: &str, params: &BulletParams) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // ヘッダー（簡易版）
    // 実際のやねうら王形式はより複雑なヘッダーを持つ
    // ここでは raw 形式で出力

    println!("Writing L0 weights (LEB128 compressed)...");
    // L0 weights: LEB128 圧縮
    // bullet は column-major、やねうら王は row-major の場合があるので注意
    // ここでは column-major のまま出力
    for &w in &params.l0_weights {
        write_leb128(&mut writer, i32::from(w))?;
    }

    println!("Writing L0 bias...");
    // L0 bias: raw i16
    for &b in &params.l0_bias {
        writer.write_all(&b.to_le_bytes())?;
    }

    println!("Writing L1 weights...");
    // L1 weights: raw i16
    for &w in &params.l1_weights {
        writer.write_all(&w.to_le_bytes())?;
    }

    println!("Writing L1 bias...");
    // L1 bias: raw i16
    for &b in &params.l1_bias {
        writer.write_all(&b.to_le_bytes())?;
    }

    println!("Writing L2 weights...");
    // L2 weights: raw i16
    for &w in &params.l2_weights {
        writer.write_all(&w.to_le_bytes())?;
    }

    println!("Writing L2 bias...");
    // L2 bias: raw i16
    for &b in &params.l2_bias {
        writer.write_all(&b.to_le_bytes())?;
    }

    println!("Writing output weights...");
    // Output weights: raw i16
    for &w in &params.out_weights {
        writer.write_all(&w.to_le_bytes())?;
    }

    println!("Writing output bias...");
    // Output bias: raw i16
    for &b in &params.out_bias {
        writer.write_all(&b.to_le_bytes())?;
    }

    writer.flush()?;
    Ok(())
}

/// Column-major から Row-major に転置
#[allow(dead_code)]
fn transpose(src: &[i16], rows: usize, cols: usize) -> Vec<i16> {
    let mut dst = vec![0i16; src.len()];
    for r in 0..rows {
        for c in 0..cols {
            dst[r * cols + c] = src[c * rows + r];
        }
    }
    dst
}

// =============================================================================
// メイン
// =============================================================================

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: {} <input.bin> <output.nnue>", args[0]);
        eprintln!();
        eprintln!("Convert bullet quantised.bin to NNUE format");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} checkpoints/shogi-halfka-hm-10/quantised.bin eval/shogi.nnue", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    println!("=== Shogi NNUE Export Tool ===");
    println!("Input:  {}", input_path);
    println!("Output: {}", output_path);
    println!();

    println!("Network configuration:");
    println!("  Input: {} (HalfKA_hm)", INPUT_SIZE);
    println!("  L1: {}", L1_SIZE);
    println!("  L2: {}", L2_SIZE);
    println!("  L3: {}", L3_SIZE);
    println!();

    // 読み込み
    println!("Reading {}...", input_path);
    let params = match read_quantised_bin(input_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error reading input file: {}", e);
            std::process::exit(1);
        }
    };

    println!("Read {} parameters:", input_path);
    println!("  L0 weights: {} values", params.l0_weights.len());
    println!("  L0 bias: {} values", params.l0_bias.len());
    println!("  L1 weights: {} values", params.l1_weights.len());
    println!("  L1 bias: {} values", params.l1_bias.len());
    println!("  L2 weights: {} values", params.l2_weights.len());
    println!("  L2 bias: {} values", params.l2_bias.len());
    println!("  Out weights: {} values", params.out_weights.len());
    println!("  Out bias: {} values", params.out_bias.len());
    println!();

    // 書き出し
    println!("Writing {}...", output_path);
    if let Err(e) = write_nnue(output_path, &params) {
        eprintln!("Error writing output file: {}", e);
        std::process::exit(1);
    }

    println!();
    println!("Done!");
    println!();
    println!("Note: This is a simplified export. For full YaneuraOu compatibility,");
    println!("you may need to adjust the header format and weight layout.");
}
