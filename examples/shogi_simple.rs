/*
Shogi NNUE Training Script

Usage:
    cargo run --release --example shogi_simple -- [OPTIONS]

Options:
    --arch <ARCH>       Architecture preset (default: 256x2-32-32)
                        Presets: 256x2-32-32, 512x2-8-96, 512x2-32-32, 1024x2-8-32
    --l1 <SIZE>         L1 (accumulator) size (overrides preset)
    --l2 <SIZE>         L2 (hidden layer 1) size
    --l3 <SIZE>         L3 (hidden layer 2) size
    --data <PATH>       Training data path (comma-separated for multiple files)
    --batch-size <N>    Batch size (default: 16384)
    --superbatches <N>  Number of superbatches (default: 100)
    --lr <RATE>         Initial learning rate (default: 0.001)
    --wdl <LAMBDA>      WDL lambda (default: 0.75)
    --scale <N>         Eval scale (default: 1020)
                        FV_SCALE = QA*QB/scale (rounded)
                        QA=255 (SCReLU): 16320/scale -> 510->32, 1020->16
                        QA=127 (CReLU):  8128/scale  -> 508->16, 254->32
                        Note: Default (QA=255, scale=1020) -> FV_SCALE=16
                        For FV_SCALE=32: --qa 255 --scale 510 or --qa 127 --scale 254
    --save-rate <N>     Save interval in superbatches (default: 10)
    --threads <N>       Number of threads (default: 4)
    --output <DIR>      Output directory (default: checkpoints)
    --net-id <NAME>     Network ID (default: shogi-halfka-hm)
    --weight-decay <F>  Weight decay (default: 0.01)

Examples:
    # Train with default settings
    cargo run --release --example shogi_simple -- --data data/train.bin

    # Train with 512x2-8-96 architecture
    cargo run --release --example shogi_simple -- --arch 512x2-8-96 --data data/train.bin

    # Train with custom sizes
    cargo run --release --example shogi_simple -- --l1 1024 --l2 16 --l3 64 --data data/train.bin
*/

use std::path::PathBuf;

use bullet_lib::{
    game::inputs::{ShogiHalfKA_hm, ShogiHalfKP, SparseInputType},
    nn::optimiser::{self, AdamWParams, RAdamParams, RangerParams},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};
use clap::{Parser, ValueEnum};

/// Feature set selection
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum FeatureSet {
    /// HalfKA_hm - Half-Mirrored King-All (73,305 dimensions)
    #[default]
    HalfkaHm,
    /// HalfKP - King-Piece (125,388 dimensions, no mirror)
    HalfKP,
}

/// Output format selection
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum OutputFormat {
    /// bullet format: all i16 (l0w, l0b, l1w, l1b, l2w, l2b, outw, outb)
    #[default]
    Bullet,
    /// rust-core format: L0 i16, L1-Out biases i32 + weights i8, with NNUE header
    RustCore,
}

/// Activation function selection
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum ActivationType {
    /// SCReLU - Squared Clipped ReLU: y = clamp(x, 0, qa)²
    /// Higher expressiveness, used in modern Stockfish
    #[default]
    Screlu,
    /// CReLU - Clipped ReLU: y = clamp(x, 0, qa)
    /// Traditional activation, used in YaneuraOu/Suisho
    Crelu,
}

// =============================================================================
// CLI Arguments
// =============================================================================

/// Optimizer selection
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum OptimizerType {
    /// AdamW - fast convergence but may be unstable with sparse inputs
    AdamW,
    /// RAdam - Rectified Adam, more stable
    RAdam,
    /// Ranger - RAdam + Lookahead (recommended by nnue-pytorch)
    #[default]
    Ranger,
}

#[derive(Parser, Debug)]
#[command(name = "shogi_simple")]
#[command(about = "Shogi NNUE training script")]
struct Args {
    /// Feature set (halfka-hm or halfkp)
    /// halfka-hm: HalfKA_hm (73,305 dims, Half-Mirror) - nnue-pytorch compatible
    /// halfkp: HalfKP (125,388 dims, no mirror) - classic NNUE
    #[arg(long, value_enum, default_value = "halfka-hm")]
    features: FeatureSet,

    /// Output format (bullet or rust-core)
    /// bullet: all i16, no header (default)
    /// rust-core: NNUE header + L0 i16 + L1-Out biases i32 + weights i8
    #[arg(long, value_enum, default_value = "bullet")]
    output_format: OutputFormat,

    /// Activation function (screlu or crelu)
    /// screlu: Squared Clipped ReLU - higher expressiveness (default)
    /// crelu: Clipped ReLU - traditional, used in YaneuraOu/Suisho
    #[arg(long, value_enum, default_value = "screlu")]
    activation: ActivationType,

    /// Architecture preset
    /// Presets: 256x2-32-32, 512x2-8-96, 512x2-32-32, 1024x2-8-32
    #[arg(long, default_value = "256x2-32-32")]
    arch: String,

    /// Optimizer (adamw, radam, ranger)
    /// ranger = RAdam + Lookahead (same as nnue-pytorch recommendation)
    #[arg(long, value_enum, default_value = "ranger")]
    optimizer: OptimizerType,

    /// L1 (accumulator) size (overrides preset)
    #[arg(long)]
    l1: Option<usize>,

    /// L2 (hidden layer 1) size
    #[arg(long)]
    l2: Option<usize>,

    /// L3 (hidden layer 2) size
    #[arg(long)]
    l3: Option<usize>,

    /// Training data path (comma-separated for multiple files)
    #[arg(long, default_value = "data/train.bin")]
    data: String,

    /// Batch size
    #[arg(long, default_value = "16384")]
    batch_size: usize,

    /// Number of superbatches
    #[arg(long, default_value = "100")]
    superbatches: usize,

    /// Initial learning rate
    #[arg(long, default_value = "0.001")]
    lr: f32,

    /// WDL lambda (0.0=eval only, 1.0=game result only)
    #[arg(long, default_value = "0.75")]
    wdl: f32,

    /// Eval scale for training target sigmoid(score / scale).
    /// FV_SCALE = QA*QB/scale (rounded).
    /// Recommended divisors for exact FV_SCALE:
    ///   QA=255 (SCReLU): 510->32, 1020->16, 340->48
    ///   QA=127 (CReLU):  508->16, 254->32, 1016->8
    /// Note: Default (QA=255, scale=1020) gives FV_SCALE=16.
    /// For FV_SCALE=32: use --qa 255 --scale 510  (SCReLU)
    ///                  or  --qa 127 --scale 254  (CReLU)
    #[arg(long, default_value = "1020")]
    scale: i32,

    /// Save interval (superbatches)
    #[arg(long, default_value = "10")]
    save_rate: usize,

    /// Number of threads
    #[arg(long, default_value = "4")]
    threads: usize,

    /// Output directory
    #[arg(long, default_value = "checkpoints")]
    output: PathBuf,

    /// Network ID
    #[arg(long, default_value = "shogi-halfka-hm")]
    net_id: String,

    /// Quantization factor QA (for L0)
    #[arg(long, default_value = "255")]
    qa: i16,

    /// Quantization factor QB (for later layers)
    #[arg(long, default_value = "64")]
    qb: i16,

    /// Weight decay (L2 regularization)
    #[arg(long, default_value = "0.01")]
    weight_decay: f32,
}

// =============================================================================
// Architecture Definition
// =============================================================================

#[derive(Debug, Clone, Copy)]
struct Architecture {
    l1: usize, // Accumulator size
    l2: usize, // Hidden layer 1 size
    l3: usize, // Hidden layer 2 size
}

impl Architecture {
    /// Get architecture from preset name
    fn from_preset(name: &str) -> Option<Self> {
        match name {
            "256x2-32-32" => Some(Self { l1: 256, l2: 32, l3: 32 }),
            "512x2-8-96" => Some(Self { l1: 512, l2: 8, l3: 96 }),
            "512x2-32-32" => Some(Self { l1: 512, l2: 32, l3: 32 }),
            "1024x2-8-32" => Some(Self { l1: 1024, l2: 8, l3: 32 }),
            "1024x2-16-64" => Some(Self { l1: 1024, l2: 16, l3: 64 }),
            _ => None,
        }
    }

    /// List of available presets
    fn available_presets() -> &'static [&'static str] {
        &["256x2-32-32", "512x2-8-96", "512x2-32-32", "1024x2-8-32", "1024x2-16-64"]
    }

    /// Display string
    fn display(&self) -> String {
        format!("{}x2-{}-{}", self.l1, self.l2, self.l3)
    }
}

// =============================================================================
// SIMD Padding Utilities
// =============================================================================

/// 32バイトアライメントにパディング
fn pad32(size: usize) -> usize {
    size.div_ceil(32) * 32
}

/// rust-core 用に重みをパディング
///
/// rust-core は SIMD 最適化のため、各層の入力次元を32の倍数にパディングする。
/// 例: 入力次元8 → パディング後32 (24個の0を追加)
///
/// # Arguments
/// * `weights` - row-major の重み [out_dim * in_dim]
/// * `out_dim` - 出力次元
/// * `in_dim` - 入力次元 (パディング前)
fn pad_weights_for_simd(weights: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let padded_in_dim = pad32(in_dim);

    // パディング不要な場合はそのまま返す
    if padded_in_dim == in_dim {
        return weights.to_vec();
    }

    let mut result = vec![0.0f32; out_dim * padded_in_dim];

    for o in 0..out_dim {
        for i in 0..in_dim {
            result[o * padded_in_dim + i] = weights[o * in_dim + i];
        }
        // 残りは0で埋める (既にvec![0.0; ...]で初期化済み)
    }

    result
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let args = Args::parse();

    // Determine architecture
    let mut arch = Architecture::from_preset(&args.arch).unwrap_or_else(|| {
        eprintln!("Unknown architecture preset: {}", args.arch);
        eprintln!("Available presets: {:?}", Architecture::available_presets());
        std::process::exit(1);
    });

    // Override with individual settings
    if let Some(l1) = args.l1 {
        arch.l1 = l1;
    }
    if let Some(l2) = args.l2 {
        arch.l2 = l2;
    }
    if let Some(l3) = args.l3 {
        arch.l3 = l3;
    }

    let l1_size = arch.l1;
    let l2_size = arch.l2;
    let l3_size = arch.l3;

    // Quantization factors
    let qa = args.qa;
    let qb = args.qb;

    // Feature set info
    let (feature_name, input_size) = match args.features {
        FeatureSet::HalfkaHm => ("HalfKA_hm", ShogiHalfKA_hm.num_inputs()),
        FeatureSet::HalfKP => ("HalfKP", ShogiHalfKP.num_inputs()),
    };

    // Optimizer name
    let optimizer_name = match args.optimizer {
        OptimizerType::AdamW => "AdamW",
        OptimizerType::RAdam => "RAdam",
        OptimizerType::Ranger => "Ranger (RAdam + Lookahead)",
    };

    // Activation function name
    let activation_name = match args.activation {
        ActivationType::Screlu => "SCReLU",
        ActivationType::Crelu => "CReLU",
    };

    // Validate QA and activation combination
    let recommended_qa = match args.activation {
        ActivationType::Screlu => 255,
        ActivationType::Crelu => 127,
    };
    if qa != recommended_qa {
        eprintln!("WARNING: QA={} is not recommended for {} activation.", qa, activation_name);
        eprintln!("         Recommended: --qa {} --activation {}",
            recommended_qa,
            match args.activation {
                ActivationType::Screlu => "screlu",
                ActivationType::Crelu => "crelu",
            }
        );
        eprintln!("         Using non-standard QA may cause evaluation scale mismatch.");
        eprintln!();
        eprint!("Continue anyway? [y/N]: ");
        use std::io::{self, Write};
        io::stderr().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim().to_lowercase();
        if input != "y" && input != "yes" {
            eprintln!("Aborted.");
            std::process::exit(1);
        }
        eprintln!();
    }

    // Print configuration
    println!("=== Shogi NNUE Training ===");
    println!("Features: {} ({} dimensions)", feature_name, input_size);
    println!("Architecture: {} (L1={}, L2={}, L3={})", arch.display(), l1_size, l2_size, l3_size);
    println!("Network: {} -> {}x2 -> {} -> {} -> 1", input_size, l1_size, l2_size, l3_size);
    println!("Activation: {}", activation_name);
    println!("Optimizer: {}", optimizer_name);
    println!("Weight decay: {}", args.weight_decay);
    println!("Scale: {}", args.scale);
    println!("Quantization: QA={}, QB={}", qa, qb);
    println!("Batch size: {}", args.batch_size);
    println!("Superbatches: {}", args.superbatches);
    println!("Learning rate: {}", args.lr);
    println!("WDL lambda: {}", args.wdl);
    println!("Save rate: {}", args.save_rate);
    println!("Threads: {}", args.threads);
    println!("Output: {}", args.output.display());
    println!("Net ID: {}", args.net_id);
    println!("Data: {}", args.data);
    println!("===========================");

    // Training schedule
    let schedule = TrainingSchedule {
        net_id: args.net_id,
        eval_scale: args.scale as f32,
        steps: TrainingSteps {
            batch_size: args.batch_size,
            batches_per_superbatch: 6104, // ~100M positions/superbatch
            start_superbatch: 1,
            end_superbatch: args.superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: args.wdl },
        lr_scheduler: lr::StepLR { start: args.lr, gamma: 0.3, step: 30 },
        save_rate: args.save_rate,
    };

    // Local settings
    let output_dir = args.output.to_str().unwrap_or("checkpoints");
    let settings =
        LocalSettings { threads: args.threads, test_set: None, output_directory: output_dir, batch_queue_size: 64 };

    // Data loader
    let data_files: Vec<&str> = args.data.split(',').collect();
    let data_loader = DirectSequentialDataLoader::new(&data_files);

    // SavedFormat configuration
    // This directly outputs the final format for your engine.
    // Customize as needed:
    //   - .transpose() to change matrix layout
    //   - SavedFormat::custom(bytes) to add headers
    //   - .transform(|store, vals| ...) for custom transformations
    let save_format: Vec<SavedFormat> = match args.output_format {
        OutputFormat::Bullet => {
            // bullet format: all i16 (default)
            vec![
                SavedFormat::id("l0w").round().quantise::<i16>(qa),
                SavedFormat::id("l0b").round().quantise::<i16>(qa),
                SavedFormat::id("l1w").round().quantise::<i16>(qb),
                SavedFormat::id("l1b").round().quantise::<i16>(qa * qb),
                SavedFormat::id("l2w").round().quantise::<i16>(qb),
                SavedFormat::id("l2b").round().quantise::<i16>(qa * qb),
                SavedFormat::id("outw").round().quantise::<i16>(qb),
                SavedFormat::id("outb").round().quantise::<i16>(qa * qb),
            ]
        }
        OutputFormat::RustCore => {
            // rust-core format: NNUE header + L0 i16 + L1-Out biases i32 + weights i8
            //
            // File layout:
            // - Header: version (u32), hash (u32), arch_len (u32), arch_string
            // - FeatureTransformer layer hash (u32)
            // - L0: biases i16[L1], weights i16[INPUT×L1]
            // - Network layer hash (u32)
            // - L1: biases i32[L2], weights i8[L2×(L1*2)]
            // - L2: biases i32[L3], weights i8[L3×L2]
            // - Output: biases i32[1], weights i8[1×L3]

            // NNUE version (YaneuraOu/Stockfish compatible)
            const NNUE_VERSION: u32 = 0x7AF32F16;

            // Build architecture string with features and activation info
            // Include fv_scale metadata for rust-core inference
            // FV_SCALE = (QA × QB) / scale (四捨五入)
            //
            // 重要: l2/l3 を明示的に含める（rust-core がパースできるようにするため）
            // rust-core は AffineTransform[...] パターンがない場合、l2/l3 フィールドを使用
            let qa_qb = i32::from(qa) * i32::from(qb);
            let fv_scale = (qa_qb + args.scale / 2) / args.scale;
            let arch_str = format!(
                "Features={}[{}->{}x2]{},fv_scale={},l2={},l3={},qa={},qb={},scale={}",
                feature_name,
                input_size,
                l1_size,
                if matches!(args.activation, ActivationType::Screlu) { "-SCReLU" } else { "" },
                fv_scale,
                l2_size,
                l3_size,
                qa,
                qb,
                args.scale
            );
            let arch_bytes = arch_str.as_bytes();

            // Build header
            let mut header = Vec::new();
            header.extend_from_slice(&NNUE_VERSION.to_le_bytes());
            header.extend_from_slice(&0u32.to_le_bytes()); // hash (dummy)
            header.extend_from_slice(&(arch_bytes.len() as u32).to_le_bytes());
            header.extend_from_slice(arch_bytes);

            // Layer hashes (dummy values, rust-core skips validation)
            let ft_hash = 0u32.to_le_bytes().to_vec();
            let network_hash = 0u32.to_le_bytes().to_vec();

            vec![
                // Header
                SavedFormat::custom(header),
                // FeatureTransformer layer hash
                SavedFormat::custom(ft_hash),
                // L0: biases first, then weights (rust-core order)
                SavedFormat::id("l0b").round().quantise::<i16>(qa),
                SavedFormat::id("l0w").round().quantise::<i16>(qa),
                // Network layer hash
                SavedFormat::custom(network_hash),
                // L1-Output層の重みは .transpose() で row-major に変換
                // 理由: Stockfish/nnue-pytorch は row-major で推論する
                // bullet 内部は column-major だが、これは GPU (cuBLAS) 最適化のため
                // 変換コストは出力時の1回のみで、学習効率には影響しない
                //
                // 重要: rust-core は SIMD 最適化のため 32バイトアライメントを要求
                // 各層の入力次元を pad32() でパディングする必要がある
                //
                // L1: biases i32, weights i8 (row-major, padded)
                // 入力次元: l1*2 → pad32(l1*2)
                SavedFormat::id("l1b").round().quantise::<i32>(i32::from(qa) * i32::from(qb)),
                SavedFormat::id("l1w").transpose().transform({
                    let out_dim = l2_size;
                    let in_dim = l1_size * 2;
                    move |_, vals| pad_weights_for_simd(&vals, out_dim, in_dim)
                }).round().quantise::<i8>(qb),
                // L2: biases i32, weights i8 (row-major, padded)
                // 入力次元: l2 → pad32(l2)
                SavedFormat::id("l2b").round().quantise::<i32>(i32::from(qa) * i32::from(qb)),
                SavedFormat::id("l2w").transpose().transform({
                    let out_dim = l3_size;
                    let in_dim = l2_size;
                    move |_, vals| pad_weights_for_simd(&vals, out_dim, in_dim)
                }).round().quantise::<i8>(qb),
                // Output: biases i32, weights i8 (row-major, padded)
                // 入力次元: l3 → pad32(l3)
                SavedFormat::id("outb").round().quantise::<i32>(i32::from(qa) * i32::from(qb)),
                SavedFormat::id("outw").transpose().transform({
                    let out_dim = 1;
                    let in_dim = l3_size;
                    move |_, vals| pad_weights_for_simd(&vals, out_dim, in_dim)
                }).round().quantise::<i8>(qb),
            ]
        }
    };

    // Network builder macro with SCReLU activation
    macro_rules! build_trainer_screlu {
        ($opt:expr, $input:expr) => {
            ValueTrainerBuilder::default()
                .dual_perspective()
                .optimiser($opt)
                .inputs($input)
                .save_format(&save_format)
                .loss_fn(|output, target| output.sigmoid().squared_error(target))
                .build(|builder, stm_inputs, ntm_inputs| {
                    let l0 = builder.new_affine("l0", input_size, l1_size);
                    let l1 = builder.new_affine("l1", 2 * l1_size, l2_size);
                    let l2 = builder.new_affine("l2", l2_size, l3_size);
                    let out = builder.new_affine("out", l3_size, 1);

                    let stm_hidden = l0.forward(stm_inputs).screlu();
                    let ntm_hidden = l0.forward(ntm_inputs).screlu();
                    let combined = stm_hidden.concat(ntm_hidden);

                    let hidden1 = l1.forward(combined).screlu();
                    let hidden2 = l2.forward(hidden1).screlu();

                    out.forward(hidden2)
                })
        };
    }

    // Network builder macro with CReLU (Clipped ReLU) activation
    macro_rules! build_trainer_crelu {
        ($opt:expr, $input:expr) => {
            ValueTrainerBuilder::default()
                .dual_perspective()
                .optimiser($opt)
                .inputs($input)
                .save_format(&save_format)
                .loss_fn(|output, target| output.sigmoid().squared_error(target))
                .build(|builder, stm_inputs, ntm_inputs| {
                    let l0 = builder.new_affine("l0", input_size, l1_size);
                    let l1 = builder.new_affine("l1", 2 * l1_size, l2_size);
                    let l2 = builder.new_affine("l2", l2_size, l3_size);
                    let out = builder.new_affine("out", l3_size, 1);

                    let stm_hidden = l0.forward(stm_inputs).crelu();
                    let ntm_hidden = l0.forward(ntm_inputs).crelu();
                    let combined = stm_hidden.concat(ntm_hidden);

                    let hidden1 = l1.forward(combined).crelu();
                    let hidden2 = l2.forward(hidden1).crelu();

                    out.forward(hidden2)
                })
        };
    }

    // Run training macro (to reduce duplication across feature sets and activations)
    macro_rules! run_training {
        ($input:expr, screlu) => {{
            let weight_decay = args.weight_decay;
            match args.optimizer {
                OptimizerType::AdamW => {
                    let mut trainer = build_trainer_screlu!(optimiser::AdamW, $input);
                    trainer.optimiser.set_params(AdamWParams { decay: weight_decay, ..Default::default() });
                    trainer.run(&schedule, &settings, &data_loader);
                }
                OptimizerType::RAdam => {
                    let mut trainer = build_trainer_screlu!(optimiser::RAdam, $input);
                    let params: RAdamParams = RAdamParams { decay: weight_decay, ..Default::default() };
                    trainer.optimiser.set_params(params.into());
                    trainer.run(&schedule, &settings, &data_loader);
                }
                OptimizerType::Ranger => {
                    let mut trainer = build_trainer_screlu!(optimiser::Ranger, $input);
                    trainer.optimiser.set_params(RangerParams { decay: weight_decay, ..Default::default() });
                    trainer.run(&schedule, &settings, &data_loader);
                }
            }
        }};
        ($input:expr, crelu) => {{
            let weight_decay = args.weight_decay;
            match args.optimizer {
                OptimizerType::AdamW => {
                    let mut trainer = build_trainer_crelu!(optimiser::AdamW, $input);
                    trainer.optimiser.set_params(AdamWParams { decay: weight_decay, ..Default::default() });
                    trainer.run(&schedule, &settings, &data_loader);
                }
                OptimizerType::RAdam => {
                    let mut trainer = build_trainer_crelu!(optimiser::RAdam, $input);
                    let params: RAdamParams = RAdamParams { decay: weight_decay, ..Default::default() };
                    trainer.optimiser.set_params(params.into());
                    trainer.run(&schedule, &settings, &data_loader);
                }
                OptimizerType::Ranger => {
                    let mut trainer = build_trainer_crelu!(optimiser::Ranger, $input);
                    trainer.optimiser.set_params(RangerParams { decay: weight_decay, ..Default::default() });
                    trainer.run(&schedule, &settings, &data_loader);
                }
            }
        }};
    }

    // Run training based on feature set and activation
    match (args.features, args.activation) {
        (FeatureSet::HalfkaHm, ActivationType::Screlu) => run_training!(ShogiHalfKA_hm, screlu),
        (FeatureSet::HalfkaHm, ActivationType::Crelu) => run_training!(ShogiHalfKA_hm, crelu),
        (FeatureSet::HalfKP, ActivationType::Screlu) => run_training!(ShogiHalfKP, screlu),
        (FeatureSet::HalfKP, ActivationType::Crelu) => run_training!(ShogiHalfKP, crelu),
    }
}

// =============================================================================
// Inference Network Structure (reference for engine integration)
// =============================================================================

/// Square Clipped ReLU - activation function
#[inline]
fn _screlu(x: i16, qa: i16) -> i32 {
    let y = i32::from(x).clamp(0, i32::from(qa));
    y * y
}
