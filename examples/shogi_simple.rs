/*
将棋 NNUE 学習スクリプト

使用方法:
    cargo run --release --example shogi_simple -- [OPTIONS]

オプション:
    --arch <ARCH>       アーキテクチャプリセット (デフォルト: 256x2-32-32)
                        プリセット: 256x2-32-32, 512x2-8-96, 512x2-32-32, 1024x2-8-32
    --l1 <SIZE>         L1 (アキュムレータ) サイズ (プリセットを上書き)
    --l2 <SIZE>         L2 (中間層1) サイズ
    --l3 <SIZE>         L3 (中間層2) サイズ
    --data <PATH>       訓練データパス (複数指定可、カンマ区切り)
    --batch-size <N>    バッチサイズ (デフォルト: 16384)
    --superbatches <N>  Superbatch 数 (デフォルト: 100)
    --lr <RATE>         初期学習率 (デフォルト: 0.001)
    --wdl <LAMBDA>      WDL lambda (デフォルト: 0.75)
    --scale <N>         評価値スケール (デフォルト: 600)
    --save-rate <N>     保存間隔 (デフォルト: 10)
    --threads <N>       スレッド数 (デフォルト: 4)
    --output <DIR>      出力ディレクトリ (デフォルト: checkpoints)
    --net-id <NAME>     ネットワークID (デフォルト: shogi-halfka-hm)
    --weight-decay <F>  Weight decay (デフォルト: 0.01)

例:
    # デフォルト設定で学習
    cargo run --release --example shogi_simple -- --data data/train.bin

    # 512x2-8-96 アーキテクチャで学習
    cargo run --release --example shogi_simple -- --arch 512x2-8-96 --data data/train.bin

    # カスタムサイズで学習
    cargo run --release --example shogi_simple -- --l1 1024 --l2 16 --l3 64 --data data/train.bin
*/

use std::path::PathBuf;

use bullet_lib::{
    game::inputs::{ShogiHalfKA_hm, SparseInputType},
    nn::optimiser::{self, AdamWParams, RAdamParams, RangerParams},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};
use clap::{Parser, ValueEnum};

// =============================================================================
// CLI 引数定義
// =============================================================================

/// オプティマイザ選択
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum OptimizerType {
    /// AdamW - 高速収束だがスパース入力に不安定な可能性
    AdamW,
    /// RAdam - Rectified Adam、より安定
    RAdam,
    /// Ranger - RAdam + Lookahead（nnue-pytorch推奨）
    #[default]
    Ranger,
}

#[derive(Parser, Debug)]
#[command(name = "shogi_simple")]
#[command(about = "将棋 NNUE 学習スクリプト")]
struct Args {
    /// アーキテクチャプリセット
    /// プリセット: 256x2-32-32, 512x2-8-96, 512x2-32-32, 1024x2-8-32
    #[arg(long, default_value = "256x2-32-32")]
    arch: String,

    /// オプティマイザ (adamw, radam, ranger)
    /// ranger = RAdam + Lookahead（nnue-pytorch推奨と同じ）
    #[arg(long, value_enum, default_value = "ranger")]
    optimizer: OptimizerType,

    /// L1 (アキュムレータ) サイズ (プリセットを上書き)
    #[arg(long)]
    l1: Option<usize>,

    /// L2 (中間層1) サイズ
    #[arg(long)]
    l2: Option<usize>,

    /// L3 (中間層2) サイズ
    #[arg(long)]
    l3: Option<usize>,

    /// 訓練データパス (複数指定可、カンマ区切り)
    #[arg(long, default_value = "data/train.bin")]
    data: String,

    /// バッチサイズ
    #[arg(long, default_value = "16384")]
    batch_size: usize,

    /// Superbatch 数
    #[arg(long, default_value = "100")]
    superbatches: usize,

    /// 初期学習率
    #[arg(long, default_value = "0.001")]
    lr: f32,

    /// WDL lambda (0.0=勝敗のみ, 1.0=評価値のみ)
    #[arg(long, default_value = "0.75")]
    wdl: f32,

    /// 評価値スケール
    #[arg(long, default_value = "600")]
    scale: i32,

    /// 保存間隔 (superbatch)
    #[arg(long, default_value = "10")]
    save_rate: usize,

    /// スレッド数
    #[arg(long, default_value = "4")]
    threads: usize,

    /// 出力ディレクトリ
    #[arg(long, default_value = "checkpoints")]
    output: PathBuf,

    /// ネットワークID
    #[arg(long, default_value = "shogi-halfka-hm")]
    net_id: String,

    /// 量子化係数 QA (L0用)
    #[arg(long, default_value = "127")]
    qa: i16,

    /// 量子化係数 QB (後段層用)
    #[arg(long, default_value = "64")]
    qb: i16,

    /// Weight decay (L2正則化)
    #[arg(long, default_value = "0.01")]
    weight_decay: f32,
}

// =============================================================================
// アーキテクチャ定義
// =============================================================================

#[derive(Debug, Clone, Copy)]
struct Architecture {
    l1: usize, // アキュムレータサイズ
    l2: usize, // 中間層1サイズ
    l3: usize, // 中間層2サイズ
}

impl Architecture {
    /// プリセット名からアーキテクチャを取得
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

    /// 利用可能なプリセット一覧
    fn available_presets() -> &'static [&'static str] {
        &["256x2-32-32", "512x2-8-96", "512x2-32-32", "1024x2-8-32", "1024x2-16-64"]
    }

    /// 表示用文字列
    fn display(&self) -> String {
        format!("{}x2-{}-{}", self.l1, self.l2, self.l3)
    }
}

// =============================================================================
// メイン処理
// =============================================================================

fn main() {
    let args = Args::parse();

    // アーキテクチャ決定
    let mut arch = Architecture::from_preset(&args.arch).unwrap_or_else(|| {
        eprintln!("Unknown architecture preset: {}", args.arch);
        eprintln!("Available presets: {:?}", Architecture::available_presets());
        std::process::exit(1);
    });

    // 個別指定で上書き
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

    // 量子化係数
    let qa = args.qa;
    let qb = args.qb;

    // 入力特徴量
    let input = ShogiHalfKA_hm;
    let input_size = input.num_inputs();

    // オプティマイザ名
    let optimizer_name = match args.optimizer {
        OptimizerType::AdamW => "AdamW",
        OptimizerType::RAdam => "RAdam",
        OptimizerType::Ranger => "Ranger (RAdam + Lookahead)",
    };

    // 設定表示
    println!("=== Shogi NNUE Training ===");
    println!("Architecture: {} (L1={}, L2={}, L3={})", arch.display(), l1_size, l2_size, l3_size);
    println!("Network: {} -> {}x2 -> {} -> {} -> 1", input_size, l1_size, l2_size, l3_size);
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

    // 学習スケジュール
    let schedule = TrainingSchedule {
        net_id: args.net_id,
        eval_scale: args.scale as f32,
        steps: TrainingSteps {
            batch_size: args.batch_size,
            batches_per_superbatch: 6104, // 約1億局面/superbatch
            start_superbatch: 1,
            end_superbatch: args.superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: args.wdl },
        lr_scheduler: lr::StepLR { start: args.lr, gamma: 0.3, step: 30 },
        save_rate: args.save_rate,
    };

    // ローカル設定
    let output_dir = args.output.to_str().unwrap_or("checkpoints");
    let settings =
        LocalSettings { threads: args.threads, test_set: None, output_directory: output_dir, batch_queue_size: 64 };

    // データローダー
    let data_files: Vec<&str> = args.data.split(',').collect();
    let data_loader = DirectSequentialDataLoader::new(&data_files);

    // SaveFormat 定義
    let save_format = [
        SavedFormat::id("l0w").round().quantise::<i16>(qa),
        SavedFormat::id("l0b").round().quantise::<i16>(qa),
        SavedFormat::id("l1w").round().quantise::<i16>(qb),
        SavedFormat::id("l1b").round().quantise::<i16>(qa * qb),
        SavedFormat::id("l2w").round().quantise::<i16>(qb),
        SavedFormat::id("l2b").round().quantise::<i16>(qa * qb),
        SavedFormat::id("outw").round().quantise::<i16>(qb),
        SavedFormat::id("outb").round().quantise::<i16>(qa * qb),
    ];

    // ネットワーク構築マクロ（重複を減らすため）
    macro_rules! build_trainer {
        ($opt:expr) => {
            ValueTrainerBuilder::default()
                .dual_perspective()
                .optimiser($opt)
                .inputs(input)
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

    // オプティマイザに応じてトレーナーを構築・実行
    let weight_decay = args.weight_decay;
    match args.optimizer {
        OptimizerType::AdamW => {
            let mut trainer = build_trainer!(optimiser::AdamW);
            trainer.optimiser.set_params(AdamWParams { decay: weight_decay, ..Default::default() });
            trainer.run(&schedule, &settings, &data_loader);
        }
        OptimizerType::RAdam => {
            let mut trainer = build_trainer!(optimiser::RAdam);
            let params: RAdamParams = RAdamParams { decay: weight_decay, ..Default::default() };
            trainer.optimiser.set_params(params.into());
            trainer.run(&schedule, &settings, &data_loader);
        }
        OptimizerType::Ranger => {
            let mut trainer = build_trainer!(optimiser::Ranger);
            trainer.optimiser.set_params(RangerParams { decay: weight_decay, ..Default::default() });
            trainer.run(&schedule, &settings, &data_loader);
        }
    }
}

// =============================================================================
// 推論用ネットワーク構造体（エンジン組み込み用参考）
// =============================================================================

/// Square Clipped ReLU - 活性化関数
#[inline]
fn _screlu(x: i16, qa: i16) -> i32 {
    let y = i32::from(x).clamp(0, i32::from(qa));
    y * y
}
