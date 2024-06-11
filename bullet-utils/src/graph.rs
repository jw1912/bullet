use structopt::StructOpt;

use std::io::BufRead;

use plotters::prelude::*;

#[derive(StructOpt)]
pub struct GraphOptions {
    // #[structopt(required = true, short, long)]
    // input_folder: PathBuf,
}

const LOG_FILES: &[&str] = &[
    "checkpoints/optimiser-benchmark-screlu-64n-10/log.txt",
    "checkpoints/optimiser-benchmark-64n-10/log.txt",
    "checkpoints/optimiser-benchmark-32n-10/log.txt",
    "checkpoints/optimiser-benchmark-16n-10/log.txt",
    "checkpoints/optimiser-benchmark-8n-10/log.txt",
    "checkpoints/optimiser-benchmark-4n-10/log.txt",
    // Add more log files as needed
];

const COLOURS: &[RGBColor] = &[
    RGBColor(31, 119, 180),
    RGBColor(255, 127, 14),
    RGBColor(44, 160, 44),
    RGBColor(214, 39, 40),
    RGBColor(148, 103, 189),
    RGBColor(140, 86, 75),
    RGBColor(227, 119, 194),
    RGBColor(127, 127, 127),
    RGBColor(188, 189, 34),
    RGBColor(23, 190, 207),
];

const CHART_BG_COLOUR: RGBAColor = RGBAColor(234, 234, 242, 1.0);

const LEGEND_STROKE_WIDTH: u32 = 4;
const RAW_LINE_STROKE_WIDTH: u32 = 1;
const SMOOTH_LINE_STROKE_WIDTH: u32 = 2;

const X_LABEL_AREA_SIZE: i32 = 120;
const Y_LABEL_AREA_SIZE: i32 = 140;
const LEGEND_AREA_SIZE: i32 = 100;
const LEGEND_DRAW_OFFSET: i32 = 90;

const TITLE_FONT_SIZE: i32 = 80;
const LABEL_FONT_SIZE: i32 = 50;
const LEGEND_FONT_SIZE: i32 = 50;
const TICKS_FONT_SIZE: i32 = 40;

const FONT: &str = "Iosevka";

const MARGIN: i32 = 40;

/// 4k UHD
const IMG_DIMS: (u32, u32) = (3840, 2160);

const NOISY_PLOT_OPACITY: f64 = 0.06;

/// Calculates the simple moving average
fn moving_average(data: &[f64], window_size: usize) -> Vec<f64> {
    data.windows(window_size).map(|window| window.iter().sum::<f64>() / window_size as f64).collect()
}

impl GraphOptions {
    pub fn run(&self) {
        // Create the output directory if it doesn't exist
        let plot_dir = "visualisation/plots";
        std::fs::create_dir_all(plot_dir).expect("Failed to create plots directory.");

        let mut data_sequences = Vec::new();

        for &log_file in LOG_FILES {
            let file = std::fs::File::open(log_file).expect("Failed to open log file!");
            let reader = std::io::BufReader::new(file);
            let run_name = std::path::Path::new(log_file)
                .parent()
                .unwrap()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .trim_end_matches(|c| "-0123456789".contains(c))
                .to_string();
            let data: Vec<f64> = reader
                .lines()
                .map(|line| line.unwrap())
                .map(|line| line.split_whitespace().rev().nth(0).unwrap().parse().unwrap())
                .collect();
            data_sequences.push((run_name, data));
        }

        // Determine cutoff guard and graph dimensions.
        let x_min = 0;
        let x_max = data_sequences[0].1.len() as i32;
        let mut y_min = f64::MAX;
        let mut y_max = f64::MIN;
        let mut guard = 0.0f64;
        for (_, data) in data_sequences.iter() {
            y_min = y_min.min(data.iter().copied().reduce(f64::min).expect("Empty data sequence encountered!"));
            y_max = y_max.max(data.iter().copied().reduce(f64::max).expect("Empty data sequence encountered!"));
            let tail = &data[data.len() / 4..];
            let max_tail = tail.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_tail = tail.iter().cloned().fold(f64::INFINITY, f64::min);
            let diff = max_tail - min_tail;
            let this_guard = min_tail + diff * 2.0;
            guard = guard.max(this_guard);
        }

        // Plot the full loss sequences
        let output_path_full = format!("{}/rs-training_loss_full.png", plot_dir);
        let root = BitMapBackend::new(&output_path_full, IMG_DIMS).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root)
            .caption("Training loss over time (full)", (FONT, TITLE_FONT_SIZE))
            .margin(MARGIN)
            .x_label_area_size(X_LABEL_AREA_SIZE)
            .y_label_area_size(Y_LABEL_AREA_SIZE)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)
            .unwrap();

        chart.plotting_area().fill(&CHART_BG_COLOUR).unwrap();

        chart.configure_mesh()
            .x_label_style((FONT, TICKS_FONT_SIZE).into_font())
            .y_label_style((FONT, TICKS_FONT_SIZE).into_font())
            .axis_desc_style((FONT, LABEL_FONT_SIZE).into_font())
            .x_desc("Batch")
            .y_desc("Loss")
            .draw()
            .unwrap();

        for (i, (run_name, data)) in data_sequences.iter().enumerate() {
            let window_size = 10;
            let smoothed_loss = moving_average(data, window_size);

            let x_vals: Vec<_> = (0..data.len()).collect();
            chart
                .draw_series(LineSeries::new(
                    x_vals.iter().zip(data.iter()).map(|(&x, &y)| (x as i32, y)),
                    ShapeStyle::from(COLOURS[i % COLOURS.len()].mix(NOISY_PLOT_OPACITY))
                        .stroke_width(RAW_LINE_STROKE_WIDTH),
                ))
                .unwrap()
                .label(run_name)
                .legend(move |(x, y)| {
                    PathElement::new(
                        [(x, y), (x + LEGEND_DRAW_OFFSET, y)],
                        ShapeStyle::from(COLOURS[i % COLOURS.len()]).stroke_width(LEGEND_STROKE_WIDTH),
                    )
                });

            let smoothed_x_vals: Vec<_> = (window_size - 1..data.len()).collect();
            chart
                .draw_series(LineSeries::new(
                    smoothed_x_vals.iter().zip(smoothed_loss.iter()).map(|(&x, &y)| (x as i32, y)),
                    ShapeStyle::from(COLOURS[i % COLOURS.len()]).stroke_width(SMOOTH_LINE_STROKE_WIDTH),
                ))
                .unwrap();

            // chart.configure_series_labels().position(SeriesLabelPosition::UpperRight);
        }

        chart
            .configure_series_labels()
            .border_style(BLACK)
            .background_style(WHITE.mix(0.8))
            .position(SeriesLabelPosition::UpperRight)
            .legend_area_size(LEGEND_AREA_SIZE)
            .draw()
            .unwrap();
        root.present().unwrap();

        // Plot the loss sequences excluding the initial few epochs
        let output_path_trimmed = format!("{}/rs-training_loss_trimmed.png", plot_dir);
        let root = BitMapBackend::new(&output_path_trimmed, IMG_DIMS).into_drawing_area();
        root.fill(&WHITE).unwrap();

        y_max = y_max.min(guard);

        let mut chart = ChartBuilder::on(&root)
            .caption("Training loss over time (clipped)", (FONT, TITLE_FONT_SIZE))
            .margin(MARGIN)
            .x_label_area_size(X_LABEL_AREA_SIZE)
            .y_label_area_size(Y_LABEL_AREA_SIZE)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)
            .unwrap();

        chart.plotting_area().fill(&CHART_BG_COLOUR).unwrap();

        chart.configure_mesh()
            .x_label_style((FONT, TICKS_FONT_SIZE).into_font())
            .y_label_style((FONT, TICKS_FONT_SIZE).into_font())
            .axis_desc_style((FONT, LABEL_FONT_SIZE).into_font())
            .x_desc("Batch")
            .y_desc("Loss")
            .draw()
            .unwrap();

        for (i, (run_name, data)) in data_sequences.iter().enumerate() {
            let window_size = 200;
            let smoothed_loss = moving_average(data, window_size);
            let last_exceeding_instance = data.iter().rposition(|&loss| loss > guard).unwrap();
            let cutoff = last_exceeding_instance + 1;

            let x_vals: Vec<_> = (cutoff..data.len()).collect();
            chart
                .draw_series(LineSeries::new(
                    x_vals.iter().zip(data[cutoff..].iter()).map(|(&x, &y)| (x as i32, y)),
                    ShapeStyle::from(COLOURS[i % COLOURS.len()].mix(NOISY_PLOT_OPACITY))
                        .stroke_width(RAW_LINE_STROKE_WIDTH),
                ))
                .unwrap()
                .label(run_name)
                .legend(move |(x, y)| {
                    PathElement::new(
                        [(x, y), (x + LEGEND_DRAW_OFFSET, y)],
                        ShapeStyle::from(COLOURS[i % COLOURS.len()]).stroke_width(LEGEND_STROKE_WIDTH),
                    )
                });

            let smoothed_x_vals: Vec<_> = (cutoff + window_size - 1..data.len()).collect();
            chart
                .draw_series(LineSeries::new(
                    smoothed_x_vals.iter().zip(smoothed_loss[cutoff..].iter()).map(|(&x, &y)| (x as i32, y)),
                    ShapeStyle::from(COLOURS[i % COLOURS.len()]).stroke_width(SMOOTH_LINE_STROKE_WIDTH),
                ))
                .unwrap();
        }

        chart
            .configure_series_labels()
            .border_style(BLACK)
            .background_style(WHITE.mix(0.8))
            .position(SeriesLabelPosition::UpperRight)
            .legend_area_size(LEGEND_AREA_SIZE)
            .label_font((FONT, LEGEND_FONT_SIZE).into_font())
            .draw()
            .unwrap();
        root.present().unwrap();

        println!("Full plot saved to {}", output_path_full);
        println!("Trimmed plot saved to {}", output_path_trimmed);
    }
}
