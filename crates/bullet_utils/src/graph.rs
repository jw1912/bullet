use anyhow::{bail, Context};
use structopt::StructOpt;

use std::{ffi::OsStr, io::BufRead, path::PathBuf};

use plotters::prelude::*;

#[derive(StructOpt)]
pub struct GraphOptions {
    /// Files to process
    #[structopt(name = "FILE", parse(from_os_str))]
    files: Vec<PathBuf>,
}

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
const IMG_DIMS: (u32, u32) = (2560, 1440);

const NOISY_PLOT_OPACITY: f64 = 0.001;

/// Calculates the simple moving average
fn moving_average(data: &[(usize, f64)], window_size: usize) -> Vec<(usize, f64)> {
    data.windows(window_size)
        .map(|window| {
            let mean = window.iter().map(|p| p.1).sum::<f64>() / window_size as f64;
            let idx = window[window_size / 2].0;
            (idx, mean)
        })
        .collect()
}

impl GraphOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        // If there weren't any paths passed, do nothing.
        if self.files.is_empty() {
            bail!("No log files provided! \nUsage: \n $ bullet-utils graph log1.txt log2.txt...");
        }

        // Create the output directory if it doesn't exist
        let plot_dir = "visualisation/plots";
        std::fs::create_dir_all(plot_dir).with_context(|| "Failed to create plots directory.")?;

        let mut data_sequences = Vec::new();

        for (i, log_file) in self.files.iter().enumerate() {
            let lf_string = log_file.to_string_lossy();
            let file =
                std::fs::File::open(log_file).with_context(|| format!("Failed to open log file {lf_string}."))?;
            if file.metadata().with_context(|| format!("Failed to get file metadata of {lf_string}."))?.is_dir() {
                bail!("Cannot read a directory as a log file! You should only pass correctly-formatted text files to this utility.\n    Problematic file: {lf_string}");
            }
            let reader = std::io::BufReader::new(file);
            let mut run_name = log_file
                .parent()
                .and_then(std::path::Path::file_name)
                .and_then(OsStr::to_str)
                .map(|s| s.trim_end_matches(|c| "0123456789".contains(c)).to_string())
                .unwrap_or_else(|| format!("log-{}", i + 1));
            if lf_string.contains("validation") {
                run_name.push_str(" (test)")
            }
            let data = reader
                .lines()
                .enumerate()
                .map(|(line_no, res)| {
                    let line = res?;

                    let sbatch_text = line
                        .split(',')
                        .nth(0)
                        .and_then(|f| match f.split_once(':') {
                            Some(("superbatch", v)) => Some(v),
                            None => Some(f),
                            _ => None,
                        })
                        .with_context(|| {
                            format!("No superbatch found in line {line_no} of file {}.", log_file.to_string_lossy())
                        })?;
                    let sbatch = sbatch_text
                        .trim()
                        .parse()
                        .with_context(|| format!("Failed to parse \"{sbatch_text}\" as usize."))?;

                    let batch_text = line
                        .split(',')
                        .nth(1)
                        .and_then(|f| match f.split_once(':') {
                            Some(("batch", v)) => Some(v),
                            None => Some(f),
                            _ => None,
                        })
                        .with_context(|| {
                            format!("No batch found in line {line_no} of file {}.", log_file.to_string_lossy())
                        })?;
                    let batch = batch_text
                        .trim()
                        .parse()
                        .with_context(|| format!("Failed to parse \"{batch_text}\" as usize."))?;

                    let loss_text = line
                        .split(',')
                        .nth(2)
                        .and_then(|f| match f.split_once(':') {
                            Some(("loss", v)) => Some(v),
                            None => Some(f),
                            _ => None,
                        })
                        .with_context(|| {
                            format!("No loss found in line {line_no} of file {}.", log_file.to_string_lossy())
                        })?;
                    let loss =
                        loss_text.trim().parse().with_context(|| format!("Failed to parse \"{loss_text}\" as f64."))?;

                    Ok((sbatch, batch, loss))
                })
                .collect::<anyhow::Result<Vec<(usize, usize, f64)>>>()?;
            if data.is_empty() {
                bail!("{} contains no data.", log_file.to_string_lossy());
            }

            // sort out batches
            let batches_per_superbatch = data.iter().map(|p| p.1).max().unwrap() + 1;
            let data =
                data.into_iter().map(|(sb, b, l)| ((sb - 1) * batches_per_superbatch + b, l)).collect::<Vec<_>>();

            data_sequences.push((run_name, data));
        }

        // Determine cutoff guard and graph dimensions.
        let x_min = 0;
        let x_max = (data_sequences[0].1.len() * 32) as i32;
        let mut y_min = f64::MAX;
        let mut y_max = f64::MIN;
        let mut guard = 0.0f64;
        for (_, data) in data_sequences.iter() {
            y_min = y_min
                .min(data.iter().map(|p| p.1).reduce(f64::min).with_context(|| "Empty data sequence encountered!")?);
            y_max = y_max
                .max(data.iter().map(|p| p.1).reduce(f64::max).with_context(|| "Empty data sequence encountered!")?);
            let tail = &data[data.len() / 4..];
            let max_tail = tail.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
            let min_tail = tail.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
            let diff = max_tail - min_tail;
            let this_guard = min_tail + diff * 2.0;
            guard = guard.max(this_guard);
        }

        // Plot the full loss sequences
        let output_path_full = format!("{}/rs-training_loss_full.png", plot_dir);
        let root = BitMapBackend::new(&output_path_full, IMG_DIMS).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Training loss over time (full)", (FONT, TITLE_FONT_SIZE))
            .margin(MARGIN)
            .x_label_area_size(X_LABEL_AREA_SIZE)
            .y_label_area_size(Y_LABEL_AREA_SIZE)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

        chart.plotting_area().fill(&CHART_BG_COLOUR)?;

        chart
            .configure_mesh()
            .x_label_style((FONT, TICKS_FONT_SIZE).into_font())
            .y_label_style((FONT, TICKS_FONT_SIZE).into_font())
            .axis_desc_style((FONT, LABEL_FONT_SIZE).into_font())
            .x_desc("Batch")
            .y_desc("Loss")
            .draw()?;

        for (i, (run_name, data)) in data_sequences.iter().enumerate() {
            let window_size = 10;
            let smoothed_loss = moving_average(data, window_size);

            chart
                .draw_series(LineSeries::new(
                    data.iter().map(|&(x, y)| (x as i32, y)),
                    ShapeStyle::from(COLOURS[i % COLOURS.len()].mix(NOISY_PLOT_OPACITY))
                        .stroke_width(RAW_LINE_STROKE_WIDTH),
                ))?
                .label(run_name)
                .legend(move |(x, y)| {
                    PathElement::new(
                        [(x, y), (x + LEGEND_DRAW_OFFSET, y)],
                        ShapeStyle::from(COLOURS[i % COLOURS.len()]).stroke_width(LEGEND_STROKE_WIDTH),
                    )
                });

            chart.draw_series(LineSeries::new(
                smoothed_loss.iter().map(|&(x, y)| (x as i32, y)),
                ShapeStyle::from(COLOURS[i % COLOURS.len()]).stroke_width(SMOOTH_LINE_STROKE_WIDTH),
            ))?;

            // chart.configure_series_labels().position(SeriesLabelPosition::UpperRight);
        }

        chart
            .configure_series_labels()
            .border_style(BLACK)
            .background_style(WHITE.mix(0.8))
            .position(SeriesLabelPosition::UpperRight)
            .legend_area_size(LEGEND_AREA_SIZE)
            .draw()?;
        root.present()?;

        // Plot the loss sequences excluding the initial few epochs
        let output_path_trimmed = format!("{}/rs-training_loss_trimmed.png", plot_dir);
        let root = BitMapBackend::new(&output_path_trimmed, IMG_DIMS).into_drawing_area();
        root.fill(&WHITE)?;

        y_max = y_max.min(guard);

        let mut chart = ChartBuilder::on(&root)
            .caption("Training loss over time (clipped)", (FONT, TITLE_FONT_SIZE))
            .margin(MARGIN)
            .x_label_area_size(X_LABEL_AREA_SIZE)
            .y_label_area_size(Y_LABEL_AREA_SIZE)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

        chart.plotting_area().fill(&CHART_BG_COLOUR)?;

        chart
            .configure_mesh()
            .x_label_style((FONT, TICKS_FONT_SIZE).into_font())
            .y_label_style((FONT, TICKS_FONT_SIZE).into_font())
            .axis_desc_style((FONT, LABEL_FONT_SIZE).into_font())
            .x_desc("Batch")
            .y_desc("Loss")
            .draw()?;

        for (i, (run_name, data)) in data_sequences.iter().enumerate() {
            let window_size = 200;
            let smoothed_loss = moving_average(data, window_size);
            let last_exceeding_instance = data.iter().map(|p| p.1).rposition(|loss| loss > guard).unwrap_or(0);
            let cutoff = last_exceeding_instance + 1;

            chart
                .draw_series(LineSeries::new(
                    data[cutoff..].iter().map(|&(x, y)| (x as i32, y)),
                    ShapeStyle::from(COLOURS[i % COLOURS.len()].mix(NOISY_PLOT_OPACITY))
                        .stroke_width(RAW_LINE_STROKE_WIDTH),
                ))?
                .label(run_name)
                .legend(move |(x, y)| {
                    PathElement::new(
                        [(x, y), (x + LEGEND_DRAW_OFFSET, y)],
                        ShapeStyle::from(COLOURS[i % COLOURS.len()]).stroke_width(LEGEND_STROKE_WIDTH),
                    )
                });

            chart.draw_series(LineSeries::new(
                smoothed_loss[cutoff..].iter().map(|&(x, y)| (x as i32, y)),
                ShapeStyle::from(COLOURS[i % COLOURS.len()]).stroke_width(SMOOTH_LINE_STROKE_WIDTH),
            ))?;
        }

        chart
            .configure_series_labels()
            .border_style(BLACK)
            .background_style(WHITE.mix(0.8))
            .position(SeriesLabelPosition::UpperRight)
            .legend_area_size(LEGEND_AREA_SIZE)
            .label_font((FONT, LEGEND_FONT_SIZE).into_font())
            .draw()?;
        root.present()?;

        println!("Full plot saved to {}", output_path_full);
        println!("Trimmed plot saved to {}", output_path_trimmed);

        Ok(())
    }
}
