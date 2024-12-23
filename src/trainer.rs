mod loader;
mod logger;
mod schedule;
mod settings;

use std::{fs::File, sync::mpsc, thread, time::Instant};

use crate::network::Network;
pub use loader::DirectSequentialDataLoader;
pub use schedule::{lr, wdl, TrainingSchedule, TrainingSteps};
pub use settings::LocalSettings;

use bulletformat::ChessBoard;

pub struct Trainer {
    network: Box<Network>,
    adamw: AdamW,
}

impl Default for Trainer {
    fn default() -> Self {
        Self { network: Network::random(), adamw: AdamW::default() }
    }
}

impl Trainer {
    #[allow(unused)]
    pub fn from_checkpoint(path: &str) -> std::io::Result<Self> {
        Ok(Self {
            network: Network::read(&mut File::open(format!("{path}/network.bin"))?)?,
            adamw: AdamW {
                momentum: Network::read(&mut File::open(format!("{path}/momentum.bin"))?)?,
                velocity: Network::read(&mut File::open(format!("{path}/velocity.bin"))?)?,
                ..Default::default()
            },
        })
    }

    pub fn save_to_checkpoint(&self, path: &str) -> std::io::Result<()> {
        std::fs::create_dir(path).unwrap_or(());

        self.network.write(&mut File::create(format!("{path}/network.bin"))?)?;
        self.adamw.momentum.write(&mut File::create(format!("{path}/momentum.bin"))?)?;
        self.adamw.velocity.write(&mut File::create(format!("{path}/velocity.bin"))?)?;
        self.network.write_quantised(&mut File::create(format!("{path}/quantised.bin"))?)
    }

    pub fn run<LR: lr::LrScheduler, WDL: wdl::WdlScheduler>(
        &mut self,
        loader: DirectSequentialDataLoader,
        schedule: &TrainingSchedule<LR, WDL>,
        settings: &LocalSettings,
    ) {
        logger::clear_colours();
        println!("{}", logger::ansi("Beginning Training", "34;1"));
        schedule.display();
        settings.display();

        let positions = loader.count_positions();
        let steps = schedule.steps;
        let pos_per_sb = steps.batch_size * steps.batches_per_superbatch;
        let sbs = steps.end_superbatch - steps.start_superbatch + 1;
        let total_pos = pos_per_sb * sbs;
        let iters = total_pos as f64 / positions as f64;

        println!("Positions              : {}", logger::ansi(positions, 31));
        println!("Total Epochs           : {}", logger::ansi(format!("{iters:.2}"), 31));

        let timer = Instant::now();
        let threads = settings.threads;
        let out_dir = settings.output_directory.to_string();
        let out_dir = out_dir.as_str();

        std::fs::create_dir(out_dir).unwrap_or(());

        let (sender, receiver) = mpsc::sync_channel(settings.batch_queue_size);

        let dataloader = std::thread::spawn(move || {
            loader.map_batches(steps.batch_size, |batch| sender.send(batch.to_vec()).is_err());
        });

        let mut prev_lr = schedule.lr(0, 1);
        let mut superbatch = steps.start_superbatch;
        let mut curr_batch = 0;
        let mut superbatch_timer = Instant::now();
        let mut running_loss = 0.0;

        'training: while let Ok(batch) = receiver.recv() {
            let lrate = schedule.lr(curr_batch, superbatch);
            let wdl = schedule.wdl(curr_batch, superbatch);

            if lrate != prev_lr {
                println!("LR Dropped to {}", logger::ansi(lrate, logger::num_cs()));
            }

            prev_lr = lrate;

            let this_batch_size = batch.len();
            let adj = 1.0 / this_batch_size as f32;

            let mut error = 0.0;
            let grads = gradients_batch(&batch, &self.network, &mut error, schedule.eval_scale, wdl, threads);
            self.adamw.update_weights(&mut self.network, &grads, adj, lrate);

            error *= adj;
            running_loss += error;

            if curr_batch % 128 == 0 {
                logger::report_superbatch_progress(
                    superbatch,
                    steps.batch_size,
                    steps.batches_per_superbatch,
                    curr_batch,
                    &superbatch_timer,
                );
            }

            curr_batch += 1;

            if curr_batch % steps.batches_per_superbatch == 0 {
                let error = running_loss / steps.batches_per_superbatch as f32;
                running_loss = 0.0;

                let total_time = timer.elapsed().as_secs_f32();
                let sb_time = superbatch_timer.elapsed().as_secs_f32();

                logger::report_superbatch_finished(superbatch, error, sb_time, total_time, pos_per_sb);
                logger::report_time_left(steps, superbatch, total_time);

                if schedule.should_save(superbatch) {
                    let name = format!("{}-{superbatch}", schedule.net_id());
                    let out_dir = settings.output_directory;
                    let path = format!("{out_dir}/{name}");
                    self.save_to_checkpoint(&path).expect("Failure in writing to checkpoint!");
                    println!("Saved [{}] to {}", logger::ansi(name, 31), logger::ansi(path, 31));
                }

                superbatch += 1;
                curr_batch = 0;
                superbatch_timer = Instant::now();

                if superbatch > steps.end_superbatch {
                    break 'training;
                }
            }
        }

        let total_time = timer.elapsed().as_secs();
        let (hours, minutes, seconds) = logger::seconds_to_hms(total_time as u32);

        println!(
            "Total Training Time: {}h {}m {}s",
            logger::ansi(hours, logger::num_cs()),
            logger::ansi(minutes, logger::num_cs()),
            logger::ansi(seconds, logger::num_cs()),
        );

        drop(receiver);
        dataloader.join().unwrap();
    }
}

fn gradients_batch(
    batch: &[ChessBoard],
    nnue: &Network,
    error: &mut f32,
    scale: f32,
    blend: f32,
    threads: usize,
) -> Box<Network> {
    let rscale = 1.0 / scale;
    let size = batch.len() / threads;
    let mut errors = vec![0.0; threads];
    let mut grad = Network::new();

    thread::scope(|s| {
        batch
            .chunks(size)
            .zip(errors.iter_mut())
            .map(|(chunk, error)| {
                s.spawn(move || {
                    let mut grad = Network::new();
                    for pos in chunk {
                        *error += nnue.update_single_grad(&mut grad, pos, blend, rscale);
                    }
                    grad
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|p| p.join().unwrap())
            .for_each(|part| grad.add(&part));
    });
    let batch_error = errors.iter().sum::<f32>();
    *error += batch_error;
    grad
}

pub struct AdamW {
    velocity: Box<Network>,
    momentum: Box<Network>,
    decay: f32,
}

impl Default for AdamW {
    fn default() -> Self {
        Self { velocity: Network::new(), momentum: Network::new(), decay: 0.01 }
    }
}

impl AdamW {
    pub fn update_weights(&mut self, nnue: &mut Network, grads: &Network, adj: f32, rate: f32) {
        let decay = 1.0 - self.decay * rate;
        nnue.update(&mut self.momentum, &mut self.velocity, grads, decay, adj, rate);
    }
}
