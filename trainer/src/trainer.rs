use data::Position;

use crate::{
    activation::Activation,
    arch::{test_eval, update_single_grad, NNUEParams, QuantisedNNUE},
    optimiser::Optimiser,
    scheduler::LrScheduler,
};

use std::{
    fs::{File, metadata},
    io::{BufRead, BufReader, stdout, Write},
    thread,
    time::Instant,
};

#[macro_export]
macro_rules! ansi {
    ($x:expr) => {
        format!("\x1b[36m{}\x1b[0m", $x)
    };
    ($x:expr, $y:expr) => {
        format!("\x1b[{}m{}\x1b[0m", $y, $x)
    };
}

pub struct Trainer<Opt: Optimiser> {
    file: String,
    threads: usize,
    scheduler: LrScheduler,
    blend: f32,
    skip_prop: f32,
    optimiser: Opt,
}

impl<Opt: Optimiser> Trainer<Opt> {
    #[must_use]
    pub fn new(
        file: String,
        threads: usize,
        scheduler: LrScheduler,
        blend: f32,
        skip_prop: f32,
        optimiser: Opt,
    ) -> Self {
        Self {
            file,
            threads,
            scheduler,
            blend,
            skip_prop,
            optimiser,
        }
    }

    pub fn report_settings(&self) {
        println!("File Path      : {}", ansi!(self.file, "32;1"));
        println!("Threads        : {}", ansi!(self.threads, 31));
        println!("Learning Rate  : {}", ansi!(self.scheduler.lr(), 31));
        println!("WDL Proportion : {}", ansi!(self.blend, 31));
        println!("Skip Proportion: {}", ansi!(self.skip_prop, 31));
    }

    pub fn run<Act: Activation>(
        &mut self,
        nnue: &mut NNUEParams,
        max_epochs: usize,
        net_name: &str,
        save_rate: usize,
        batch_size: usize,
        scale: f32,
    ) {
        println!("{}", ansi!("Beginning Training", "34;1"));
        let reciprocal_scale = 1.0 / scale;
        let file_size = metadata(&self.file).unwrap().len();
        let num = file_size / std::mem::size_of::<Position>() as u64;
        let batches = num / batch_size as u64 + 1;

        // display settings to user so they can verify
        self.report_settings();
        println!("Max Epochs     : {}", ansi!(max_epochs, 31));
        println!("Save Rate      : {}", ansi!(save_rate, 31));
        println!("Batch Size     : {}", ansi!(batch_size, 31));
        println!("Net Name       : {}", ansi!(net_name, "32;1"));
        println!("LR Scheduler   : {}", self.scheduler);
        println!("Scale          : {}", ansi!(format!("{scale:.0}"), 31));
        println!("Positions      : {}", ansi!(num, 31));

        let timer = Instant::now();

        let mut error;

        for epoch in 1..=max_epochs {
            let epoch_timer = Instant::now();
            error = 0.0;
            let mut finished_batches = 0;

            let cap = 1024 * batch_size * std::mem::size_of::<Position>();
            let mut file = BufReader::with_capacity(cap, File::open(&self.file).unwrap());

            while let Ok(buf) = file.fill_buf() {
                // finished reading file
                if buf.is_empty() {
                    break;
                }

                let buf_ref: &[Position] = unsafe { data::util::to_slice_with_lifetime(buf) };

                for batch in buf_ref.chunks(batch_size) {
                    let adj = 2. / batch.len() as f32;
                    let gradients = self.gradients::<Act>(nnue, batch, &mut error, reciprocal_scale);

                    self.optimiser
                        .update_weights(nnue, &gradients, adj, self.scheduler.lr());

                    if finished_batches % 500 == 0 {
                        let pct = finished_batches as f32 / batches as f32 * 100.0;
                        let positions = finished_batches * batch_size;
                        let pos_per_sec = positions as f32 / epoch_timer.elapsed().as_secs_f32();
                        print!(
                            "epoch {} [{}% ({}/{} batches, {} pos/sec)]\r",
                            ansi!(epoch),
                            ansi!(format!("{pct:.1}")),
                            ansi!(finished_batches),
                            ansi!(batches),
                            ansi!(format!("{pos_per_sec:.0}")),
                        );
                        let _ = stdout().flush();
                    }

                    finished_batches += 1;
                }

                let consumed = buf.len();
                file.consume(consumed);
            }

            error /= num as f32;

            let epoch_time = epoch_timer.elapsed().as_secs_f32();

            println!(
                "epoch {} | time {} | running loss {} | {} pos/sec | total time {}",
                ansi!(epoch),
                ansi!(format!("{epoch_time:.2}")),
                ansi!(format!("{error:.6}")),
                ansi!(format!("{:.0}", num.max(1) as f32 / epoch_time)),
                ansi!(format!("{:.2}", timer.elapsed().as_secs_f32())),
            );

            self.scheduler.adjust(epoch);

            if epoch % save_rate == 0 && epoch != max_epochs {
                let net_path = format!("nets/{net_name}-epoch{epoch}.bin");

                let qnnue = QuantisedNNUE::from_unquantised(nnue);
                qnnue.write_to_bin(&net_path).unwrap();

                println!("Saved [{}]", ansi!(net_path, "32;1"));
            }
        }

        test_eval::<Act>(nnue);
    }

    fn gradients<Act: Activation>(
        &self,
        nnue: &NNUEParams,
        batch: &[Position],
        error: &mut f32,
        scale: f32,
    ) -> Box<NNUEParams> {
        let size = batch.len() / self.threads;
        let mut errors = vec![0.0; self.threads];
        let mut grad = NNUEParams::new();
        let blend = self.blend;
        let skip_prop = self.skip_prop;
        thread::scope(|s| {
            batch
                .chunks(size)
                .zip(errors.iter_mut())
                .map(|(chunk, error)| {
                    s.spawn(|| gradients_batch::<Act>(chunk, nnue, error, blend, skip_prop, scale))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|p| p.join().unwrap())
                .for_each(|part| *grad += &part);
        });
        *error += errors.iter().sum::<f32>();
        grad
    }
}

fn gradients_batch<Act: Activation>(
    positions: &[Position],
    nnue: &NNUEParams,
    error: &mut f32,
    blend: f32,
    skip_prop: f32,
    scale: f32,
) -> Box<NNUEParams> {
    let mut grad = NNUEParams::new();
    let mut rand = crate::rng::Rand::default();
    for pos in positions {
        if rand.rand(1.0) < skip_prop {
            continue;
        }

        update_single_grad::<Act>(pos, nnue, &mut grad, error, blend, scale);
    }
    grad
}
