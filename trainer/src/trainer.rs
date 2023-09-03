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
        println!("File Path      : {:?}", self.file);
        println!("Threads        : {}", self.threads);
        println!("Learning Rate  : {}", self.scheduler.lr());
        println!("WDL Proportion : {}", self.blend);
        println!("Skip Proportion: {}", self.skip_prop);
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

        let reciprocal_scale = 1.0 / scale;
        let file_size = metadata(&self.file).unwrap().len();
        let num = file_size / std::mem::size_of::<Position>() as u64;
        let batches = num / batch_size as u64 + 1;

        // display settings to user so they can verify
        self.report_settings();
        println!("Max Epochs     : {max_epochs}");
        println!("Save Rate      : {save_rate}");
        println!("Batch Size     : {batch_size}");
        println!("Net Name       : {net_name:?}");
        println!("LR Scheduler   : {}", self.scheduler);
        println!("Scale          : {scale:.6}");
        println!("Positions      : {num}");

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
                        print!("epoch {epoch} [{pct:.1}% ({finished_batches} / {batches} Batches, {pos_per_sec:.0} pos/sec)]\r");
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
                "epoch {epoch} | time {epoch_time:.2} | running loss {error:.6} | lr {} | {:.0} pos/sec | total time {:.2}",
                self.scheduler.lr(),
                num.max(1) as f32 / epoch_time,
                timer.elapsed().as_secs_f32(),
            );

            self.scheduler.adjust(epoch);

            if epoch % save_rate == 0 && epoch != max_epochs {
                let qnnue = QuantisedNNUE::from_unquantised(nnue);
                qnnue
                    .write_to_bin(&format!("nets/{net_name}-{epoch}.bin"))
                    .unwrap();

                println!("Saved [nets/{net_name}-{epoch}.bin]");
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
