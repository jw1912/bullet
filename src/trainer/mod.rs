mod gradient;
pub mod optimiser;
pub mod scheduler;

use crate::{
    Data,
    network::{activation::Activation, NNUEParams, QuantisedNNUE},
    util::to_slice_with_lifetime,
};

use gradient::gradients;
use optimiser::Optimiser;
use scheduler::LrScheduler;

use std::{
    fs::{metadata, File},
    io::{stdout, BufRead, BufReader, Write},
    thread,
    time::Instant,
};

#[macro_export]
macro_rules! ansi {
    ($x:expr, $y:expr) => {
        format!("\x1b[{}m{}\x1b[0m", $y, $x)
    };
    ($x:expr, $y:expr, $esc:expr) => {
        format!("\x1b[{}m{}\x1b[0m{}", $y, $x, $esc)
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

    pub fn report_settings(&self, esc: &str) {
        println!("File Path      : {}", ansi!(self.file, "32;1", esc));
        println!("Threads        : {}", ansi!(self.threads, 31, esc));
        println!("WDL Proportion : {}", ansi!(self.blend, 31, esc));
        println!("Skip Proportion: {}", ansi!(self.skip_prop, 31, esc));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run<Act: Activation>(
        &mut self,
        nnue: &mut NNUEParams,
        max_epochs: usize,
        net_name: &str,
        save_rate: usize,
        batch_size: usize,
        scale: f32,
        cbcs: bool,
    ) {
        let esc = if cbcs { "\x1b[38;5;225m" } else { "" };
        let num_cs = if cbcs { 35 } else { 36 };
        print!("{esc}");

        println!("{}", ansi!("Beginning Training", "34;1", esc));
        let reciprocal_scale = 1.0 / scale;
        let file_size = metadata(&self.file).unwrap().len();
        let num = file_size / std::mem::size_of::<Data>() as u64;
        let batches = num / batch_size as u64 + 1;

        // display settings to user so they can verify
        self.report_settings(esc);
        println!("Max Epochs     : {}", ansi!(max_epochs, 31, esc));
        println!("Save Rate      : {}", ansi!(save_rate, 31, esc));
        println!("Batch Size     : {}", ansi!(batch_size, 31, esc));
        println!("Net Name       : {}", ansi!(net_name, "32;1", esc));
        println!("LR Scheduler   : {}", self.scheduler.colourful(esc));
        println!("Scale          : {}", ansi!(format!("{scale:.0}"), 31, esc));
        println!("Datas      : {}", ansi!(num, 31, esc));

        let timer = Instant::now();

        let mut error;

        for epoch in 1..=max_epochs {
            let epoch_timer = Instant::now();
            error = 0.0;
            let mut finished_batches = 0;

            let cap = 128 * batch_size * std::mem::size_of::<Data>();
            let file_path = self.file.clone();

            use std::sync::mpsc::sync_channel;

            let (sender, reciever) = sync_channel::<Vec<u8>>(2);

            let dataloader = std::thread::spawn(move || {
                let mut file = BufReader::with_capacity(cap, File::open(&file_path).unwrap());
                while let Ok(buf) = file.fill_buf() {
                    if buf.is_empty() {
                        break;
                    }

                    sender.send(buf.to_vec()).unwrap();

                    let consumed = buf.len();
                    file.consume(consumed);
                }
            });

            while let Ok(buf) = reciever.recv() {
                let buf_ref: &[Data] = unsafe { to_slice_with_lifetime(&buf) };

                for batch in buf_ref.chunks(batch_size) {
                    let adj = 2. / batch.len() as f32;
                    let gradients =
                        self.gradients::<Act>(nnue, batch, &mut error, reciprocal_scale);

                    self.optimiser
                        .update_weights(nnue, &gradients, adj, self.scheduler.lr());

                    if finished_batches % 500 == 0 {
                        let pct = finished_batches as f32 / batches as f32 * 100.0;
                        let positions = finished_batches * batch_size;
                        let pos_per_sec = positions as f32 / epoch_timer.elapsed().as_secs_f32();
                        print!(
                            "epoch {} [{}% ({}/{} batches, {} pos/sec)]\r",
                            ansi!(epoch, num_cs, esc),
                            ansi!(format!("{pct:.1}"), 35, esc),
                            ansi!(finished_batches, num_cs, esc),
                            ansi!(batches, num_cs, esc),
                            ansi!(format!("{pos_per_sec:.0}"), num_cs, esc),
                        );
                        let _ = stdout().flush();
                    }

                    finished_batches += 1;
                }
            }

            dataloader.join().unwrap();

            error /= num as f32;

            let epoch_time = epoch_timer.elapsed().as_secs_f32();

            println!(
                "epoch {} | time {} | running loss {} | {} pos/sec | total time {}",
                ansi!(epoch, num_cs, esc),
                ansi!(format!("{epoch_time:.2}"), num_cs, esc),
                ansi!(format!("{error:.6}"), num_cs, esc),
                ansi!(
                    format!("{:.0}", num.max(1) as f32 / epoch_time),
                    num_cs,
                    esc
                ),
                ansi!(format!("{:.2}", timer.elapsed().as_secs_f32()), num_cs, esc),
            );

            self.scheduler.adjust(epoch, num_cs, esc);

            if epoch % save_rate == 0 && epoch != max_epochs {
                let net_path = format!("nets/{net_name}-epoch{epoch}.bin");

                let qnnue = QuantisedNNUE::from_unquantised(nnue);
                qnnue.write_to_bin(&net_path).unwrap();

                println!("Saved [{}]", ansi!(net_path, "32;1"));
            }
        }
    }

    fn gradients<Act: Activation>(
        &self,
        nnue: &NNUEParams,
        batch: &[Data],
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
                    s.spawn(|| gradients::<Act>(chunk, nnue, error, blend, skip_prop, scale))
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
