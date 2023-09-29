use crate::scheduler::LrScheduler;

#[cfg(feature = "gpu")]
use cuda::{free_preallocations, preallocate ,update_weights, copy_weights_from_gpu, util::cuda_copy_to_gpu};

#[cfg(feature = "gpu")]
use cpu::NETWORK_SIZE;

use cpu::{quantise_and_write, NetworkParams, AdamW};

use common::{
    util::{to_slice_with_lifetime, write_to_bin},
    Data
};

use std::{
    fs::{create_dir, metadata, File},
    io::{stdout, BufRead, BufReader, Write},
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

#[repr(C)]
pub struct MetaData {
    pub epoch: usize,
}

impl MetaData {
    pub fn load(path: &str) -> Self {
        use std::io::Read;
        let mut file = File::open(path).unwrap();
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).unwrap();

        assert_eq!(buf.len(), std::mem::size_of::<MetaData>());

        let mut res = [0u8; std::mem::size_of::<MetaData>()];

        for (i, &byte) in buf.iter().enumerate() {
            res[i] = byte;
        }

        unsafe { std::mem::transmute(res) }
    }
}

pub struct Trainer {
    file: String,
    threads: usize,
    scheduler: LrScheduler,
    blend: f32,
    skip_prop: f32,
    pub optimiser: AdamW,
}

impl Trainer {
    #[must_use]
    pub fn new(
        file: String,
        threads: usize,
        scheduler: LrScheduler,
        blend: f32,
        skip_prop: f32,
        optimiser: AdamW,
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

    pub fn save(&self, nnue: &NetworkParams, name: &str, epoch: usize) {
        let path = format!("checkpoints/{name}");
        create_dir(&path).unwrap_or(());

        quantise_and_write(nnue, &format!("nets/{name}.bin"));

        let meta = MetaData {
            epoch: epoch + 1,
        };

        const META_SIZE: usize = std::mem::size_of::<MetaData>();
        write_to_bin::<MetaData, META_SIZE>(&meta, &format!("{path}/metadata.bin"), false).unwrap();
        nnue.write_to_bin(&format!("{path}/params.bin")).unwrap();
        self.optimiser.momentum.write_to_bin(&format!("{path}/momentum.bin")).unwrap();
        self.optimiser.velocity.write_to_bin(&format!("{path}/velocity.bin")).unwrap();
    }

    pub fn report_settings(&self, esc: &str) {
        println!("File Path      : {}", ansi!(self.file, "32;1", esc));
        println!("Threads        : {}", ansi!(self.threads, 31, esc));
        println!("WDL Proportion : {}", ansi!(self.blend, 31, esc));
        println!("Skip Proportion: {}", ansi!(self.skip_prop, 31, esc));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        nnue: &mut NetworkParams,
        start_epoch: usize,
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
        let rscale = 1.0 / scale;
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
        println!("Positions      : {}", ansi!(num, 31, esc));

        // fast forward lr scheduler
        for i in 1..start_epoch {
            self.scheduler.adjust(i, num_cs, esc)
        }

        let timer = Instant::now();

        let mut error;

        #[cfg(feature = "gpu")]
        let ptrs = preallocate(batch_size);

        #[cfg(feature = "gpu")]
        {
            cuda_copy_to_gpu(ptrs.7, nnue as *const NetworkParams, 1);
            cuda_copy_to_gpu(ptrs.8, self.optimiser.momentum.as_ptr(), NETWORK_SIZE);
            cuda_copy_to_gpu(ptrs.9, self.optimiser.velocity.as_ptr(), NETWORK_SIZE);
        }

        for epoch in start_epoch..=max_epochs {
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
                    #[cfg(not(feature = "gpu"))]
                    {
                        use crate::gradient::gradients_batch_cpu;
                        let gradients = gradients_batch_cpu(batch, nnue, &mut error, rscale, self.blend, self.skip_prop, self.threads);
                        self.optimiser
                            .update_weights(nnue, &gradients, adj, self.scheduler.lr());
                    }

                    #[cfg(feature = "gpu")]
                    unsafe {
                        use crate::gradient::gradients_batch_gpu;
                        gradients_batch_gpu(batch, &mut error, rscale, self.blend, self.skip_prop, self.threads, ptrs);
                        update_weights(adj, self.optimiser.decay, self.scheduler.lr(), ptrs);
                    }

                    if finished_batches % 128 == 0 {
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

            if epoch % save_rate == 0 || epoch == max_epochs {
                let net_path = format!("{net_name}-epoch{epoch}");

                #[cfg(feature = "gpu")]
                copy_weights_from_gpu(
                    nnue,
                    &mut self.optimiser.momentum,
                    &mut self.optimiser.velocity,
                    ptrs
                );

                self.save(nnue, &net_path, epoch);

                println!("Saved [{}]", ansi!(net_path, "32;1"));
            }
        }

        #[cfg(feature = "gpu")]
        free_preallocations(ptrs);
    }
}
