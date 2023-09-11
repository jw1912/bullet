mod gradient;
mod optimiser;
mod scheduler;
mod trainer;

use cpu::NetworkParams;


use optimiser::AdamW;
use scheduler::{LrScheduler, SchedulerType};
use trainer::{Trainer, MetaData};

fn main() {
    let mut args = std::env::args();
    args.next();

    // all of these will be provided and validated by the `run.py` script
    let file_path = args.next().unwrap();
    let net_name = &args.next().unwrap();
    let threads = args.next().unwrap().parse().unwrap();
    let lr_start = args.next().unwrap().parse().unwrap();
    let blend = args.next().unwrap().parse().unwrap();
    let max_epochs: usize = args.next().unwrap().parse().unwrap();
    let batch_size = args.next().unwrap().parse().unwrap();
    let save_rate = args.next().unwrap().parse().unwrap();
    let skip_prop = args.next().unwrap().parse().unwrap();
    let lr_end: f32 = args.next().unwrap().parse().unwrap();
    let lr_step = args.next().unwrap().parse().unwrap();
    let lr_drop = args.next().unwrap().parse().unwrap();
    let lr_gamma = args.next().unwrap().parse().unwrap();
    let scale = args.next().unwrap().parse().unwrap();
    let cbcs = args.next().unwrap().parse().unwrap();
    let resume: String = args.next().unwrap().parse().unwrap();

    let mut scheduler = LrScheduler::new(lr_start, 1.0, SchedulerType::Drop(1000));

    if lr_end != 0.0 {
        scheduler.set_type(SchedulerType::Step(1));
        let gamma = (lr_start / lr_end).ln() / (max_epochs - 1).max(1) as f32;
        scheduler.set_gamma((-gamma).exp());
    }

    if lr_step != 0 {
        scheduler.set_type(SchedulerType::Step(lr_step));
        scheduler.set_gamma(lr_gamma);
    }

    if lr_drop != 0 {
        scheduler.set_type(SchedulerType::Drop(lr_drop));
        scheduler.set_gamma(lr_gamma);
    }

    let optimiser = AdamW::default();

    let mut trainer = Trainer::new(file_path, threads, scheduler, blend, skip_prop, optimiser);
    let mut params = NetworkParams::random();
    let mut start_epoch = 1;

    if resume != "no_way" {
        let meta = MetaData::load(&format!("{resume}/metadata.bin"));
        start_epoch = meta.epoch;
        println!("Resuming at epoch {}...", start_epoch);
        trainer.optimiser.momentum.load_from_bin(&format!("{resume}/momentum.bin"));
        trainer.optimiser.velocity.load_from_bin(&format!("{resume}/velocity.bin"));
        params.load_from_bin(&format!("{resume}/params.bin"));
    };

    // carry out tuning
    trainer.run(
        &mut params,
        start_epoch,
        max_epochs,
        net_name,
        save_rate,
        batch_size,
        scale,
        cbcs,
    );
}
