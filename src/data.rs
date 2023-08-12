use std::{fs::File, io::{BufRead, BufReader}, thread, time::Instant};
use crate::{arch::{NNUEParams, update_single_grad}, position::Position};

#[derive(Default)]
pub struct Data(Vec<Position>, pub usize);

impl Data {
    pub fn num(&self) -> f64 {
        self.0.len() as f64
    }

    pub fn add_contents(&mut self, file_name: &str) {
        let timer = Instant::now();
        let (mut wins, mut losses, mut draws) = (0, 0, 0);
        let file = File::open(file_name).unwrap();
        for line in BufReader::new(file).lines().map(|ln| ln.unwrap()) {
            let res: Position = line.parse().unwrap();
            let int = (res.result * 2.0) as u64;
            match int {
                2 => wins += 1,
                0 => losses += 1,
                1 => draws += 1,
                _ => unreachable!(),
            }
            self.0.push(res);
        }
        let elapsed = timer.elapsed().as_secs_f64();
        let pps = self.num() / elapsed;
        println!("{} positions in {elapsed:.2} seconds, {pps:.2} pos/sec", self.num());
        println!("wins {wins} losses {losses} draws {draws}");
    }

    pub fn gradients(&self, nnue: &NNUEParams, error: &mut f64) -> NNUEParams {
        let size = self.0.len() / self.1;
        let mut errors = vec![0.0; self.1];
        let grad = thread::scope(|s| {
            self.0
                .chunks(size)
                .zip(errors.iter_mut())
                .map(|(chunk, error)| s.spawn(|| gradients_batch(chunk, nnue, error)))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|p| p.join().unwrap_or_default())
                .fold(NNUEParams::default(), |a, b| a + b)
        });
        *error = errors.iter().sum();
        grad
    }
}

fn gradients_batch(positions: &[Position], nnue: &NNUEParams, error: &mut f64) -> NNUEParams {
    let mut grad = NNUEParams::default();
    for pos in positions {
        update_single_grad(pos, nnue, &mut grad, error)
    }
    grad
}