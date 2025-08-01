use std::{
    fs::File,
    io::{BufReader, Cursor},
    sync::mpsc::{self, SyncSender},
};

use crate::game::formats::bulletformat::ChessBoard;

use super::{rng::SimpleRand, DataLoader};

use montyformat::{
    chess::{Move, Position},
    FastDeserialise, MontyValueFormat,
};

#[derive(Clone)]
pub struct MontyBinpackLoader<T: Fn(&Position, Move, i16, f32) -> bool> {
    file_paths: Vec<String>,
    buffer_size: usize,
    threads: usize,
    filter: T,
}

impl<T: Fn(&Position, Move, i16, f32) -> bool> MontyBinpackLoader<T> {
    pub fn new(path: &str, buffer_size_mb: usize, threads: usize, filter: T) -> Self {
        Self::new_concat_multiple(&[path], buffer_size_mb, threads, filter)
    }

    pub fn new_concat_multiple(paths: &[&str], buffer_size_mb: usize, threads: usize, filter: T) -> Self {
        Self {
            file_paths: paths.iter().map(|x| x.to_string()).collect(),
            buffer_size: buffer_size_mb * 1024 * 1024 / std::mem::size_of::<ChessBoard>() / 2,
            threads,
            filter,
        }
    }
}

impl<T> DataLoader<ChessBoard> for MontyBinpackLoader<T>
where
    T: Fn(&Position, Move, i16, f32) -> bool + Clone + Send + Sync + 'static,
{
    fn data_file_paths(&self) -> &[String] {
        &self.file_paths
    }

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_batches<F: FnMut(&[ChessBoard]) -> bool>(&self, _: usize, batch_size: usize, mut f: F) {
        let mut shuffle_buffer = Vec::new();
        shuffle_buffer.reserve_exact(self.buffer_size);

        let file_paths = self.file_paths.clone();
        let buffer_size = self.buffer_size;

        let (sender, receiver) = mpsc::sync_channel::<Vec<u8>>(256);
        let (msg_sender, msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || 'dataloading: loop {
            for file_path in &file_paths {
                let mut reader = BufReader::new(File::open(file_path.as_str()).unwrap());

                let mut buffer = Vec::new();
                while let Ok(()) = MontyValueFormat::deserialise_fast_into_buffer(&mut reader, &mut buffer) {
                    if msg_receiver.try_recv().unwrap_or(false) || sender.send(buffer).is_err() {
                        break 'dataloading;
                    }

                    buffer = Vec::new();
                }
            }
        });

        let (game_sender, game_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(4 * self.threads);
        let (game_msg_sender, game_msg_receiver) = mpsc::sync_channel::<bool>(1);

        let threads = self.threads;
        let filter = self.filter.clone();

        std::thread::spawn(move || {
            let mut reusable = Vec::new();
            'dataloading: while let Ok(game_bytes) = receiver.recv() {
                if game_msg_receiver.try_recv().unwrap_or(false) {
                    msg_sender.send(true).unwrap();
                    break 'dataloading;
                }

                reusable.push(game_bytes);

                if reusable.len() % (8192 * threads) == 0 {
                    convert_buffer(threads, &game_sender, &reusable, &filter);
                    reusable.clear();
                }
            }
        });

        let (buffer_sender, buffer_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(0);
        let (buffer_msg_sender, buffer_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            'dataloading: while let Ok(game) = game_receiver.recv() {
                if buffer_msg_receiver.try_recv().unwrap_or(false) {
                    game_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }

                if shuffle_buffer.len() + game.len() < shuffle_buffer.capacity() {
                    shuffle_buffer.extend_from_slice(&game);
                } else {
                    let diff = shuffle_buffer.capacity() - shuffle_buffer.len();
                    if diff > 0 {
                        shuffle_buffer.extend_from_slice(&game[..diff]);
                    }

                    shuffle(&mut shuffle_buffer);

                    if buffer_msg_receiver.try_recv().unwrap_or(false) || buffer_sender.send(shuffle_buffer).is_err() {
                        game_msg_sender.send(true).unwrap();
                        break 'dataloading;
                    }

                    shuffle_buffer = Vec::new();
                    shuffle_buffer.reserve_exact(buffer_size);
                    shuffle_buffer.extend_from_slice(&game[diff..]);
                }
            }
        });

        'dataloading: while let Ok(shuffle_buffer) = buffer_receiver.recv() {
            for batch in shuffle_buffer.chunks(batch_size) {
                let should_break = f(batch);

                if should_break {
                    buffer_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }
            }
        }

        drop(buffer_receiver);
    }
}

fn convert_buffer<T: Fn(&Position, Move, i16, f32) -> bool + Send + Sync>(
    threads: usize,
    sender: &SyncSender<Vec<ChessBoard>>,
    games: &[Vec<u8>],
    filter: &T,
) {
    let chunk_size = games.len().div_ceil(threads);

    std::thread::scope(|s| {
        for chunk in games.chunks(chunk_size) {
            let this_sender = sender.clone();
            s.spawn(move || {
                let mut buffer = Vec::new();

                for game_bytes in chunk {
                    parse_into_buffer(game_bytes, &mut buffer, filter);
                }

                this_sender.send(buffer)
            });
        }
    });
}

fn parse_into_buffer<T: Fn(&Position, Move, i16, f32) -> bool>(
    game_bytes: &[u8],
    buffer: &mut Vec<ChessBoard>,
    filter: &T,
) {
    let mut reader = Cursor::new(game_bytes);
    let game = MontyValueFormat::deserialise_from(&mut reader, Vec::new()).unwrap();

    let mut pos = game.startpos;
    let castling = game.castling;

    for data in game.moves {
        if filter(&pos, data.best_move, data.score, game.result) {
            buffer.push(ChessBoard::from_raw(pos.bbs(), pos.stm(), data.score, game.result).unwrap());
        }

        pos.make(data.best_move, &castling);
    }
}

fn shuffle(data: &mut [ChessBoard]) {
    let mut rng = SimpleRand::with_seed();

    for i in (0..data.len()).rev() {
        let idx = rng.rng() as usize % (i + 1);
        data.swap(idx, i);
    }
}
