use std::{
    fs::File,
    io::{BufReader, Cursor},
    sync::{
        Arc,
        mpsc::{self, SyncSender},
    },
};

use crate::game::formats::bulletformat::ChessBoard;

use super::shuffle::{Shuffle, convert_chunks};

use bullet_trainer::reader::DataReader;
use montyformat::{
    FastDeserialise, MontyValueFormat,
    chess::{Move, Position},
};
use rayon::ThreadPool;

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
            threads: threads.max(1),
            filter,
        }
    }
}

impl<T> DataReader<ChessBoard> for MontyBinpackLoader<T>
where
    T: Fn(&Position, Move, i16, f32) -> bool + Clone + Send + Sync + 'static,
{
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, _: usize, mut f: F) {
        let buffer_size = self.buffer_size.max(1);
        let mut shuffle_buffer = Vec::new();
        shuffle_buffer.reserve_exact(buffer_size);
        let mut shuffle = Shuffle::new(self.threads);

        let file_paths = self.file_paths.clone();

        let (sender, receiver) = mpsc::sync_channel::<Vec<u8>>(4);
        let (msg_sender, msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            'dataloading: loop {
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
            }
        });

        let (game_sender, game_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(4 * self.threads);
        let (game_msg_sender, game_msg_receiver) = mpsc::sync_channel::<bool>(1);

        let threads = self.threads;
        let filter = self.filter.clone();
        let decode_pool = Arc::clone(&shuffle.pool);

        std::thread::spawn(move || {
            let mut reusable = Vec::new();
            
            'dataloading: while let Ok(game_bytes) = receiver.recv() {
                if game_msg_receiver.try_recv().unwrap_or(false) {
                    let _ = msg_sender.send(true);
                    break 'dataloading;
                }

                reusable.push(game_bytes);

                if reusable.len() % (8192 * threads) == 0 {
                    if convert_buffer(&decode_pool, &game_sender, &reusable, &filter) {
                        let _ = msg_sender.send(true);
                        break 'dataloading;
                    }

                    reusable.clear();
                }
            }
        });

        let (buffer_sender, buffer_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(0);
        let (buffer_msg_sender, buffer_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            'dataloading: while let Ok(game) = game_receiver.recv() {
                if buffer_msg_receiver.try_recv().unwrap_or(false) {
                    let _ = game_msg_sender.send(true);
                    break 'dataloading;
                }

                let mut remaining = game.as_slice();
                while !remaining.is_empty() {
                    let count = remaining.len().min(buffer_size - shuffle_buffer.len());
                    shuffle_buffer.extend_from_slice(&remaining[..count]);
                    remaining = &remaining[count..];

                    if shuffle_buffer.len() == buffer_size {
                        shuffle.shuffle(&mut shuffle_buffer);

                        if buffer_msg_receiver.try_recv().unwrap_or(false)
                            || buffer_sender.send(shuffle_buffer).is_err()
                        {
                            let _ = game_msg_sender.send(true);
                            break 'dataloading;
                        }

                        shuffle_buffer = Vec::new();
                        shuffle_buffer.reserve_exact(buffer_size);
                    }
                }
            }
        });

        'dataloading: while let Ok(shuffle_buffer) = buffer_receiver.recv() {
            if f(&shuffle_buffer) {
                let _ = buffer_msg_sender.send(true);
                break 'dataloading;
            }
        }
    }
}

fn convert_buffer<T: Fn(&Position, Move, i16, f32) -> bool + Send + Sync>(
    pool: &ThreadPool,
    sender: &SyncSender<Vec<ChessBoard>>,
    games: &[Vec<u8>],
    filter: &T,
) -> bool {
    convert_chunks(pool, sender, games, 1024, |chunk| {
        let mut buffer = Vec::new();

        for game_bytes in chunk {
            parse_into_buffer(game_bytes, &mut buffer, filter);
        }

        buffer
    })
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
