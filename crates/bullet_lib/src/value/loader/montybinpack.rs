use std::{
    fs::File,
    io::{BufReader, Cursor},
    sync::mpsc::{self, SyncSender},
};

use crate::game::formats::bulletformat::ChessBoard;

use super::shuffle::{SHUFFLE_THREADS, ShuffledDataLoader, convert_chunks};

use bullet_trainer::reader::DataReader;
use montyformat::{
    FastDeserialise, MontyValueFormat,
    chess::{Move, Position},
};
use rayon::{ThreadPool, ThreadPoolBuilder};

#[derive(Clone)]
pub struct MontyBinpackLoader<T: Fn(&Position, Move, i16, f32) -> bool>(
    ShuffledDataLoader<UnshuffledMontyBinpackLoader<T>>,
);

impl<T: Fn(&Position, Move, i16, f32) -> bool> MontyBinpackLoader<T> {
    pub fn new(path: &str, buffer_size_mb: usize, threads: usize, filter: T) -> Self {
        Self::new_concat_multiple(&[path], buffer_size_mb, threads, filter)
    }

    pub fn new_concat_multiple(paths: &[&str], buffer_size_mb: usize, threads: usize, filter: T) -> Self {
        let reader = UnshuffledMontyBinpackLoader::new_concat_multiple(paths, threads, filter);
        Self(ShuffledDataLoader::new(reader, Some(buffer_size_mb), threads.min(SHUFFLE_THREADS)))
    }
}

impl<T> DataReader<ChessBoard> for MontyBinpackLoader<T>
where
    T: Fn(&Position, Move, i16, f32) -> bool + Clone + Send + Sync + 'static,
{
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, skip_count: usize, f: F) {
        self.0.read_chunks(skip_count, f);
    }
}

#[derive(Clone)]
pub struct UnshuffledMontyBinpackLoader<T: Fn(&Position, Move, i16, f32) -> bool> {
    file_paths: Vec<String>,
    threads: usize,
    filter: T,
}

impl<T: Fn(&Position, Move, i16, f32) -> bool> UnshuffledMontyBinpackLoader<T> {
    pub fn new(path: &str, threads: usize, filter: T) -> Self {
        Self::new_concat_multiple(&[path], threads, filter)
    }

    pub fn new_concat_multiple(paths: &[&str], threads: usize, filter: T) -> Self {
        Self { file_paths: paths.iter().map(|x| x.to_string()).collect(), threads: threads.max(1), filter }
    }
}

impl<T> DataReader<ChessBoard> for UnshuffledMontyBinpackLoader<T>
where
    T: Fn(&Position, Move, i16, f32) -> bool + Clone + Send + Sync + 'static,
{
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, _: usize, mut f: F) {
        let file_paths = self.file_paths.clone();
        let threads = self.threads;

        let (sender, receiver) = mpsc::sync_channel::<Vec<Vec<u8>>>(4);
        let (msg_sender, msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut games = Vec::new();

            'dataloading: loop {
                for file_path in &file_paths {
                    let mut reader = BufReader::new(File::open(file_path.as_str()).unwrap());

                    let mut buffer = Vec::new();
                    while let Ok(()) = MontyValueFormat::deserialise_fast_into_buffer(&mut reader, &mut buffer) {
                        games.push(std::mem::take(&mut buffer));

                        if games.len().is_multiple_of(8192 * threads) {
                            if msg_receiver.try_recv().unwrap_or(false) || sender.send(games).is_err() {
                                break 'dataloading;
                            }

                            games = Vec::new();
                        }
                    }
                }
            }
        });

        let (game_sender, game_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(4 * self.threads);
        let (game_msg_sender, game_msg_receiver) = mpsc::sync_channel::<bool>(1);

        let filter = self.filter.clone();
        let pool = ThreadPoolBuilder::new().num_threads(threads).build().unwrap();

        std::thread::spawn(move || {
            'dataloading: while let Ok(games) = receiver.recv() {
                if game_msg_receiver.try_recv().unwrap_or(false) {
                    let _ = msg_sender.send(true);
                    break 'dataloading;
                }

                if convert_buffer(&pool, &game_sender, &games, &filter) {
                    let _ = msg_sender.send(true);
                    break 'dataloading;
                }
            }
        });

        'dataloading: while let Ok(games) = game_receiver.recv() {
            if f(&games) {
                let _ = game_msg_sender.send(true);
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
