use std::{
    fs::File,
    io::{BufRead, BufReader, Cursor, Seek, SeekFrom},
    sync::mpsc::{self, Receiver, SyncSender},
};

use crate::game::formats::bulletformat::ChessBoard;

use super::{
    rng::seeded_rng,
    shuffle::{SHUFFLE_THREADS, ShuffledDataLoader, convert_chunks},
};

use bullet_trainer::reader::DataReader;
use rayon::{ThreadPool, ThreadPoolBuilder};
pub use viriformat::{
    chess::{board::Board, chessmove::Move},
    dataformat::{Filter, Game, WDL},
};

#[derive(Clone)]
pub enum ViriFilter {
    Builtin(Filter),
    Custom(fn(&Board, Move, i16, f32) -> bool),
}

impl From<Filter> for ViriFilter {
    fn from(value: Filter) -> Self {
        Self::Builtin(value)
    }
}

#[derive(Clone)]
pub struct ViriBinpackLoader(ShuffledDataLoader<UnshuffledViriBinpackLoader>);

impl ViriBinpackLoader {
    pub fn new(path: &str, buffer_size_mb: usize, threads: usize, filter: impl Into<ViriFilter>) -> Self {
        Self::new_concat_multiple(&[path], buffer_size_mb, threads, filter)
    }

    pub fn new_concat_multiple(
        paths: &[&str],
        buffer_size_mb: usize,
        threads: usize,
        filter: impl Into<ViriFilter>,
    ) -> Self {
        let reader = UnshuffledViriBinpackLoader::new_concat_multiple(paths, threads, filter);
        Self(ShuffledDataLoader::new(reader, Some(buffer_size_mb), threads.min(SHUFFLE_THREADS)))
    }

    pub fn new_interleave_multiple(
        paths: &[&str],
        buffer_size_mb: usize,
        threads: usize,
        filter: impl Into<ViriFilter>,
    ) -> Self {
        let reader = UnshuffledViriBinpackLoader::new_interleave_multiple(paths, threads, filter);
        Self(ShuffledDataLoader::new(reader, Some(buffer_size_mb), threads.min(SHUFFLE_THREADS)))
    }
}

impl DataReader<ChessBoard> for ViriBinpackLoader {
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, skip_count: usize, f: F) {
        self.0.read_chunks(skip_count, f);
    }
}

#[derive(Clone)]
pub struct UnshuffledViriBinpackLoader {
    file_paths: Vec<String>,
    threads: usize,
    filter: ViriFilter,
    interleave: bool,
}

impl UnshuffledViriBinpackLoader {
    pub fn new(path: &str, threads: usize, filter: impl Into<ViriFilter>) -> Self {
        Self::new_concat_multiple(&[path], threads, filter)
    }

    pub fn new_concat_multiple(paths: &[&str], threads: usize, filter: impl Into<ViriFilter>) -> Self {
        Self {
            file_paths: paths.iter().map(|x| x.to_string()).collect(),
            threads: threads.max(1),
            filter: filter.into(),
            interleave: false,
        }
    }

    pub fn new_interleave_multiple(paths: &[&str], threads: usize, filter: impl Into<ViriFilter>) -> Self {
        Self { interleave: true, ..Self::new_concat_multiple(paths, threads, filter) }
    }
}

impl DataReader<ChessBoard> for UnshuffledViriBinpackLoader {
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, _: usize, mut f: F) {
        let file_paths = self.file_paths.clone();
        let threads = self.threads;
        let filter = self.filter.clone();
        let interleave = self.interleave;
        let pool = ThreadPoolBuilder::new().num_threads(threads).build().unwrap();

        let (sender, receiver) = mpsc::sync_channel::<Vec<Vec<u8>>>(4);
        let (msg_sender, msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            if interleave {
                read_interleave(&file_paths, threads, sender, msg_receiver);
            } else {
                read_concat(&file_paths, threads, sender, msg_receiver);
            }
        });

        let (game_sender, game_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(0);
        let (game_msg_sender, game_msg_receiver) = mpsc::sync_channel::<bool>(1);

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

fn read_concat(file_paths: &[String], threads: usize, sender: SyncSender<Vec<Vec<u8>>>, msg_receiver: Receiver<bool>) {
    let mut games = Vec::new();
    loop {
        let mut count = 0;
        for file_path in file_paths {
            let mut reader = BufReader::new(File::open(file_path).unwrap());

            loop {
                let mut buf = Vec::new();
                if Game::deserialise_fast_into_buffer(&mut reader, &mut buf).is_err() {
                    break;
                }
                count += 1;
                games.push(buf);

                if games.len().is_multiple_of(8192 * threads) {
                    if msg_receiver.try_recv().unwrap_or(false) || sender.send(games).is_err() {
                        return;
                    }
                    games = Vec::new();
                }
            }
        }
        if count == 0 {
            return;
        }
    }
}

fn read_interleave(
    file_paths: &[String],
    threads: usize,
    sender: SyncSender<Vec<Vec<u8>>>,
    msg_receiver: Receiver<bool>,
) {
    const GAMES_READ_PER_FILE: u64 = 16;

    let counts: Vec<u64> = file_paths
        .iter()
        .map(|path| {
            let mut reader = BufReader::new(File::open(path).unwrap());
            let mut count = 0;
            let mut buf = Vec::new();
            while !reader.fill_buf().unwrap().is_empty() {
                buf.clear();
                Game::deserialise_fast_into_buffer(&mut reader, &mut buf).unwrap();
                count += 1;
            }
            count
        })
        .collect();
    let total = counts.iter().sum::<u64>();
    if total == 0 {
        return;
    }

    let mut games = Vec::new();
    let mut rng = seeded_rng();

    'dataloading: loop {
        let mut streams: Vec<_> = counts.iter().map(|&count| (count, 0)).collect();
        let mut remaining = total;

        while remaining > 0 {
            let mut spot = rng.rand_range(0..remaining);
            let mut idx = 0;
            while streams[idx].0 <= spot {
                spot -= streams[idx].0;
                idx += 1;
            }

            let (games_left, offset) = &mut streams[idx];
            let mut reader = BufReader::new(File::open(&file_paths[idx]).unwrap());
            reader.seek(SeekFrom::Start(*offset)).unwrap();

            for _ in 0..GAMES_READ_PER_FILE.min(*games_left) {
                let mut buf = Vec::new();
                Game::deserialise_fast_into_buffer(&mut reader, &mut buf).unwrap();
                *offset += buf.len() as u64;
                *games_left -= 1;
                remaining -= 1;
                games.push(buf);

                if games.len().is_multiple_of(8192 * threads) {
                    if msg_receiver.try_recv().unwrap_or(false) || sender.send(games).is_err() {
                        break 'dataloading;
                    }

                    games = Vec::new();
                }
            }
        }
    }
}

fn convert_buffer(
    pool: &ThreadPool,
    sender: &SyncSender<Vec<ChessBoard>>,
    games: &[Vec<u8>],
    filter: &ViriFilter,
) -> bool {
    let chunk_size = games.len().div_ceil(pool.current_num_threads()).clamp(1, 1024);

    convert_chunks(pool, sender, games, chunk_size, |chunk| {
        let mut buffer = Vec::new();

        let mut reusable = Vec::new();
        for game_bytes in chunk {
            let game = Game::deserialise_from(&mut Cursor::new(game_bytes), reusable).unwrap();
            parse_into_buffer(&game, &mut buffer, filter);
            reusable = game.moves;
        }

        buffer
    })
}

fn parse_into_buffer(game: &Game, buffer: &mut Vec<ChessBoard>, filter: &ViriFilter) {
    match filter {
        ViriFilter::Builtin(filter) => {
            game.splat_to_bulletformat(
                |board| {
                    buffer.push(board);
                    Ok(())
                },
                filter,
            )
            .unwrap();
        }
        ViriFilter::Custom(filter) => {
            game.splat_to_bulletformat_with_filter_callback(
                |board| {
                    buffer.push(board);
                    Ok(())
                },
                |mv, eval, board, wdl, _| {
                    !filter(
                        board,
                        mv,
                        eval as i16,
                        match wdl {
                            WDL::Win => 1.0,
                            WDL::Draw => 0.5,
                            WDL::Loss => 0.0,
                        },
                    )
                },
            )
            .unwrap();
        }
    }
}
