use std::{
    fs::File,
    io::{BufRead, BufReader, Cursor, Seek, SeekFrom},
    sync::mpsc::{self, Receiver, SyncSender},
};

use crate::game::formats::bulletformat::ChessBoard;

use super::rng::seeded_rng;

use bullet_trainer::reader::DataReader;
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
pub struct ViriBinpackLoader {
    file_paths: Vec<String>,
    buffer_size: usize,
    threads: usize,
    filter: ViriFilter,
    interleave: bool,
}

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
        Self {
            file_paths: paths.iter().map(|x| x.to_string()).collect(),
            buffer_size: buffer_size_mb * 1024 * 1024 / std::mem::size_of::<ChessBoard>() / 2,
            threads,
            filter: filter.into(),
            interleave: false,
        }
    }

    pub fn new_interleave_multiple(
        paths: &[&str],
        buffer_size_mb: usize,
        threads: usize,
        filter: impl Into<ViriFilter>,
    ) -> Self {
        Self { interleave: true, ..Self::new_concat_multiple(paths, buffer_size_mb, threads, filter) }
    }
}

impl DataReader<ChessBoard> for ViriBinpackLoader {
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, _: usize, mut f: F) {
        let mut shuffle_buffer = Vec::new();
        shuffle_buffer.reserve_exact(self.buffer_size);

        let file_paths = self.file_paths.clone();
        let buffer_size = self.buffer_size;
        let threads = self.threads;
        let filter = self.filter.clone();
        let interleave = self.interleave;

        let (sender, receiver) = mpsc::sync_channel::<Vec<Vec<u8>>>(4);
        let (msg_sender, msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            if interleave {
                read_interleave(&file_paths, threads, sender, msg_receiver);
            } else {
                read_concat(&file_paths, threads, sender, msg_receiver);
            }
        });

        let (game_sender, game_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(4 * self.threads);
        let (game_msg_sender, game_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            'dataloading: while let Ok(games) = receiver.recv() {
                if game_msg_receiver.try_recv().unwrap_or(false) {
                    msg_sender.send(true).unwrap();
                    break 'dataloading;
                }

                convert_buffer(threads, &game_sender, &games, &filter);
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
            if f(&shuffle_buffer) {
                buffer_msg_sender.send(true).unwrap();
                break 'dataloading;
            }
        }

        drop(buffer_receiver);
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

fn convert_buffer(threads: usize, sender: &SyncSender<Vec<ChessBoard>>, games: &[Vec<u8>], filter: &ViriFilter) {
    let chunk_size = games.len().div_ceil(threads);

    std::thread::scope(|s| {
        for chunk in games.chunks(chunk_size) {
            let this_sender = sender.clone();
            s.spawn(move || {
                let mut buffer = Vec::new();

                let mut reusable = Vec::new();
                for game_bytes in chunk {
                    let game = Game::deserialise_from(&mut Cursor::new(game_bytes), reusable).unwrap();
                    parse_into_buffer(&game, &mut buffer, filter);
                    reusable = game.moves;
                }

                this_sender.send(buffer)
            });
        }
    });
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

fn shuffle(data: &mut [ChessBoard]) {
    let mut rng = seeded_rng();

    for i in (0..data.len()).rev() {
        let idx = rng.rand_range(0..i as u64 + 1) as usize;
        data.swap(idx, i);
    }
}
