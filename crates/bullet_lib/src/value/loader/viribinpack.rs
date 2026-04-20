use std::{
    fs::File,
    io::{BufReader, Cursor},
    sync::mpsc::{self, SyncSender},
};

use crate::game::formats::bulletformat::ChessBoard;

use super::{DataLoader, rng::SimpleRand};

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
        }
    }
}

impl DataLoader<ChessBoard> for ViriBinpackLoader {
    fn data_file_paths(&self) -> &[String] {
        &self.file_paths
    }

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, _: usize, mut f: F) {
        let mut shuffle_buffer = Vec::new();
        shuffle_buffer.reserve_exact(self.buffer_size);

        let file_paths = self.file_paths.clone();
        let buffer_size = self.buffer_size;
        let threads = self.threads;
        let filter = self.filter.clone();

        let (sender, receiver) = mpsc::sync_channel::<Vec<Vec<u8>>>(256);
        let (msg_sender, msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut games = Vec::new();

            'dataloading: loop {
                for file_path in &file_paths {
                    let mut reader = BufReader::new(File::open(file_path.as_str()).unwrap());

                    loop {
                        let mut buf = Vec::new();
                        if Game::deserialise_fast_into_buffer(&mut reader, &mut buf).is_err() {
                            break;
                        }

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
    let mut rng = SimpleRand::with_seed();

    for i in (0..data.len()).rev() {
        let idx = rng.rng() as usize % (i + 1);
        data.swap(idx, i);
    }
}
