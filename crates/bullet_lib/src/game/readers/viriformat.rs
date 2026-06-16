use std::{
    fs::File,
    io::{BufReader, Cursor},
    mem,
    sync::mpsc,
    thread,
};

use bullet_trainer::{reader::DataReader, rng::Rng};

use bulletformat::ChessBoard;
pub use viriformat::{
    chess::{
        board::Board,
        chessmove::{Move, MoveFlags},
        piece::{Colour, Piece, PieceType},
    },
    dataformat::{Game, WDL},
};

use crate::{game::formats::ChessDatapoint, value::loader::GameResult};

#[derive(Clone)]
pub struct ViriformatReader<F> {
    file_paths: Vec<String>,
    buffer_size_mb: usize,
    threads: usize,
    filter: F,
}

impl<F> DataReader<ChessBoard> for ViriformatReader<F>
where
    F: Fn(&Board, Move, i16, WDL, &mut Rng) -> bool + Clone + Send + Sync,
{
    fn read_chunks<T: FnMut(&[ChessBoard]) -> bool>(&self, _: usize, f: T) {
        thread::scope(|s| self.read(s, &|board, score, wdl| board.to_bulletformat(wdl as u8, score).unwrap(), f));
    }
}

impl<F> DataReader<ChessDatapoint> for ViriformatReader<F>
where
    F: Fn(&Board, Move, i16, WDL, &mut Rng) -> bool + Clone + Send + Sync,
{
    fn read_chunks<T: FnMut(&[ChessDatapoint]) -> bool>(&self, _: usize, f: T) {
        thread::scope(|s| {
            self.read(
                s,
                &|board, score, wdl| ChessDatapoint {
                    bbs: [
                        board.pieces.colours[0].inner(),
                        board.pieces.all_pawns().inner(),
                        board.pieces.all_knights().inner(),
                        board.pieces.all_bishops().inner(),
                        board.pieces.all_rooks().inner(),
                        board.pieces.all_queens().inner(),
                        board.pieces.all_queens().inner(),
                    ],
                    stm: board.turn() == Colour::Black,
                    score,
                    result: match wdl {
                        WDL::Loss => GameResult::Loss,
                        WDL::Draw => GameResult::Draw,
                        WDL::Win => GameResult::Win,
                    },
                    fullm: board.full_move_number().try_into().unwrap(),
                    halfm: board.fifty_move_counter(),
                },
                f,
            )
        });
    }
}

impl<F> ViriformatReader<F>
where
    F: Fn(&Board, Move, i16, WDL, &mut Rng) -> bool + Sync,
{
    pub fn new(path: &str, buffer_size_mb: usize, threads: usize, should_keep: F) -> Self {
        Self::new_concat_multiple(&[path], buffer_size_mb, threads, should_keep)
    }

    pub fn new_concat_multiple(paths: &[&str], buffer_size_mb: usize, threads: usize, should_keep: F) -> Self {
        Self { file_paths: paths.iter().map(|x| x.to_string()).collect(), buffer_size_mb, threads, filter: should_keep }
    }

    pub fn read<'a, 'b, D, M, T>(&'a self, s: &'a thread::Scope<'a, 'b>, map: &'a M, mut func: T)
    where
        D: Clone + Send + 'static,
        M: Fn(&Board, i16, WDL) -> D + Sync,
        T: FnMut(&[D]) -> bool,
    {
        let threads = self.threads;
        let buffer_size = (self.buffer_size_mb * 1024 * 1024).div_ceil(mem::size_of::<D>());

        let mut shuffle_buffer = Vec::new();
        shuffle_buffer.reserve_exact(buffer_size);

        let mut senders = Vec::with_capacity(threads);
        let mut receivers = Vec::with_capacity(threads);
        for _ in 0..threads {
            let (sender, receiver) = mpsc::sync_channel::<Vec<Vec<u8>>>(2);
            senders.push(sender);
            receivers.push(receiver);
        }

        let mut rng = Rng::seeded();

        s.spawn(move || {
            let mut games = Vec::new();
            let mut tick = 0;

            'dataloading: loop {
                for file_path in &self.file_paths {
                    let mut reader = BufReader::new(File::open(file_path.as_str()).unwrap());

                    loop {
                        let mut buf = Vec::new();
                        if Game::deserialise_fast_into_buffer(&mut reader, &mut buf).is_err() {
                            break;
                        }

                        games.push(buf);

                        if games.len().is_multiple_of(8192) {
                            if senders[tick % threads].send(games).is_err() {
                                break 'dataloading;
                            }

                            games = Vec::new();
                            tick += 1;
                        }
                    }
                }
            }
        });

        let (game_sender, game_receiver) = mpsc::sync_channel::<Vec<D>>(4 * self.threads);

        for receiver in receivers {
            let mut trng = Rng::new(rng.sample());
            let sender = game_sender.clone();

            s.spawn(move || {
                let mut reusable = Vec::new();

                'dataloading: while let Ok(games) = receiver.recv() {
                    let mut buffer = Vec::new();

                    for game_bytes in games {
                        let game = Game::deserialise_from(&mut Cursor::new(game_bytes), reusable).unwrap();
                        parse_into_buffer(&game, &mut buffer, &mut trng, map, &self.filter);
                        reusable = game.moves;
                    }

                    if sender.send(buffer).is_err() {
                        break 'dataloading;
                    }
                }
            });
        }

        let (buffer_sender, buffer_receiver) = mpsc::sync_channel::<Vec<D>>(0);

        s.spawn(move || {
            'dataloading: while let Ok(game) = game_receiver.recv() {
                if shuffle_buffer.len() + game.len() < shuffle_buffer.capacity() {
                    shuffle_buffer.extend_from_slice(&game);
                } else {
                    let diff = shuffle_buffer.capacity() - shuffle_buffer.len();
                    if diff > 0 {
                        shuffle_buffer.extend_from_slice(&game[..diff]);
                    }

                    rng.shuffle(&mut shuffle_buffer);

                    if buffer_sender.send(shuffle_buffer).is_err() {
                        break 'dataloading;
                    }

                    shuffle_buffer = Vec::new();
                    shuffle_buffer.reserve_exact(buffer_size);
                    shuffle_buffer.extend_from_slice(&game[diff..]);
                }
            }
        });

        'dataloading: while let Ok(shuffle_buffer) = buffer_receiver.recv() {
            if func(&shuffle_buffer) {
                break 'dataloading;
            }
        }
    }
}

fn parse_into_buffer<D, M, F>(game: &Game, buffer: &mut Vec<D>, rng: &mut Rng, map: &M, should_keep: &F)
where
    M: Fn(&Board, i16, WDL) -> D,
    F: Fn(&Board, Move, i16, WDL, &mut Rng) -> bool,
{
    let (mut board, _, wdl, _) = game.initial_position.unpack();
    let outcome = WDL::from_packed(wdl);

    for &(mv, eval) in &game.moves {
        let eval = eval.get();
        if should_keep(&board, mv, eval, outcome, rng) {
            buffer.push(map(&board, eval, outcome));
        }
        board.make_move_simple(mv);
    }
}
