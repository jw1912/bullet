use std::{
    fs::File,
    io::BufReader,
    sync::mpsc::{self, SyncSender},
    thread,
};

use bullet_trainer::reader::DataReader;
use sfbinpack::{ChunkReader, read_chunk_into};
pub use sfbinpack::{
    TrainingDataEntry,
    chess::{
        color::Color,
        r#move::{Move, MoveType},
        piecetype::PieceType,
    },
};

use crate::game::formats::bulletformat::ChessBoard;

use super::rng::SimpleRand;

fn convert_to_bulletformat(entry: &TrainingDataEntry) -> ChessBoard {
    let mut bbs = [0; 8];

    let stm = usize::from(entry.pos.side_to_move().ordinal());
    let pc_bb =
        |pt| entry.pos.pieces_bb_color(Color::Black, pt).bits() | entry.pos.pieces_bb_color(Color::White, pt).bits();

    bbs[0] = entry.pos.pieces_bb(Color::White).bits();
    bbs[1] = entry.pos.pieces_bb(Color::Black).bits();
    bbs[2] = pc_bb(PieceType::Pawn);
    bbs[3] = pc_bb(PieceType::Knight);
    bbs[4] = pc_bb(PieceType::Bishop);
    bbs[5] = pc_bb(PieceType::Rook);
    bbs[6] = pc_bb(PieceType::Queen);
    bbs[7] = pc_bb(PieceType::King);

    let mut score = entry.score;
    let mut result = f32::from(1 + entry.result) / 2.0;

    if stm > 0 {
        score = -score;
        result = 1.0 - result;
    }

    ChessBoard::from_raw(bbs, stm, score, result).expect("Binpack must be malformed!")
}

#[derive(Clone)]
pub struct SfBinpackLoader<T: Fn(&TrainingDataEntry) -> bool> {
    file_paths: Vec<String>,
    buffer_size: usize,
    threads: usize,
    filter: T,
}

impl<T: Fn(&TrainingDataEntry) -> bool> SfBinpackLoader<T> {
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

impl<T> DataReader<ChessBoard> for SfBinpackLoader<T>
where
    T: Fn(&TrainingDataEntry) -> bool + Clone + Send + Sync + 'static,
{
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, _: usize, mut f: F) {
        let file_paths = self.file_paths.clone();
        let buffer_size = self.buffer_size;
        let threads = self.threads;
        let filter = self.filter.clone();
        let reader_buffer_size = threads.max(1);

        let (reader_sender, reader_receiver) = mpsc::sync_channel::<Vec<Vec<u8>>>(4);
        let (reader_msg_sender, reader_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut games = Vec::new();

            'dataloading: loop {
                for file_path in &file_paths {
                    let mut reader = BufReader::new(File::open(file_path.as_str()).unwrap());

                    let mut chunk = Vec::new();

                    while read_chunk_into(&mut reader, &mut chunk).unwrap() {
                        games.push(std::mem::take(&mut chunk));

                        if games.len() == reader_buffer_size {
                            if reader_msg_receiver.try_recv().unwrap_or(false) || reader_sender.send(games).is_err() {
                                break 'dataloading;
                            }

                            games = Vec::new();
                        }
                    }

                    if !games.is_empty() {
                        if reader_msg_receiver.try_recv().unwrap_or(false) || reader_sender.send(games).is_err() {
                            break 'dataloading;
                        }

                        games = Vec::new();
                    }
                }
            }
        });

        let (converted_sender, converted_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(4 * threads);
        let (converted_msg_sender, converted_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let filter = &filter;
            let mut should_break = false;
            'dataloading: while let Ok(chunks) = reader_receiver.recv() {
                if should_break || converted_msg_receiver.try_recv().unwrap_or(false) {
                    reader_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }

                should_break = convert_buffer(threads, &converted_sender, &chunks, filter);

                if should_break {
                    reader_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }
            }
        });

        let (buffer_sender, buffer_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(0);
        let (buffer_msg_sender, buffer_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut shuffle_buffer = Vec::with_capacity(buffer_size);

            'dataloading: while let Ok(converted) = converted_receiver.recv() {
                for entry in converted {
                    shuffle_buffer.push(entry);

                    if shuffle_buffer.len() == buffer_size {
                        shuffle(&mut shuffle_buffer);

                        if buffer_msg_receiver.try_recv().unwrap_or(false)
                            || buffer_sender.send(shuffle_buffer).is_err()
                        {
                            converted_msg_sender.send(true).unwrap();
                            break 'dataloading;
                        }

                        shuffle_buffer = Vec::with_capacity(buffer_size);
                    }
                }
            }
        });

        'dataloading: while let Ok(shuffle_buffer) = buffer_receiver.recv() {
            if f(&shuffle_buffer) {
                buffer_msg_sender.send(true).unwrap();
                break 'dataloading;
            }
        }
    }
}

fn convert_buffer<T>(threads: usize, sender: &SyncSender<Vec<ChessBoard>>, chunks: &[Vec<u8>], filter: &T) -> bool
where
    T: Fn(&TrainingDataEntry) -> bool + Sync,
{
    let chunk_size = chunks.len().div_ceil(threads);
    let mut should_break = false;

    thread::scope(|s| {
        let mut handles = Vec::new();

        for chunk_group in chunks.chunks(chunk_size) {
            let this_sender = sender.clone();
            let handle = s.spawn(move || {
                let mut buffer = Vec::new();

                for chunk in chunk_group {
                    let mut reader = ChunkReader::default();

                    while reader.has_next(chunk) {
                        let entry = reader.next(chunk);
                        if filter(&entry) {
                            buffer.push(convert_to_bulletformat(&entry));
                        }
                    }
                }

                this_sender.send(buffer).is_err()
            });

            handles.push(handle);
        }

        for handle in handles {
            if handle.join().unwrap() {
                should_break = true;
            }
        }
    });

    should_break
}

fn shuffle(data: &mut [ChessBoard]) {
    let mut rng = SimpleRand::with_seed();

    for i in (0..data.len()).rev() {
        let idx = rng.rng() as usize % (i + 1);
        data.swap(idx, i);
    }
}
