use std::{
    fs::File,
    io::BufReader,
    sync::mpsc::{self, SyncSender},
};

use bullet_trainer::reader::DataReader;
use rayon::{ThreadPool, ThreadPoolBuilder};
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

use super::shuffle::{SHUFFLE_THREADS, ShuffledDataLoader, convert_chunks};

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
pub struct SfBinpackLoader<T: Fn(&TrainingDataEntry) -> bool>(ShuffledDataLoader<UnshuffledSfBinpackLoader<T>>);

impl<T: Fn(&TrainingDataEntry) -> bool> SfBinpackLoader<T> {
    pub fn new(path: &str, buffer_size_mb: usize, threads: usize, filter: T) -> Self {
        Self::new_concat_multiple(&[path], buffer_size_mb, threads, filter)
    }

    pub fn new_concat_multiple(paths: &[&str], buffer_size_mb: usize, threads: usize, filter: T) -> Self {
        let reader = UnshuffledSfBinpackLoader::new_concat_multiple(paths, threads, filter);
        Self(ShuffledDataLoader::new(reader, Some(buffer_size_mb), threads.min(SHUFFLE_THREADS)))
    }
}

impl<T> DataReader<ChessBoard> for SfBinpackLoader<T>
where
    T: Fn(&TrainingDataEntry) -> bool + Clone + Send + Sync + 'static,
{
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, skip_count: usize, f: F) {
        self.0.read_chunks(skip_count, f);
    }
}

#[derive(Clone)]
pub struct UnshuffledSfBinpackLoader<T: Fn(&TrainingDataEntry) -> bool> {
    file_paths: Vec<String>,
    threads: usize,
    filter: T,
}

impl<T: Fn(&TrainingDataEntry) -> bool> UnshuffledSfBinpackLoader<T> {
    pub fn new(path: &str, threads: usize, filter: T) -> Self {
        Self::new_concat_multiple(&[path], threads, filter)
    }

    pub fn new_concat_multiple(paths: &[&str], threads: usize, filter: T) -> Self {
        Self { file_paths: paths.iter().map(|x| x.to_string()).collect(), threads: threads.max(1), filter }
    }
}

impl<T> DataReader<ChessBoard> for UnshuffledSfBinpackLoader<T>
where
    T: Fn(&TrainingDataEntry) -> bool + Clone + Send + Sync + 'static,
{
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, _: usize, mut f: F) {
        let file_paths = self.file_paths.clone();
        let threads = self.threads;
        let filter = self.filter.clone();
        let reader_buffer_size = threads;
        let pool = ThreadPoolBuilder::new().num_threads(threads).build().unwrap();

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

        let (converted_sender, converted_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(0);
        let (converted_msg_sender, converted_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let filter = &filter;
            'dataloading: while let Ok(chunks) = reader_receiver.recv() {
                if converted_msg_receiver.try_recv().unwrap_or(false) {
                    let _ = reader_msg_sender.send(true);
                    break 'dataloading;
                }

                if convert_buffer(&pool, &converted_sender, &chunks, filter) {
                    let _ = reader_msg_sender.send(true);
                    break 'dataloading;
                }
            }
        });

        'dataloading: while let Ok(converted) = converted_receiver.recv() {
            if f(&converted) {
                let _ = converted_msg_sender.send(true);
                break 'dataloading;
            }
        }
    }
}

fn convert_buffer<T>(pool: &ThreadPool, sender: &SyncSender<Vec<ChessBoard>>, chunks: &[Vec<u8>], filter: &T) -> bool
where
    T: Fn(&TrainingDataEntry) -> bool + Sync,
{
    convert_chunks(pool, sender, chunks, 1, |chunk_group| {
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

        buffer
    })
}
