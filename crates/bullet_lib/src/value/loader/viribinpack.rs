use crate::game::{formats::bulletformat::ChessBoard, readers::viriformat::ViriformatReader};

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
    buffer_size_mb: usize,
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
            buffer_size_mb,
            threads,
            filter: filter.into(),
        }
    }
}

impl DataReader<ChessBoard> for ViriBinpackLoader {
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, skip_count: usize, f: F) {
        let paths = self.file_paths.iter().map(String::as_str).collect::<Vec<_>>();

        match &self.filter {
            ViriFilter::Builtin(filter) => {
                let reader = ViriformatReader::new_concat_multiple(
                    &paths,
                    self.buffer_size_mb,
                    self.threads,
                    |board, mv, score, wdl, _| filter.should_filter(mv, score.into(), board, wdl, &mut rand::rng()),
                );
                reader.read_chunks(skip_count, f);
            }
            ViriFilter::Custom(filter) => {
                let reader = ViriformatReader::new_concat_multiple(
                    &paths,
                    self.buffer_size_mb,
                    self.threads,
                    |board, mv, score, wdl, _| filter(board, mv, score, f32::from(wdl as u8) / 2.0),
                );

                reader.read_chunks(skip_count, f);
            }
        }
    }
}
