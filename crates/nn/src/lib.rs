pub mod games;
pub mod train;

pub use bullet_compiler as compiler;
pub use bullet_gpu as gpu;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    rows: usize,
    cols: usize,
}

impl From<(usize, usize)> for Shape {
    fn from(value: (usize, usize)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl Shape {
    pub fn new(rows: usize, cols: usize) -> Shape {
        Self { rows, cols }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn size(&self) -> usize {
        self.rows * self.cols
    }
}
