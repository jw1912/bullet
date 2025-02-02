use std::num::NonZeroUsize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    rows: usize,
    cols: usize,
    batch_size: Option<NonZeroUsize>,
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(size) = self.batch_size() {
            write!(f, "[{} x {}; {size}]", self.rows, self.cols)
        } else {
            write!(f, "{} x {}", self.rows, self.cols)
        }
    }
}

impl std::ops::Mul<Shape> for Shape {
    type Output = Shape;
    fn mul(self, rhs: Shape) -> Self::Output {
        assert_eq!(self.cols, rhs.rows, "{self} * {rhs} is not possible!");
        Self { cols: rhs.cols, rows: self.rows, batch_size: Self::get_batch_size(&self, &rhs) }
    }
}

impl Shape {
    pub fn get_batch_size(a: &Self, b: &Self) -> Option<NonZeroUsize> {
        match (a.batch_size, b.batch_size) {
            (None, None) => None,
            (None, Some(x)) => Some(x),
            (Some(x), None) => Some(x),
            (Some(x), Some(y)) => {
                assert_eq!(x, y, "Invalid combination of batch sizes: {x} != {y}");
                Some(x)
            }
        }
    }

    pub fn from_raw(rows: usize, cols: usize, batch_size: Option<usize>) -> Self {
        assert!(cols > 0, "Cannot have 0 columns!");
        assert!(rows > 0, "Cannot have 0 rows!");
        let batch_size = batch_size.map(|x| NonZeroUsize::new(x).unwrap());
        Self { cols, rows, batch_size }
    }

    pub fn new(rows: usize, cols: usize) -> Self {
        Self::from_raw(rows, cols, None)
    }

    pub fn new_batched(rows: usize, cols: usize, batch_size: usize) -> Self {
        assert!(batch_size > 0, "Cannot have batch size 0!");
        Self::from_raw(rows, cols, Some(batch_size))
    }

    pub fn transpose(&self) -> Self {
        Self { cols: self.rows, rows: self.cols, batch_size: self.batch_size }
    }

    pub fn maybe_transpose(&self, trans: bool) -> Self {
        if trans {
            self.transpose()
        } else {
            *self
        }
    }

    pub fn without_batch_size(&self) -> Self {
        Self { rows: self.rows, cols: self.cols, batch_size: None }
    }

    pub fn reshape(&mut self, rows: usize, cols: usize) {
        assert_eq!(self.rows * self.cols, cols * rows, "Invalid reshape!");
        self.cols = cols;
        self.rows = rows;
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn size(&self) -> usize {
        self.cols * self.rows * self.batch_size().unwrap_or(1)
    }

    pub fn batch_size(&self) -> Option<usize> {
        self.batch_size.map(NonZeroUsize::get)
    }
}
