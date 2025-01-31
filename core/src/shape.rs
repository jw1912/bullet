#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    rows: usize,
    cols: usize,
    batch_size: Option<usize>,
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} x {}", self.rows, self.cols)
    }
}

impl std::ops::Mul<Shape> for Shape {
    type Output = Shape;
    fn mul(self, rhs: Shape) -> Self::Output {
        assert_eq!(self.cols, rhs.rows, "{self} * {rhs} is not possible!");
        assert!(self.batch_size == rhs.batch_size || self.batch_size.is_none() || rhs.batch_size.is_none());

        let batch_size = if self.batch_size.is_none() { rhs.batch_size } else { self.batch_size };

        Self { cols: rhs.cols, rows: self.rows, batch_size }
    }
}

impl Shape {
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(cols > 0, "Cannot have 0 columns!");
        assert!(rows > 0, "Cannot have 0 rows!");
        Self { cols, rows, batch_size: None }
    }

    pub fn new_batched(rows: usize, cols: usize, batch_size: usize) -> Self {
        assert!(cols > 0, "Cannot have 0 columns!");
        assert!(rows > 0, "Cannot have 0 rows!");
        assert!(batch_size > 0, "Cannot have batch size 0!");
        Self { cols, rows, batch_size: Some(batch_size) }
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

    pub fn batch_size(&self) -> Option<usize> {
        self.batch_size
    }

    pub fn size(&self) -> usize {
        self.cols * self.rows * self.batch_size.unwrap_or(1)
    }
}