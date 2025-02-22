#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Shape {
    rows: usize,
    cols: usize,
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} x {}", self.rows, self.cols)
    }
}

impl std::ops::Mul<Shape> for Shape {
    type Output = Shape;
    fn mul(self, rhs: Shape) -> Self::Output {
        self.matmul(rhs).unwrap()
    }
}

impl Shape {
    pub fn matmul(self, rhs: Shape) -> Option<Self> {
        (self.cols == rhs.rows).then_some(Self { cols: rhs.cols, rows: self.rows })
    }

    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(cols > 0, "Cannot have 0 columns!");
        assert!(rows > 0, "Cannot have 0 rows!");
        Self { cols, rows }
    }

    pub fn transpose(&self) -> Self {
        Self { cols: self.rows, rows: self.cols }
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

    pub fn size(&self) -> usize {
        self.cols * self.rows
    }
}
