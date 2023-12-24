#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    cols: usize,
    rows: usize,
}

impl std::ops::Mul<Shape> for Shape {
    type Output = Shape;
    fn mul(self, rhs: Shape) -> Self::Output {
        assert_eq!(self.cols, rhs.rows);

        Self {
            cols: rhs.cols,
            rows: self.rows,
        }
    }
}

impl Shape {
    pub fn new(cols: usize, rows: usize) -> Self {
        assert!(cols > 0, "Cannot have 0 columns!");
        assert!(rows > 0, "Cannot have 0 rows!");
        Self { cols, rows }
    }

    pub fn reshape(&mut self, cols: usize, rows: usize) {
        assert_eq!(self.size(), cols * rows, "Invalid reshape!");
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
