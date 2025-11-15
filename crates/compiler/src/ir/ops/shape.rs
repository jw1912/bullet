use crate::Size;

#[derive(Clone, PartialEq, Eq)]
pub struct Shape(Vec<Size>);

impl From<Size> for Shape {
    fn from(value: Size) -> Self {
        Self(vec![value])
    }
}

impl<T: AsRef<[usize]>> From<T> for Shape {
    fn from(value: T) -> Self {
        Self(value.as_ref().iter().map(|&x| x.into()).collect())
    }
}

impl<T: Into<Shape>> std::ops::Add<T> for Shape {
    type Output = Shape;

    fn add(mut self, rhs: T) -> Shape {
        self.0.extend_from_slice(&rhs.into().0);
        self
    }
}

impl std::ops::Add<Shape> for Size {
    type Output = Shape;

    fn add(self, rhs: Shape) -> Shape {
        let mut shape = Shape(vec![self]);
        shape.0.extend_from_slice(&rhs.0);
        shape
    }
}

impl<T: AsRef<[usize]>> std::ops::Add<T> for Size {
    type Output = Shape;

    fn add(self, rhs: T) -> Shape {
        let mut shape = Shape(vec![self]);
        let v: Vec<_> = rhs.as_ref().iter().map(|&x| x.into()).collect();
        shape.0.extend_from_slice(&v);
        shape
    }
}

impl std::ops::Add<Shape> for usize {
    type Output = Shape;

    fn add(self, rhs: Shape) -> Shape {
        let mut shape = Shape(vec![self.into()]);
        shape.0.extend_from_slice(&rhs.0);
        shape
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = Size;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0[0])?;

        for x in self.0.iter().skip(1) {
            write!(f, "x{x:?}")?;
        }

        Ok(())
    }
}

impl Shape {
    pub fn size(&self) -> Size {
        self.0.iter().fold(Size::constant(1), |x, &y| x * y)
    }

    pub fn inner(&self) -> &[Size] {
        &self.0
    }
}
