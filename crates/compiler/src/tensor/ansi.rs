use std::{fmt, ops::Add};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ansi {
    Clear = 0,
    Bold,
    Faint,
    Italic,
    Underline,
    SlowBlink,
    Black = 30,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
    BrightBlack = 90,
    BrightRed,
    BrightGreen,
    BrightYellow,
    BrightBlue,
    BrightMagenta,
    BrightCyan,
    BrightWhite,
}

impl Ansi {
    pub fn rgb(red: u8, green: u8, blue: u8) -> AnsiComb {
        AnsiComb(vec![38, 2, red, green, blue])
    }
}

impl fmt::Display for Ansi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\x1b[{}m", *self as u8)
    }
}

impl Add<Self> for Ansi {
    type Output = AnsiComb;

    fn add(self, rhs: Self) -> Self::Output {
        AnsiComb(vec![self as u8, rhs as u8])
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct AnsiComb(Vec<u8>);

impl fmt::Display for AnsiComb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\x1b[")?;

        for (i, &x) in self.0.iter().enumerate() {
            if i != 0 {
                write!(f, ";")?;
            }

            write!(f, "{x}")?;
        }

        write!(f, "m")
    }
}

impl Add<Self> for AnsiComb {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        AnsiComb([self.0, rhs.0].concat())
    }
}

impl Add<Ansi> for AnsiComb {
    type Output = Self;

    fn add(self, rhs: Ansi) -> Self {
        let mut res = self.clone();
        res.0.push(rhs as u8);
        res
    }
}

impl Add<AnsiComb> for Ansi {
    type Output = AnsiComb;

    fn add(self, rhs: AnsiComb) -> AnsiComb {
        rhs + self
    }
}
