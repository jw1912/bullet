use std::fmt;

#[derive(Default)]
pub struct Program {}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "program start")?;
        write!(f, "program end")
    }
}
