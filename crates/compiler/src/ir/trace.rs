use std::{fmt, rc::Rc};

use crate::{
    ir::{
        graph::{IrError, IrGraph},
        transform::IrTransform,
    },
    utils::Ansi,
};

#[derive(Clone)]
pub enum IRTrace {
    Root(IrError),
    Frame(Box<IrGraph>, Rc<dyn IrTransform>, Rc<Self>),
}

impl IRTrace {
    pub fn frame(&self, f: &mut impl fmt::Write) -> fmt::Result {
        match self {
            Self::Root(err) => write!(f, "{err:?}"),
            Self::Frame(graph, transform, _) => {
                let orange = Ansi::rgb(212, 114, 34);
                let clear = Ansi::Clear;

                writeln!(f, "{orange}Error applying{clear}")?;
                writeln!(f, "{transform:?}")?;
                writeln!(f, "{orange}on graph{clear}")?;
                write!(f, "{}", graph.as_highlighted())
            }
        }
    }

    pub fn full_string(&self, f: &mut impl fmt::Write, frame: usize) -> fmt::Result {
        writeln!(f, "{}Depth {frame}:{}", Ansi::rgb(255, 0, 0), Ansi::Clear)?;

        self.frame(f)?;

        if let Self::Frame(_, _, inner) = self {
            writeln!(f)?;
            inner.full_string(f, frame + 1)?;
        }

        Ok(())
    }
}

impl From<IrError> for IRTrace {
    fn from(value: IrError) -> Self {
        Self::Root(value)
    }
}

impl<T: Into<String>> From<T> for IRTrace {
    fn from(value: T) -> Self {
        Self::Root(value.into().into())
    }
}

impl fmt::Debug for IRTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.full_string(f, 0)
    }
}

#[derive(Clone, Debug)]
enum IRHistoryFrame {
    ScopeStart,
    ScopeEnd,
    Entry(Rc<dyn IrTransform>),
}

#[derive(Clone, Debug, Default)]
pub struct IRHistory(Vec<IRHistoryFrame>);

impl fmt::Display for IRHistory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut scope = 0;

        write!(f, "IR Transform History")?;

        for entry in &self.0 {
            match entry {
                IRHistoryFrame::ScopeStart => scope += 1,
                IRHistoryFrame::ScopeEnd => scope -= 1,
                IRHistoryFrame::Entry(entry) => {
                    writeln!(f)?;
                    write!(f, "{}|-- {entry:?}", " ".repeat(4 * scope))?;
                }
            }
        }

        Ok(())
    }
}

impl IRHistory {
    pub fn start_scope(&mut self) {
        self.0.push(IRHistoryFrame::ScopeStart);
    }

    pub fn end_scope(&mut self) {
        self.0.push(IRHistoryFrame::ScopeEnd);
    }

    pub fn push(&mut self, transform: Rc<dyn IrTransform>) {
        self.0.push(IRHistoryFrame::Entry(transform));
    }
}
