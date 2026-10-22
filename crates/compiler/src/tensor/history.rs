use std::{cell::RefCell, collections::HashMap, fmt::Write, io, path::Path, rc::Rc};

use crate::tensor::{TensorIR, Token, transform::IRTransform};

mod highlight;
mod svg;

/// Records the displayed form of a `TensorIR` after each core edit (see `transform::modify`),
/// so that the optimisation and lowering process can be replayed as an animation.
///
/// Cloning gives another handle to the same history, and a single history can hold
/// several recordings, e.g. forward and backward lowering of the same model.
#[derive(Clone, Debug, Default)]
pub struct IRHistory(Rc<RefCell<Vec<Recording>>>);

#[derive(Debug)]
pub struct Recording {
    pub name: String,
    pub frames: Vec<Frame>,
    stack: Vec<&'static str>,
    /// Last frame is a chain of this kind of edit, which can be extended
    chain: Option<&'static str>,
}

#[derive(Clone, Debug)]
pub struct Frame {
    /// Stack of transforms that were being applied, outermost first,
    /// ending with the core edit (or last of a merged chain of them).
    pub passes: Vec<&'static str>,
    /// Debug form of the core transform(s) that produced this frame, one per line.
    pub edit: String,
    /// Display form of the IR after the edit.
    pub ir: Vec<Line>,
}

/// A line of `TensorIR` display output, as plain text and highlighted tokens.
pub type Line = Vec<(Option<Token>, String)>;

impl Frame {
    pub fn ir_text(&self) -> String {
        let lines = self.ir.iter().map(|line| line.iter().map(|(_, text)| text.as_str()).collect::<String>());
        lines.collect::<Vec<_>>().join("\n")
    }
}

/// Append a token of `TensorIR` display output to `lines`, which is non-empty.
fn push_token(lines: &mut Vec<Line>, token: Option<Token>, text: &str) {
    for (i, part) in text.split('\n').enumerate() {
        if i != 0 {
            lines.push(Vec::new());
        }

        let line = lines.last_mut().unwrap();
        match line.last_mut() {
            Some((None, plain)) if token.is_none() => plain.push_str(part),
            _ if part.is_empty() => {}
            _ => line.push((token, part.to_string())),
        }
    }
}

/// Handle to a single recording within an `IRHistory`, attached to a `TensorIR`.
#[derive(Clone, Debug)]
pub(crate) struct Recorder {
    history: IRHistory,
    index: usize,
}

impl Recorder {
    fn with<T>(&self, f: impl FnOnce(&mut Recording) -> T) -> T {
        f(&mut self.history.0.borrow_mut()[self.index])
    }

    pub(crate) fn push_pass(&self, type_name: &'static str) {
        self.with(|r| r.stack.push(short_type_name(type_name)));
    }

    pub(crate) fn pop_pass(&self) {
        self.with(|r| r.stack.pop());
    }

    pub(crate) fn record(&self, ir: &TensorIR, edit: &dyn IRTransform) {
        let mut lines = vec![Vec::new()];
        let written = ir.write_ir(&mut |token, text| {
            push_token(&mut lines, token, text);
            Ok(())
        });
        let text = match written {
            Ok(()) => lines,
            Err(_) => vec![vec![(None, "<failed to display IR>".to_string())]],
        };

        let kind = short_type_name(edit.type_name());
        let mergeable = MERGEABLE.contains(&kind);

        self.with(|r| {
            if r.chain != Some(kind) {
                r.chain = None;
            }

            // nothing visible changed
            if r.frames.last().is_some_and(|last| last.ir == text) {
                return;
            }

            let passes = r.stack.clone();

            let edit_text = format!("{edit:?}");

            // merge consecutive edits of the same kind from the same pass into a single frame
            if mergeable
                && r.chain == Some(kind)
                && let Some(last) = r.frames.last_mut()
                && last.passes == passes
            {
                write!(last.edit, "\n{edit_text}").unwrap();
                last.ir = text;
                return;
            }

            r.chain = mergeable.then_some(kind);
            r.frames.push(Frame { passes, edit: edit_text, ir: text });
        });
    }
}

/// Edits for which a chain from the same pass is shown as a single frame
const MERGEABLE: [&str; 3] = ["AddOperation", "RemoveOperation", "ReorderInputs"];

fn short_type_name(name: &'static str) -> &'static str {
    let name = name.split('<').next().unwrap_or(name);
    name.rsplit("::").next().unwrap_or(name)
}

impl IRHistory {
    /// Start a new recording on `ir`, replacing any recording it was already attached to.
    pub fn attach(&self, ir: &mut TensorIR, name: impl Into<String>) {
        let mut recordings = self.0.borrow_mut();
        let index = recordings.len();
        recordings.push(Recording { name: name.into(), frames: Vec::new(), stack: Vec::new(), chain: None });
        drop(recordings);

        let recorder = Recorder { history: self.clone(), index };
        recorder.record(ir, &Attach);
        ir.recorder = Some(recorder);
    }

    pub fn num_recordings(&self) -> usize {
        self.0.borrow().len()
    }

    pub fn with_recordings<T>(&self, f: impl FnOnce(&[Recording]) -> T) -> T {
        f(&self.0.borrow())
    }

    /// Write a self-contained HTML page that animates every recording.
    pub fn write_html(&self, path: impl AsRef<Path>) -> io::Result<()> {
        std::fs::write(path, self.to_html())
    }

    /// Write a self-contained SVG that animates the recording at `index`,
    /// which unlike the HTML page can be embedded in markdown.
    pub fn write_svg(&self, path: impl AsRef<Path>, index: usize) -> io::Result<()> {
        std::fs::write(path, self.to_svg(index))
    }

    pub fn to_svg(&self, index: usize) -> String {
        svg::to_svg(&self.0.borrow()[index])
    }

    pub fn to_html(&self) -> String {
        // IR lines repeat heavily between frames, so intern them to keep the page small
        let mut lines = Vec::new();
        let mut line_ids = HashMap::new();
        let mut intern = |line: &Line| {
            *line_ids.entry(line.clone()).or_insert_with(|| {
                lines.push(json_string(&highlight::markup(line, "span")));
                lines.len() - 1
            })
        };

        let mut passes = Vec::new();
        let mut pass_ids = HashMap::new();

        let mut data = String::from("{\"recordings\":[");
        for (i, recording) in self.0.borrow().iter().enumerate() {
            if i != 0 {
                data.push(',');
            }

            write!(data, "{{\"name\":{},\"frames\":[", json_string(&recording.name)).unwrap();
            for (j, frame) in recording.frames.iter().enumerate() {
                if j != 0 {
                    data.push(',');
                }

                let pass = frame.passes.join(" > ");
                let pass = *pass_ids.entry(pass.clone()).or_insert_with(|| {
                    passes.push(json_string(&pass));
                    passes.len() - 1
                });

                let ids = frame.ir.iter().map(|l| intern(l).to_string()).collect::<Vec<_>>().join(",");
                write!(data, "[{pass},{},[{ids}]]", json_string(&frame.edit)).unwrap();
            }
            data.push_str("]}");
        }

        write!(data, "],\"passes\":[{}],\"lines\":[{}]}}", passes.join(","), lines.join(",")).unwrap();

        // stop the data from closing the script tag early
        let data = data.replace("</", "<\\/");

        HTML_TEMPLATE
            .replace("/*HIGHLIGHT_LIGHT*/", highlight::LIGHT)
            .replace("/*HIGHLIGHT_DARK*/", highlight::DARK)
            .replace("/*HIGHLIGHT_CSS*/", &highlight::css("color"))
            .replace("/*DATA*/null", &data)
    }
}

#[derive(Debug)]
struct Attach;

impl IRTransform for Attach {
    fn apply(&self, _: &mut TensorIR) -> Result<(), super::IRTrace> {
        Ok(())
    }
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => write!(out, "\\u{:04x}", c as u32).unwrap(),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

const HTML_TEMPLATE: &str = include_str!("history/history.html");
