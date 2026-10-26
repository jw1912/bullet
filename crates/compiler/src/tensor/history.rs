use std::{collections::HashMap, fmt::Write, io, path::Path};

use crate::tensor::{TensorIR, Token, transform::IRTransform};

/// Records the displayed form of a `TensorIR` after each core edit (see `transform::modify`),
/// so that the optimisation and lowering process can be replayed as an animation.
///
/// Every `TensorIR` has one, which only records between `TensorIR::start_recording`
/// and `TensorIR::stop_recording`.
#[derive(Clone, Debug, Default)]
pub struct IRHistory {
    pub frames: Vec<Frame>,
    /// Type names of the transforms being applied, tracked even when not recording
    /// so that it is correct whenever recording starts
    stack: Vec<&'static str>,
    /// Last frame is a chain of this kind of edit, which can be extended
    chain: Option<&'static str>,
    recording: bool,
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
    /// Label given to `TensorIR::start_recording`, set on the first frame each time
    /// recording is started, as the IR may have changed arbitrarily since the previous frame.
    pub label: Option<String>,
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

impl IRHistory {
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    pub(crate) fn push_pass(&mut self, type_name: &'static str) {
        self.stack.push(type_name);
    }

    pub(crate) fn pop_pass(&mut self) {
        self.stack.pop();
    }

    fn passes(&self) -> Vec<&'static str> {
        self.stack.iter().map(|&name| short_type_name(name)).collect()
    }

    /// Start recording, with `ir` the current display form of the IR.
    pub(crate) fn start(&mut self, ir: Vec<Line>, label: String) {
        self.recording = true;
        self.chain = None;
        let mut passes = self.passes();
        passes.push("StartRecording");
        self.frames.push(Frame { passes, edit: String::new(), ir, label: Some(label) });
    }

    pub(crate) fn stop(&mut self) {
        self.recording = false;
    }

    /// Record `ir`, the display form of the IR after `edit`, which should
    /// only be computed when recording.
    pub(crate) fn record(&mut self, ir: Vec<Line>, edit: &dyn IRTransform) {
        let kind = short_type_name(edit.type_name());

        // a different kind of edit ends the chain
        self.chain = self.chain.filter(|&chain| chain == kind);

        // nothing visible changed
        if self.frames.last().is_some_and(|last| last.ir == ir) {
            return;
        }

        let passes = self.passes();
        let edit_text = format!("{edit:?}");

        // merge consecutive edits of the same kind from the same pass into a single frame
        if self.chain.is_some()
            && let Some(last) = self.frames.last_mut()
            && last.passes == passes
        {
            write!(last.edit, "\n{edit_text}").unwrap();
            last.ir = ir;
            return;
        }

        self.chain = MERGEABLE.contains(&kind).then_some(kind);
        self.frames.push(Frame { passes, edit: edit_text, ir, label: None });
    }

    /// Write a self-contained HTML page that animates this history.
    pub fn write_html(&self, path: impl AsRef<Path>) -> io::Result<()> {
        std::fs::write(path, self.to_html())
    }

    pub fn to_html(&self) -> String {
        // IR lines repeat heavily between frames, so intern them to keep the page small
        let mut lines = Vec::new();
        let mut line_ids = HashMap::new();
        let mut intern = |line: &Line| {
            *line_ids.entry(line.clone()).or_insert_with(|| {
                lines.push(json_string(&markup(line)));
                lines.len() - 1
            })
        };

        let mut passes = Vec::new();
        let mut pass_ids = HashMap::new();

        let mut data = String::from("{\"frames\":[");
        for (i, frame) in self.frames.iter().enumerate() {
            if i != 0 {
                data.push(',');
            }

            let pass = frame.passes.join(" > ");
            let pass = *pass_ids.entry(pass.clone()).or_insert_with(|| {
                passes.push(json_string(&pass));
                passes.len() - 1
            });

            let ids = frame.ir.iter().map(|l| intern(l).to_string()).collect::<Vec<_>>().join(",");
            write!(data, "[{pass},{},[{ids}]", json_string(&frame.edit)).unwrap();
            if let Some(label) = &frame.label {
                write!(data, ",{}", json_string(label)).unwrap();
            }
            data.push(']');
        }

        write!(data, "],\"passes\":[{}],\"lines\":[{}]}}", passes.join(","), lines.join(",")).unwrap();

        // stop the data from closing the script tag early
        let data = data.replace("</", "<\\/");

        HTML_TEMPLATE.replace("/*DATA*/null", &data)
    }
}

/// Display form of `ir`, split into highlighted lines.
pub(crate) fn display(ir: &TensorIR) -> Vec<Line> {
    let mut lines = vec![Vec::new()];
    let written = ir.write_ir(&mut |token, text| {
        push_token(&mut lines, token, text);
        Ok(())
    });

    match written {
        Ok(()) => lines,
        Err(_) => vec![vec![(None, "<failed to display IR>".to_string())]],
    }
}

/// Edits for which a chain from the same pass is shown as a single frame
const MERGEABLE: [&str; 3] = ["AddOperation", "RemoveOperation", "ReorderInputs"];

fn short_type_name(name: &'static str) -> &'static str {
    let name = name.split('<').next().unwrap_or(name);
    name.rsplit("::").next().unwrap_or(name)
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

const HTML_TEMPLATE: &str = include_str!("history.html");

/// Markup for `line`, with each token wrapped in a `span` with its class (styled in
/// `history.html`), and all text escaped.
fn markup(line: &Line) -> String {
    let mut out = String::new();
    for (token, text) in line {
        match token {
            Some(token) => write!(out, r#"<span class="{}">{}</span>"#, token_class(*token), escape(text)).unwrap(),
            None => out.push_str(&escape(text)),
        }
    }
    out
}

fn token_class(token: Token) -> &'static str {
    match token {
        Token::Keyword => "k",
        Token::Op => "o",
        Token::Type => "t",
        Token::Number => "n",
        Token::Punct => "p",
        Token::Bracket(depth) => ["b0", "b1", "b2"][depth % 3],
    }
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}
