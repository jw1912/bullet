use std::fmt::Write;

use super::Line;
use crate::tensor::Token;

/// Styling for highlighted lines of `TensorIR` display output, shared by the HTML and SVG renderers.
///
/// Token classes are `k` (keyword), `o` (operation), `t` (type), `n` (number/shape), `p` (operator)
/// and `b0`-`b2` (brackets, coloured by nesting depth). Values and other text are left plain.
pub const LIGHT: &str = "--hl-k:#a626a4; --hl-o:#4078f2; --hl-t:#c18401; --hl-n:#986801; --hl-p:#0184bc; \
                         --hl-b0:#0431fa; --hl-b1:#319331; --hl-b2:#7b3814;";
pub const DARK: &str = "--hl-k:#e143db; --hl-o:#4ea8ef; --hl-t:#eac47c; --hl-n:#dc9966; --hl-p:#00bbc3; \
                        --hl-b0:#ffd700; --hl-b1:#da70d6; --hl-b2:#179fff;";

/// CSS rules for the token classes, colouring text with `property` (`color` for HTML, `fill` for SVG).
pub fn css(property: &str) -> String {
    let mut css = String::new();
    for class in ["k", "o", "t", "n", "p", "b0", "b1", "b2"] {
        css.push_str(&format!(".{class} {{ {property}: var(--hl-{class}); }} "));
    }
    css.push_str(".k { font-style: italic; }");
    css
}

/// Markup for `line`, with each token wrapped in a `tag` element with its class, and all text escaped.
pub fn markup(line: &Line, tag: &str) -> String {
    let mut out = String::new();
    for (token, text) in line {
        match token {
            Some(token) => write!(out, r#"<{tag} class="{}">{}</{tag}>"#, class(*token), escape(text)).unwrap(),
            None => out.push_str(&escape(text)),
        }
    }
    out
}

fn class(token: Token) -> &'static str {
    match token {
        Token::Keyword => "k",
        Token::Op => "o",
        Token::Type => "t",
        Token::Number => "n",
        Token::Punct => "p",
        Token::Bracket(depth) => ["b0", "b1", "b2"][depth % 3],
    }
}

pub fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}
