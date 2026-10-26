use std::fmt;

/// Syntax class of a token in the display form of a `TensorIR`, see `TensorIR::write_ir`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Token {
    Keyword,
    Op,
    Type,
    /// Numbers and shapes, e.g. `128x768`
    Number,
    Punct,
    /// Bracket at the given nesting depth
    Bracket(usize),
}

/// Tokenise an op name, e.g. `sparse.matmul<16384, 128x768.C, 0>`, which is free-form
/// text, starting at bracket depth `depth`.
pub(crate) fn opname(text: &str, depth: usize, emit: Emit<'_>) -> fmt::Result {
    let (name, params) = text.split_at(text.find('<').unwrap_or(text.len()));
    namespaced(name, emit)?;
    lex(params, depth, emit)
}

pub(crate) type Emit<'a> = &'a mut dyn FnMut(Option<Token>, &str) -> fmt::Result;

fn namespaced(name: &str, emit: Emit<'_>) -> fmt::Result {
    // namespace separators are operators, as in method calls
    for (i, part) in name.split('.').enumerate() {
        if i != 0 {
            emit(Some(Token::Punct), ".")?;
        }
        emit(Some(Token::Op), part)?;
    }

    Ok(())
}

fn lex(text: &str, mut depth: usize, emit: Emit<'_>) -> fmt::Result {
    let bytes = text.as_bytes();
    let mut plain = 0;
    let mut i = 0;

    let flush = |emit: Emit<'_>, start: usize, end: usize| {
        if start < end { emit(None, &text[start..end]) } else { Ok(()) }
    };

    while i < bytes.len() {
        let start = i;
        let digits = |mut i: usize| {
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            i
        };

        let token = match bytes[i] {
            b'0'..=b'9' => {
                // integers, decimals, exponents and shapes such as 128x768
                i = digits(i);
                if bytes.get(i) == Some(&b'.') && bytes.get(i + 1).is_some_and(u8::is_ascii_digit) {
                    i = digits(i + 1);
                }
                if matches!(bytes.get(i), Some(b'e' | b'E')) {
                    let j = i + 1 + usize::from(matches!(bytes.get(i + 1), Some(b'-' | b'+')));
                    if bytes.get(j).is_some_and(u8::is_ascii_digit) {
                        i = digits(j);
                    }
                }
                while bytes.get(i) == Some(&b'x') && bytes.get(i + 1).is_some_and(u8::is_ascii_digit) {
                    i = digits(i + 1);
                }
                Token::Number
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                // op names can contain hyphens, e.g. faux-quantise
                let word = |c: &u8| c.is_ascii_alphanumeric() || *c == b'_' || *c == b'.';
                while i < bytes.len()
                    && (word(&bytes[i])
                        || (bytes[i] == b'-' && bytes.get(i + 1).is_some_and(|c| c.is_ascii_alphabetic())))
                {
                    i += 1;
                }
                match &text[start..i] {
                    "f32" | "i32" | "F32" | "I32" => Token::Type,
                    // nested op names, e.g. in parameters
                    name if matches!(bytes.get(i), Some(b'<' | b'(')) => {
                        flush(emit, plain, start)?;
                        namespaced(name, emit)?;
                        plain = i;
                        continue;
                    }
                    _ => continue,
                }
            }
            b'(' | b'[' | b'{' => {
                i += 1;
                depth += 1;
                Token::Bracket(depth - 1)
            }
            b')' | b']' | b'}' => {
                i += 1;
                depth = depth.saturating_sub(1);
                Token::Bracket(depth)
            }
            b'=' | b':' | b'.' | b'-' => {
                i += 1;
                Token::Punct
            }
            _ => {
                i += text[i..].chars().next().map_or(1, char::len_utf8);
                continue;
            }
        };

        flush(emit, plain, start)?;
        emit(Some(token), &text[start..i])?;
        plain = i;
    }

    flush(emit, plain, text.len())
}
