use bullet_compiler::graph::TValue;

pub fn write_to_byte_buffer(value: &TValue, id: &str) -> std::io::Result<Vec<u8>> {
    use std::io::{Error, ErrorKind, Write};

    let TValue::F32(value) = value else { unimplemented!() };

    if !id.is_ascii() {
        return Err(Error::new(ErrorKind::InvalidInput, "IDs may not contain non-ASCII characters!"));
    }

    if id.contains('\n') {
        return Err(Error::new(ErrorKind::InvalidInput, "IDs may not contain newlines!"));
    }

    let mut id_bytes = id.chars().map(|ch| ch as u8).collect::<Vec<_>>();

    id_bytes.push(b'\n');

    let mut buf = Vec::new();

    buf.write_all(&id_bytes)?;
    buf.write_all(&usize::to_le_bytes(value.len()))?;

    for &val in value {
        buf.write_all(&f32::to_le_bytes(val))?;
    }

    Ok(buf)
}

pub fn read_from_byte_buffer(bytes: &[u8]) -> (Vec<f32>, String, usize) {
    const USIZE: usize = std::mem::size_of::<usize>();

    let mut offset = 0;

    let mut id = String::new();
    loop {
        let ch = bytes[offset];
        offset += 1;

        if ch == b'\n' {
            break;
        }

        id.push(char::from(ch));
    }

    let mut single_size = [0u8; USIZE];
    single_size.copy_from_slice(&bytes[offset..offset + USIZE]);
    offset += USIZE;

    let single_size = usize::from_le_bytes(single_size);

    let total_read = offset + single_size * 4;

    let mut values = vec![0.0; single_size];

    for (word, val) in bytes[offset..total_read].chunks_exact(4).zip(values.iter_mut()) {
        let mut buf = [0; 4];
        buf.copy_from_slice(word);
        *val = f32::from_le_bytes(buf);
    }

    (values, id, total_read)
}
