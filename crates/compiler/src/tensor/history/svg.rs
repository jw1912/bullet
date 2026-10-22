use std::{collections::HashMap, fmt::Write};

use super::{
    Line, Recording,
    highlight::{self, escape},
};

const STEP_SECONDS: f64 = 0.2;
const FONT_SIZE: usize = 12;
const LINE_HEIGHT: usize = 16;
const CHAR_WIDTH: f64 = 7.25;
const PAD: usize = 12;
const EDIT_ROWS: usize = 3;
const COLS: usize = 96;
const HEADER: usize = PAD + LINE_HEIGHT * (2 + EDIT_ROWS) + PAD;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Same(usize),
    Del,
    Add,
}

/// Line diff between `a` and `b`, as a merged list of lines, where unchanged
/// lines carry their index in `a`.
fn diff<'a>(a: &[&'a Line], b: &[&'a Line]) -> Vec<(&'a Line, Kind)> {
    let mut pre = 0;
    while pre < a.len() && pre < b.len() && a[pre] == b[pre] {
        pre += 1;
    }

    let mut suf = 0;
    while suf < a.len() - pre && suf < b.len() - pre && a[a.len() - 1 - suf] == b[b.len() - 1 - suf] {
        suf += 1;
    }

    let mut out = a[..pre].iter().enumerate().map(|(i, &l)| (l, Kind::Same(i))).collect::<Vec<_>>();
    let am = &a[pre..a.len() - suf];
    let bm = &b[pre..b.len() - suf];
    let (n, m) = (am.len(), bm.len());

    if n * m > 4_000_000 {
        // too big for LCS, treat the whole middle as replaced
        out.extend(am.iter().map(|&l| (l, Kind::Del)));
        out.extend(bm.iter().map(|&l| (l, Kind::Add)));
    } else {
        // lcs[x * (m + 1) + y] = LCS length of am[x..] and bm[y..]
        let mut lcs = vec![0u32; (n + 1) * (m + 1)];
        for x in (0..n).rev() {
            for y in (0..m).rev() {
                lcs[x * (m + 1) + y] = if am[x] == bm[y] {
                    lcs[(x + 1) * (m + 1) + y + 1] + 1
                } else {
                    lcs[(x + 1) * (m + 1) + y].max(lcs[x * (m + 1) + y + 1])
                };
            }
        }

        let (mut x, mut y) = (0, 0);
        while x < n || y < m {
            if x < n && y < m && am[x] == bm[y] {
                out.push((am[x], Kind::Same(pre + x)));
                x += 1;
                y += 1;
            } else if y >= m || (x < n && lcs[(x + 1) * (m + 1) + y] >= lcs[x * (m + 1) + y + 1]) {
                out.push((am[x], Kind::Del));
                x += 1;
            } else {
                out.push((bm[y], Kind::Add));
                y += 1;
            }
        }
    }

    out.extend(a[a.len() - suf..].iter().enumerate().map(|(i, &l)| (l, Kind::Same(a.len() - suf + i))));
    out
}

fn clip(s: &str) -> String {
    if s.chars().count() <= COLS {
        s.to_string()
    } else {
        s.chars().take(COLS - 1).chain(['…']).collect()
    }
}

/// Phase 0 = previous IR with deletions marked, 1 = deletions and additions marked,
/// 2 = new IR with additions marked, 3 = new IR without highlighting
fn shown(kind: Kind, phase: u8) -> bool {
    !((kind == Kind::Add && phase == 0) || (kind == Kind::Del && phase >= 2))
}

/// Phases a frame is shown in, given whether it has deletions and additions.
/// Without additions, phase 2 would be identical to phase 3, so it is skipped.
fn phases(dels: bool, adds: bool) -> impl Iterator<Item = u8> {
    [(0, dels), (1, dels && adds), (2, adds), (3, true)].into_iter().filter_map(|(p, show)| show.then_some(p))
}

/// A self-contained SVG that loops through `recording` using CSS animations only,
/// so that it plays where scripts are not allowed, e.g. markdown on GitHub.
pub fn to_svg(recording: &Recording) -> String {
    let frames = &recording.frames;
    let irs = frames.iter().map(|f| f.ir.iter().collect::<Vec<_>>()).collect::<Vec<_>>();
    let diffs = (0..frames.len()).map(|i| diff(if i > 0 { &irs[i - 1] } else { &irs[i] }, &irs[i])).collect::<Vec<_>>();

    // each frame is shown in up to four steps, as in the HTML page
    let mut steps = Vec::new();
    for (i, lines) in diffs.iter().enumerate() {
        let dels = lines.iter().any(|(_, k)| *k == Kind::Del);
        let adds = lines.iter().any(|(_, k)| *k == Kind::Add);
        steps.extend(phases(dels, adds).map(|phase| (i, phase)));
    }

    let max_rows = steps.iter().map(|&(i, p)| diffs[i].iter().filter(|(_, k)| shown(*k, p)).count()).max().unwrap_or(0);
    let width = (2.0 * PAD as f64 + CHAR_WIDTH * COLS as f64).round() as usize;
    let height = HEADER + PAD + LINE_HEIGHT * max_rows + PAD;
    let n = steps.len().max(1);

    let mut svg = String::new();
    write!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
<style>
svg {{ {} --bg:#fafafa; --fg:#1c1c1c; --muted:#6b6b6b; --panel:#fff; --border:#ddd; --add-bg:#d9f5dd; --del-bg:#fbdcdc; }}
@media (prefers-color-scheme: dark) {{ svg {{ {} --bg:#282b34; --fg:#abb1bf; --muted:#7f848e; --panel:#21252b; --border:#3a3f4b; --add-bg:#2b4535; --del-bg:#4d2a33; }} }}
text {{ font: {FONT_SIZE}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre; }}
g > text, use {{ fill: var(--fg); }}
.m {{ fill: var(--muted); }} .h {{ font-weight: 600; font-family: system-ui, sans-serif; font-size: 13px; }}
{}
.ab {{ fill: var(--add-bg); }} .db {{ fill: var(--del-bg); }}
.s {{ visibility: hidden; animation: show {:.2}s step-end infinite; }}
@keyframes show {{ 0% {{ visibility: visible; }} {:.5}% {{ visibility: hidden; }} }}
</style>
<defs>
"#,
        highlight::LIGHT,
        highlight::DARK,
        highlight::css("fill"),
        n as f64 * STEP_SECONDS,
        100.0 / n as f64,
    )
    .unwrap();

    // every distinct line, and every frame's listing, is defined once and referenced by id
    let mut line_ids = HashMap::new();
    for line in irs.iter().flatten() {
        let id = line_ids.len();
        line_ids.entry(*line).or_insert_with(|| {
            let text = if line.is_empty() { " ".to_string() } else { highlight::markup(line, "tspan") };
            writeln!(svg, r#"<text id="l{id}" x="{PAD}" xml:space="preserve">{text}</text>"#).unwrap();
            id
        });
    }

    for (i, ir) in irs.iter().enumerate() {
        write!(svg, r#"<g id="f{i}">"#).unwrap();
        for (row, line) in ir.iter().enumerate() {
            write!(svg, r##"<use href="#l{}" y="{}"/>"##, line_ids[line], LINE_HEIGHT * row + LINE_HEIGHT - 4).unwrap();
        }
        svg.push_str("</g>\n");
    }

    svg.push_str("</defs>\n");
    writeln!(svg, r#"<rect width="{width}" height="{height}" fill="var(--bg)"/>"#).unwrap();
    writeln!(
        svg,
        r#"<rect width="{width}" height="{HEADER}" fill="var(--panel)"/><rect y="{}" width="{width}" height="1" fill="var(--border)"/>"#,
        HEADER - 1
    )
    .unwrap();

    for (s, &(i, phase)) in steps.iter().enumerate() {
        let frame = &frames[i];
        let lines = &diffs[i];
        write!(svg, r#"<g class="s" style="animation-delay:{:.2}s">"#, s as f64 * STEP_SECONDS).unwrap();

        let added = lines.iter().filter(|(_, k)| *k == Kind::Add).count();
        let removed = lines.iter().filter(|(_, k)| *k == Kind::Del).count();
        let edits = frame.edit.lines().collect::<Vec<_>>();
        let mut delta = if i > 0 { format!("+{added} −{removed} lines") } else { String::new() };
        if edits.len() > 1 {
            write!(delta, " · {} edits", edits.len()).unwrap();
        }

        let pass = frame.passes.join(" > ");
        let pass = if pass.is_empty() { "(top level)".to_string() } else { pass };
        let y = PAD + LINE_HEIGHT - 3;
        write!(svg, r#"<text class="h" x="{PAD}" y="{y}">{}</text>"#, escape(&pass)).unwrap();
        write!(
            svg,
            // on the second row, as pass labels can be long
            r#"<text class="m" x="{}" y="{}" text-anchor="end">step {} / {} · frame {} / {}</text>"#,
            width - PAD,
            y + LINE_HEIGHT,
            s + 1,
            steps.len(),
            i + 1,
            frames.len()
        )
        .unwrap();
        write!(svg, r#"<text class="m" x="{PAD}" y="{}">{delta}</text>"#, y + LINE_HEIGHT).unwrap();

        for row in 0..EDIT_ROWS.min(edits.len()) {
            let text = if row == EDIT_ROWS - 1 && edits.len() > EDIT_ROWS {
                format!("… and {} more", edits.len() - row)
            } else {
                clip(edits[row])
            };
            let ty = y + LINE_HEIGHT * (2 + row);
            write!(svg, r#"<text class="m" x="{PAD}" y="{ty}" xml:space="preserve">{}</text>"#, escape(&text)).unwrap();
        }

        let top = HEADER + PAD;
        if phase == 3 || i == 0 {
            write!(svg, r##"<use href="#f{i}" y="{top}"/>"##).unwrap();
        } else {
            // runs of unchanged lines are windows into the previous frame's listing
            let mut run: Option<(usize, usize, usize)> = None;
            let flush = |svg: &mut String, run: Option<(usize, usize, usize)>| {
                if let Some((row, start, len)) = run {
                    write!(
                        svg,
                        r##"<svg y="{}" width="{width}" height="{}"><use href="#f{}" y="-{}"/></svg>"##,
                        top + LINE_HEIGHT * row,
                        LINE_HEIGHT * len,
                        i - 1,
                        LINE_HEIGHT * start
                    )
                    .unwrap();
                }
            };

            for (row, &(line, kind)) in lines.iter().filter(|(_, k)| shown(*k, phase)).enumerate() {
                match kind {
                    Kind::Same(idx) => match &mut run {
                        Some((r, start, len)) if *start + *len == idx && *r + *len == row => *len += 1,
                        _ => {
                            flush(&mut svg, run);
                            run = Some((row, idx, 1));
                        }
                    },
                    Kind::Add | Kind::Del => {
                        let class = if kind == Kind::Add { "a" } else { "d" };
                        let ty = top + LINE_HEIGHT * row;
                        write!(
                            svg,
                            r##"<rect class="{class}b" x="{}" y="{ty}" width="{}" height="{LINE_HEIGHT}" rx="3"/><use href="#l{}" y="{}"/>"##,
                            PAD - 6,
                            width - 2 * PAD + 12,
                            line_ids[line],
                            ty + LINE_HEIGHT - 4
                        )
                        .unwrap();
                    }
                }
            }
            flush(&mut svg, run);
        }

        svg.push_str("</g>\n");
    }

    svg.push_str("</svg>\n");
    svg
}
