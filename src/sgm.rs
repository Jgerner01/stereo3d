//! Dichte Zuordnung: Census-Transformation + Semi-Global Matching (8 Pfade),
//! kachelweise, hierarchisch (grob → fein) mit kachelweisem Disparitätsbereich,
//! Links-Rechts-Prüfung, Subpixel-Interpolation, Median- und Speckle-Filter.

use crate::imgutil::Gray;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct SgmParams {
    pub p1: u16,
    pub p2: u16,
    /// Eindeutigkeitsschwelle (z. B. 0.03 = 3 %)
    pub uniqueness: f32,
    /// zulässige Links-Rechts-Differenz [px]
    pub lr_max: f32,
    /// Mindestgröße zusammenhängender Bereiche [px]
    pub speckle_size: usize,
    /// max. Disparitätssprung innerhalb eines Bereichs [px]
    pub speckle_diff: f32,
    pub tile: usize,
}

impl Default for SgmParams {
    fn default() -> Self {
        Self { p1: 8, p2: 96, uniqueness: 0.03, lr_max: 1.0, speckle_size: 100, speckle_diff: 1.0, tile: 320 }
    }
}

/// Disparitätskarte im linken Normalbild, d' = u_l − u_r; NaN = ungültig.
#[derive(Clone, Debug)]
pub struct Disparity {
    pub w: usize,
    pub h: usize,
    pub d: Vec<f32>,
}

const CW: isize = 4; // Census 9×7
const CH: isize = 3;
const INVALID_COST: u8 = 64;

fn census_region(img: &Gray, x0: isize, y0: isize, w: usize, h: usize) -> Vec<u64> {
    let mut out = vec![0u64; w * h];
    for y in 0..h {
        for x in 0..w {
            let cx = x0 + x as isize;
            let cy = y0 + y as isize;
            let c = img.at(cx, cy);
            let mut bits = 0u64;
            for dy in -CH..=CH {
                for dx in -CW..=CW {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    bits = (bits << 1) | (img.at(cx + dx, cy + dy) < c) as u64;
                }
            }
            out[y * w + x] = bits;
        }
    }
    out
}

struct TileOut {
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    disp: Vec<f32>,
}

/// SGM in einem Bildausschnitt [x0, x0+w) × [y0, y0+h) mit d' ∈ [dmin, dmin+nd).
fn sgm_tile(
    left: &Gray,
    right: &Gray,
    lmask: Option<&[bool]>,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    dmin: i32,
    nd: usize,
    p: &SgmParams,
) -> Vec<f32> {
    let cl = census_region(left, x0 as isize, y0 as isize, w, h);
    // rechter Streifen: x_r = x_l − d
    let rx0 = x0 as isize - (dmin as isize + nd as isize - 1);
    let rw = w + nd - 1;
    let cr = census_region(right, rx0, y0 as isize, rw, h);
    let rvalid = |xr: isize| xr >= CW && xr < right.w as isize - CW;

    // Kostenvolumen
    let mut cost = vec![INVALID_COST; w * h * nd];
    for y in 0..h {
        for x in 0..w {
            let a = cl[y * w + x];
            let base = (y * w + x) * nd;
            for k in 0..nd {
                let xr = x0 as isize + x as isize - (dmin as isize + k as isize);
                if rvalid(xr) {
                    let b = cr[y * rw + (xr - rx0) as usize];
                    cost[base + k] = (a ^ b).count_ones() as u8;
                }
            }
        }
    }

    let lum = |x: usize, y: usize| left.get(x0 + x, y0 + y);
    let mut sum = vec![0u16; w * h * nd];
    let dirs: [(isize, isize); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)];
    let mut prev = vec![0u16; w * nd];
    let mut cur = vec![0u16; w * nd];
    let mut prev_min = vec![0u16; w];
    let mut cur_min = vec![0u16; w];
    for &(dx, dy) in &dirs {
        if dy == 0 {
            let mut lr = vec![0u16; nd];
            let mut lr2 = vec![0u16; nd];
            for y in 0..h {
                let xs: Box<dyn Iterator<Item = usize>> = if dx > 0 { Box::new(0..w) } else { Box::new((0..w).rev()) };
                let mut first = true;
                let mut pmin = 0u16;
                let mut px = 0usize;
                for x in xs {
                    let base = (y * w + x) * nd;
                    if first {
                        for k in 0..nd {
                            lr[k] = cost[base + k] as u16;
                        }
                        first = false;
                    } else {
                        let di = (lum(x, y) - lum(px, y)).abs();
                        let p2 = ((p.p2 as f32 / (1.0 + di / 16.0)) as u16).max(p.p1 + 1);
                        aggregate_step(&cost[base..base + nd], &lr, pmin, p.p1, p2, &mut lr2);
                        std::mem::swap(&mut lr, &mut lr2);
                    }
                    pmin = *lr.iter().min().unwrap();
                    for k in 0..nd {
                        sum[base + k] += lr[k];
                    }
                    px = x;
                }
            }
        } else {
            let ys: Vec<usize> = if dy > 0 { (0..h).collect() } else { (0..h).rev().collect() };
            let mut first_row = true;
            let mut py = 0usize;
            for &y in &ys {
                for x in 0..w {
                    let base = (y * w + x) * nd;
                    let pxi = x as isize - dx;
                    let o = &mut cur[x * nd..(x + 1) * nd];
                    if first_row || pxi < 0 || pxi >= w as isize {
                        for k in 0..nd {
                            o[k] = cost[base + k] as u16;
                        }
                    } else {
                        let pxu = pxi as usize;
                        let di = (lum(x, y) - lum(pxu, py)).abs();
                        let p2 = ((p.p2 as f32 / (1.0 + di / 16.0)) as u16).max(p.p1 + 1);
                        aggregate_step(&cost[base..base + nd], &prev[pxu * nd..(pxu + 1) * nd], prev_min[pxu], p.p1, p2, o);
                    }
                    cur_min[x] = *o.iter().min().unwrap();
                    for k in 0..nd {
                        sum[base + k] += o[k];
                    }
                }
                std::mem::swap(&mut prev, &mut cur);
                std::mem::swap(&mut prev_min, &mut cur_min);
                first_row = false;
                py = y;
            }
        }
    }
    drop(cost);

    // WTA links
    let mut dl = vec![f32::NAN; w * h];
    let mut dr_int = vec![i32::MIN; w * h];
    for y in 0..h {
        // rechte Disparitäten (für LR-Prüfung) im Tile-Bezug: x_r_tile = x − (d − dmin)?
        // Wir nutzen x_r = x0 + x − d → Index über linke Spalte x_l = x_r + d.
        let mut best_r = vec![(u16::MAX, 0i32); w + nd];
        for x in 0..w {
            let base = (y * w + x) * nd;
            for k in 0..nd {
                // x_r relativ: x − k (Verschiebung um dmin konstant) → Index x − k + nd − 1
                let ir = x + nd - 1 - k;
                let s = sum[base + k];
                if s < best_r[ir].0 {
                    best_r[ir] = (s, dmin + k as i32);
                }
            }
        }
        for x in 0..w {
            if let Some(m) = lmask {
                if !m[(y0 + y) * left.w + x0 + x] {
                    continue;
                }
            }
            let base = (y * w + x) * nd;
            let s = &sum[base..base + nd];
            let (mut kb, mut sb) = (0usize, u16::MAX);
            for k in 0..nd {
                if s[k] < sb {
                    sb = s[k];
                    kb = k;
                }
            }
            // Eindeutigkeit
            let mut s2 = u16::MAX;
            for k in 0..nd {
                if (k as isize - kb as isize).abs() > 1 && s[k] < s2 {
                    s2 = s[k];
                }
            }
            if (s2 as f32) * (1.0 - p.uniqueness) < sb as f32 {
                continue;
            }
            if kb == 0 || kb == nd - 1 {
                continue; // Rand des Suchbereichs: unsicher
            }
            // LR-Prüfung
            let ir = x + nd - 1 - kb;
            let d_r = best_r[ir].1;
            if (d_r - (dmin + kb as i32)).abs() as f32 > p.lr_max {
                continue;
            }
            let (c0, c1, c2) = (s[kb - 1] as f32, s[kb] as f32, s[kb + 1] as f32);
            let den = c0 - 2.0 * c1 + c2;
            let sub = if den > 0.0 { 0.5 * (c0 - c2) / den } else { 0.0 };
            dl[y * w + x] = dmin as f32 + kb as f32 + sub.clamp(-0.5, 0.5);
        }
        drop(best_r);
        let _ = &mut dr_int;
    }
    dl
}

#[inline]
fn aggregate_step(c: &[u8], prev: &[u16], prev_min: u16, p1: u16, p2: u16, out: &mut [u16]) {
    let nd = c.len();
    let jump = prev_min + p2;
    for k in 0..nd {
        let mut m = prev[k];
        if k > 0 {
            m = m.min(prev[k - 1] + p1);
        }
        if k + 1 < nd {
            m = m.min(prev[k + 1] + p1);
        }
        m = m.min(jump);
        out[k] = c[k] as u16 + m - prev_min;
    }
}

/// Kachelweise SGM. `range(tx0,ty0,tw,th)` liefert den Disparitätsbereich je Kachel.
fn sgm_tiled<F>(left: &Gray, right: &Gray, lmask: Option<&[bool]>, p: &SgmParams, range: F) -> Disparity
where
    F: Fn(usize, usize, usize, usize) -> Option<(i32, usize)> + Sync,
{
    let (w, h) = (left.w, left.h);
    let t = p.tile;
    let m = 24usize;
    let mut tiles = Vec::new();
    let mut ty = 0;
    while ty < h {
        let mut tx = 0;
        while tx < w {
            tiles.push((tx, ty, t.min(w - tx), t.min(h - ty)));
            tx += t;
        }
        ty += t;
    }
    let outs: Vec<TileOut> = tiles
        .par_iter()
        .filter_map(|&(tx, ty, tw, th)| {
            let (dmin, nd) = range(tx, ty, tw, th)?;
            let x0 = tx.saturating_sub(m);
            let y0 = ty.saturating_sub(m);
            let x1 = (tx + tw + m).min(w);
            let y1 = (ty + th + m).min(h);
            let d = sgm_tile(left, right, lmask, x0, y0, x1 - x0, y1 - y0, dmin, nd.max(3), p);
            // Kern ausschneiden
            let mut core = vec![f32::NAN; tw * th];
            for y in 0..th {
                for x in 0..tw {
                    core[y * tw + x] = d[(ty + y - y0) * (x1 - x0) + (tx + x - x0)];
                }
            }
            Some(TileOut { x0: tx, y0: ty, w: tw, h: th, disp: core })
        })
        .collect();
    let mut d = vec![f32::NAN; w * h];
    for o in outs {
        for y in 0..o.h {
            for x in 0..o.w {
                d[(o.y0 + y) * w + o.x0 + x] = o.disp[y * o.w + x];
            }
        }
    }
    Disparity { w, h, d }
}

/// Hierarchisches SGM. `dmin..=dmax`: globaler Bereich d' im Feinbild.
pub fn match_pair(
    left: &Gray,
    right: &Gray,
    lmask: Option<&[bool]>,
    dmin: f64,
    dmax: f64,
    p: &SgmParams,
    log: &dyn Fn(String),
) -> Disparity {
    let range = (dmax - dmin).max(8.0);
    let mut k = 1usize;
    while range / k as f64 > 96.0 && left.w / (2 * k) > 150 && left.h / (2 * k) > 150 {
        k *= 2;
    }
    if k == 1 {
        log(format!("SGM: eine Stufe, d' = {:.0} … {:.0}", dmin, dmax));
        let (a, b) = (dmin.floor() as i32 - 2, dmax.ceil() as i32 + 2);
        let mut d = sgm_tiled(left, right, lmask, p, |_, _, _, _| Some((a, (b - a + 1) as usize)));
        post_filter(&mut d, p);
        return d;
    }
    // Grobstufe
    let lc = left.downsample(k);
    let rc = right.downsample(k);
    let mc: Option<Vec<bool>> = lmask.map(|m| {
        let mut o = vec![false; lc.w * lc.h];
        for y in 0..lc.h {
            for x in 0..lc.w {
                o[y * lc.w + x] = m[(y * k + k / 2) * left.w + x * k + k / 2];
            }
        }
        o
    });
    let (a, b) = ((dmin / k as f64).floor() as i32 - 2, (dmax / k as f64).ceil() as i32 + 2);
    log(format!("SGM Grobstufe 1:{k}: {}×{} px, d' = {a} … {b}", lc.w, lc.h));
    let mut pc = p.clone();
    pc.speckle_size = (p.speckle_size / (k * k)).max(10);
    let mut dc = sgm_tiled(&lc, &rc, mc.as_deref(), &pc, |_, _, _, _| Some((a, (b - a + 1) as usize)));
    post_filter(&mut dc, &pc);
    // Feinstufe mit kachelweisem Bereich
    log(format!("SGM Feinstufe: {}×{} px", left.w, left.h));
    let kf = k as f32;
    let mut d = sgm_tiled(left, right, lmask, p, |tx, ty, tw, th| {
        let mut vals = Vec::new();
        let (cx0, cy0) = (tx / k, ty / k);
        let (cx1, cy1) = (((tx + tw) / k + 1).min(dc.w), ((ty + th) / k + 1).min(dc.h));
        for y in cy0..cy1 {
            for x in cx0..cx1 {
                let v = dc.d[y * dc.w + x];
                if v.is_finite() {
                    vals.push(v * kf);
                }
            }
        }
        if vals.len() < 10 {
            return None;
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lo = vals[vals.len() / 100];
        let hi = vals[vals.len() - 1 - vals.len() / 100];
        let a = (lo - 2.0 * kf - 4.0).floor() as i32;
        let b = (hi + 2.0 * kf + 4.0).ceil() as i32;
        Some((a, ((b - a + 1) as usize).min(384)))
    });
    post_filter(&mut d, p);
    d
}

fn post_filter(d: &mut Disparity, p: &SgmParams) {
    median3(d);
    speckle(d, p.speckle_size, p.speckle_diff);
}

fn median3(d: &mut Disparity) {
    let (w, h) = (d.w, d.h);
    let src = d.d.clone();
    d.d.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if y == 0 || y + 1 >= h {
            return;
        }
        let mut buf = [0f32; 9];
        for x in 1..w - 1 {
            if !src[y * w + x].is_finite() {
                continue;
            }
            let mut n = 0;
            for yy in y - 1..=y + 1 {
                for xx in x - 1..=x + 1 {
                    let v = src[yy * w + xx];
                    if v.is_finite() {
                        buf[n] = v;
                        n += 1;
                    }
                }
            }
            if n >= 5 {
                let s = &mut buf[..n];
                s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                row[x] = s[n / 2];
            }
        }
    });
}

fn speckle(d: &mut Disparity, min_size: usize, max_diff: f32) {
    let (w, h) = (d.w, d.h);
    let mut label = vec![0u32; w * h];
    let mut stack = Vec::new();
    let mut region = Vec::new();
    let mut cur = 0u32;
    for i in 0..w * h {
        if label[i] != 0 || !d.d[i].is_finite() {
            continue;
        }
        cur += 1;
        label[i] = cur;
        stack.push(i);
        region.clear();
        while let Some(j) = stack.pop() {
            region.push(j);
            let (x, y) = (j % w, j / w);
            let v = d.d[j];
            let mut try_push = |k: usize, label: &mut Vec<u32>| {
                if label[k] == 0 && d.d[k].is_finite() && (d.d[k] - v).abs() <= max_diff {
                    label[k] = cur;
                    stack.push(k);
                }
            };
            if x > 0 {
                try_push(j - 1, &mut label);
            }
            if x + 1 < w {
                try_push(j + 1, &mut label);
            }
            if y > 0 {
                try_push(j - w, &mut label);
            }
            if y + 1 < h {
                try_push(j + w, &mut label);
            }
        }
        if region.len() < min_size {
            for &j in &region {
                d.d[j] = f32::NAN;
            }
        }
    }
}
