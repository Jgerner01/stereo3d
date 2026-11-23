//! Einzelpunktzuordnung: Interestoperator (Harris), Kreuzkorrelation (NCC)
//! und Kleinste-Quadrate-Zuordnung (LSM, affin + radiometrisch linear).

use crate::imgutil::Gray;
use nalgebra::{DMatrix, DVector, Matrix2};

/// Normierte Kreuzkorrelation zweier Fenster (ganzzahlige Mitte links, beliebige rechts).
pub fn ncc(l: &Gray, xl: isize, yl: isize, r: &Gray, xr: f64, yr: f64, half: isize) -> Option<f32> {
    if xl - half < 0 || yl - half < 0 || xl + half >= l.w as isize || yl + half >= l.h as isize {
        return None;
    }
    let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let ix = xr.round() as isize;
    let iy = yr.round() as isize;
    let int = (xr - ix as f64).abs() < 1e-9 && (yr - iy as f64).abs() < 1e-9;
    if int && (ix - half < 0 || iy - half < 0 || ix + half >= r.w as isize || iy + half >= r.h as isize) {
        return None;
    }
    let mut n = 0f64;
    for dy in -half..=half {
        for dx in -half..=half {
            let a = l.get((xl + dx) as usize, (yl + dy) as usize) as f64;
            let b = if int {
                r.get((ix + dx) as usize, (iy + dy) as usize) as f64
            } else {
                r.sample(xr + dx as f64, yr + dy as f64)? as f64
            };
            sa += a;
            sb += b;
            saa += a * a;
            sbb += b * b;
            sab += a * b;
            n += 1.0;
        }
    }
    let va = saa - sa * sa / n;
    let vb = sbb - sb * sb / n;
    if va < 1e-6 || vb < 1e-6 {
        return None;
    }
    Some(((sab - sa * sb / n) / (va * vb).sqrt()) as f32)
}

#[derive(Clone, Debug)]
pub struct LsmResult {
    pub x: f64,
    pub y: f64,
    /// Kovarianz der Verschiebung [px²]
    pub cov: Matrix2<f64>,
    pub sigma0: f64,
    pub rho: f64,
    pub iterations: usize,
}

/// LSM: Template um (xl, yl) im linken Bild, Startwert (xr, yr) im rechten.
/// Modell: g_l(x,y) = r0 + r1 · g_r(a0 + a1 x + a2 y, b0 + b1 x + b2 y).
pub fn lsm(l: &Gray, xl: f64, yl: f64, r: &Gray, xr: f64, yr: f64, half: isize) -> Option<LsmResult> {
    let n = ((2 * half + 1) * (2 * half + 1)) as usize;
    let mut tpl = Vec::with_capacity(n);
    let mut wts = Vec::with_capacity(n);
    let sig_w = half as f64 * 0.8;
    for dy in -half..=half {
        for dx in -half..=half {
            tpl.push((dx as f64, dy as f64, l.sample(xl + dx as f64, yl + dy as f64)? as f64));
            wts.push((-(dx * dx + dy * dy) as f64 / (2.0 * sig_w * sig_w)).exp());
        }
    }
    let mut p = [xr, 1.0, 0.0, yr, 0.0, 1.0, 0.0, 1.0]; // a0 a1 a2 b0 b1 b2 r0 r1
    let mut a = DMatrix::zeros(n, 8);
    let mut b = DVector::zeros(n);
    let mut it = 0;
    let mut converged = false;
    while it < 40 {
        it += 1;
        for (i, &(x, y, g)) in tpl.iter().enumerate() {
            let u = p[0] + p[1] * x + p[2] * y;
            let v = p[3] + p[4] * x + p[5] * y;
            let (gr, gx, gy) = r.sample_grad(u, v)?;
            let (gr, gx, gy) = (gr as f64, gx as f64, gy as f64);
            let sw = wts[i].sqrt();
            let rr = p[7];
            let row = [rr * gx, rr * gx * x, rr * gx * y, rr * gy, rr * gy * x, rr * gy * y, 1.0, gr];
            for j in 0..8 {
                a[(i, j)] = row[j] * sw;
            }
            b[i] = (g - (p[6] + p[7] * gr)) * sw;
        }
        let nmat = a.transpose() * &a;
        let rhs = a.transpose() * &b;
        let dx = nmat.clone().cholesky()?.solve(&rhs);
        for j in 0..8 {
            p[j] += dx[j];
        }
        if (p[0] - xr).abs() > half as f64 || (p[3] - yr).abs() > half as f64 {
            return None;
        }
        if dx[0].abs() < 1e-4 && dx[3].abs() < 1e-4 {
            converged = true;
            break;
        }
    }
    if !converged {
        return None;
    }
    // Statistik
    let nmat = a.transpose() * &a;
    let q = nmat.try_inverse()?;
    let v = &a * DVector::zeros(8) - &b; // Residuen bei Konvergenz ≈ −b
    let s0 = (v.norm_squared() / (n as f64 - 8.0)).sqrt();
    let cov = Matrix2::new(q[(0, 0)], q[(0, 3)], q[(3, 0)], q[(3, 3)]) * s0 * s0;
    // Korrelationskoeffizient nach Anpassung
    let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for &(x, y, g) in &tpl {
        let gr = r.sample(p[0] + p[1] * x + p[2] * y, p[3] + p[4] * x + p[5] * y)? as f64;
        sa += g;
        sb += gr;
        saa += g * g;
        sbb += gr * gr;
        sab += g * gr;
    }
    let nn = n as f64;
    let rho = (sab - sa * sb / nn) / ((saa - sa * sa / nn) * (sbb - sb * sb / nn)).sqrt().max(1e-12);
    Some(LsmResult { x: p[0], y: p[3], cov, sigma0: s0, rho, iterations: it })
}

/// Harris-Ecken, gitterweise verteilt (je Zelle der stärkste Punkt).
pub fn harris_grid(img: &Gray, cells_x: usize, cells_y: usize, border: usize, mask: Option<&[bool]>) -> Vec<(usize, usize, f32)> {
    let (w, h) = (img.w, img.h);
    let mut gxx = vec![0f32; w * h];
    let mut gyy = vec![0f32; w * h];
    let mut gxy = vec![0f32; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = img.get(x + 1, y) - img.get(x - 1, y);
            let gy = img.get(x, y + 1) - img.get(x, y - 1);
            gxx[y * w + x] = gx * gx;
            gyy[y * w + x] = gy * gy;
            gxy[y * w + x] = gx * gy;
        }
    }
    let rad = 2usize;
    let cw = (w - 2 * border) / cells_x.max(1);
    let ch = (h - 2 * border) / cells_y.max(1);
    let mut out = Vec::new();
    for cy in 0..cells_y {
        for cx in 0..cells_x {
            let mut best = (0usize, 0usize, 0f32);
            for y in border + cy * ch..border + (cy + 1) * ch {
                for x in border + cx * cw..border + (cx + 1) * cw {
                    if y < rad + 1 || x < rad + 1 || y + rad + 1 >= h || x + rad + 1 >= w {
                        continue;
                    }
                    if let Some(m) = mask {
                        if !m[y * w + x] {
                            continue;
                        }
                    }
                    let (mut a, mut b, mut c) = (0f32, 0f32, 0f32);
                    for yy in y - rad..=y + rad {
                        for xx in x - rad..=x + rad {
                            a += gxx[yy * w + xx];
                            b += gyy[yy * w + xx];
                            c += gxy[yy * w + xx];
                        }
                    }
                    let resp = a * b - c * c - 0.04 * (a + b) * (a + b);
                    if resp > best.2 {
                        best = (x, y, resp);
                    }
                }
            }
            if best.2 > 0.0 {
                out.push(best);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lsm_subpixel_shift() {
        let (w, h) = (80, 80);
        let f = |x: f64, y: f64| 120.0 + 60.0 * (0.31 * x).sin() * (0.23 * y).cos() + 30.0 * (0.17 * (x + y)).sin();
        let mut l = Gray::new(w, h);
        let mut r = Gray::new(w, h);
        for y in 0..h {
            for x in 0..w {
                l.data[y * w + x] = f(x as f64, y as f64) as f32;
                r.data[y * w + x] = f(x as f64 + 2.37, y as f64 - 1.21) as f32;
            }
        }
        let res = lsm(&l, 40.0, 40.0, &r, 38.0, 41.0, 7).unwrap();
        assert!((res.x - (40.0 - 2.37)).abs() < 0.02, "{}", res.x);
        assert!((res.y - (40.0 + 1.21)).abs() < 0.02, "{}", res.y);
    }
}
