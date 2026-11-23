//! Synthetisches Luftbild-Stereopaar mit bekannter Geometrie (Prüfdatensatz).

use crate::camera::Camera;
use crate::geom::{r_cv_from_opk, Pose};
use crate::imgutil::Gray;
use crate::project::{Gcp, ImageEntry, Mark, Project, Settings};
use anyhow::Result;
use nalgebra::Vector3;
use rayon::prelude::*;
use std::path::Path;

pub const E0: f64 = 750_000.0;
pub const N0: f64 = 5_400_000.0;

fn hash(ix: i64, iy: i64, seed: u64) -> f64 {
    let mut h = (ix as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (iy as u64).wrapping_mul(0xC2B2AE3D27D4EB4F) ^ seed.wrapping_mul(0x165667B19E3779F9);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    (h >> 11) as f64 / (1u64 << 53) as f64
}

fn vnoise(x: f64, y: f64, seed: u64) -> f64 {
    let (ix, iy) = (x.floor() as i64, y.floor() as i64);
    let (fx, fy) = (x - ix as f64, y - iy as f64);
    let s = |t: f64| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (s(fx), s(fy));
    let a = hash(ix, iy, seed);
    let b = hash(ix + 1, iy, seed);
    let c = hash(ix, iy + 1, seed);
    let d = hash(ix + 1, iy + 1, seed);
    a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy
}

/// Gebäude: (xmin, xmax, ymin, ymax, Höhe über Gelände) in lokalen Koordinaten
pub const BUILDINGS: [(f64, f64, f64, f64, f64); 2] = [(30.0, 60.0, -40.0, -20.0, 12.0), (-90.0, -75.0, 40.0, 55.0, 8.0)];

/// Gelände (DGM) in lokalen Koordinaten
pub fn ground(x: f64, y: f64) -> f64 {
    300.0 + 8.0 * (x / 70.0).sin() * (y / 90.0).cos() + 0.02 * x + 2.0 * (x / 23.0 + y / 31.0).sin()
}

/// Oberfläche (DOM) in lokalen Koordinaten
pub fn surface(x: f64, y: f64) -> f64 {
    let g = ground(x, y);
    for &(x0, x1, y0, y1, h) in &BUILDINGS {
        if x >= x0 && x <= x1 && y >= y0 && y <= y1 {
            return ground(0.5 * (x0 + x1), 0.5 * (y0 + y1)) + h;
        }
    }
    g
}

fn texture(x: f64, y: f64, targets: &[(f64, f64)]) -> f64 {
    for &(tx, ty) in targets {
        let (dx, dy) = (x - tx, y - ty);
        if dx.abs() < 0.9 && dy.abs() < 0.9 {
            return if (dx >= 0.0) == (dy >= 0.0) { 245.0 } else { 15.0 };
        }
        if dx.abs() < 1.3 && dy.abs() < 1.3 {
            return 128.0;
        }
    }
    let mut v = 0.0;
    let mut amp = 1.0;
    let mut sc = 1.0 / 12.0;
    let mut norm = 0.0;
    for o in 0..7 {
        v += amp * vnoise(x * sc, y * sc, o);
        norm += amp;
        amp *= 0.62;
        sc *= 2.0;
    }
    40.0 + 175.0 * (v / norm - 0.5) * 2.2 + 87.0
}

fn render(cam: &Camera, pose: &Pose, targets: &[(f64, f64)], noise: f64, seed: u64) -> Gray {
    let (w, h) = (cam.width as usize, cam.height as usize);
    let mut img = Gray::new(w, h);
    let rt = pose.r.transpose();
    img.data.par_chunks_mut(w).enumerate().for_each(|(v, row)| {
        for (u, px) in row.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (su, sv) in [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)] {
                let n = cam.pixel_to_norm(u as f64 + su, v as f64 + sv);
                let d = rt * Vector3::new(n.x, n.y, 1.0);
                // Strahl ab Höhe 345 m abwärts verfolgen
                let c = pose.c;
                let t0 = (345.0 - c.z) / d.z;
                let step = 0.5 / d.z.abs();
                let f = |t: f64| {
                    let p = c + d * t;
                    p.z - surface(p.x, p.y)
                };
                let mut ta = t0;
                let mut tb = t0;
                for _ in 0..400 {
                    tb = ta + step;
                    if f(tb) <= 0.0 {
                        break;
                    }
                    ta = tb;
                }
                for _ in 0..30 {
                    let tm = 0.5 * (ta + tb);
                    if f(tm) > 0.0 { ta = tm } else { tb = tm }
                }
                let p = c + d * tb;
                acc += texture(p.x, p.y, targets);
            }
            let nz = (hash(u as i64, v as i64, seed) + hash(v as i64, u as i64, seed + 7) - 1.0) * noise * 2.45;
            *px = (acc / 4.0 + nz) as f32;
        }
    });
    img
}

/// Erzeugt Bilder und Projektdatei in `dir`. Rückgabe: Pfad der Projektdatei.
pub fn generate(dir: &Path) -> Result<std::path::PathBuf> {
    generate_with(dir, 2)
}

/// Wie `generate`, aber mit `n` ≥ 2 Bildern in einem Streifen (Basis ≈ 122/(n−1) m, Bild 1 und n
/// wie beim Stereopaar). Bilder: bild1.png … bildN.png.
pub fn generate_with(dir: &Path, n: usize) -> Result<std::path::PathBuf> {
    assert!(n >= 2);
    std::fs::create_dir_all(dir)?;
    let cam = Camera { width: 2400, height: 1800, fx: 3000.0, fy: 3000.0, cx: 1203.2, cy: 897.6, k1: -0.06, k2: 0.012, k3: 0.0, p1: 0.0004, p2: -0.0002 };
    let first = Pose { r: r_cv_from_opk(0.010, -0.008, 0.02), c: Vector3::new(-60.0, 2.0, 750.0) };
    let last = Pose { r: r_cv_from_opk(-0.006, 0.012, 0.035), c: Vector3::new(62.0, -3.0, 752.0) };
    let poses: Vec<Pose> = (0..n)
        .map(|k| {
            if k == 0 {
                first.clone()
            } else if k == n - 1 {
                last.clone()
            } else {
                // Zwischenbilder: lineare Interpolation der Lage, leicht abweichende Drehung
                let t = k as f64 / (n - 1) as f64;
                let w = |a: f64, b: f64| a + t * (b - a);
                Pose { r: r_cv_from_opk(w(0.010, -0.006) + 0.004, w(-0.008, 0.012) - 0.003, w(0.02, 0.035)), c: first.c + (last.c - first.c) * t }
            }
        })
        .collect();
    let ctrl: Vec<(f64, f64)> = vec![(-110.0, -95.0), (0.0, -100.0), (110.0, -90.0), (-115.0, 5.0), (115.0, 0.0), (-105.0, 95.0), (5.0, 100.0), (110.0, 95.0)];
    let checks: Vec<(f64, f64)> = vec![(-50.0, -55.0), (60.0, 30.0), (-30.0, 60.0), (80.0, -70.0)];
    let targets: Vec<(f64, f64)> = ctrl.iter().chain(&checks).cloned().collect();
    let mut p = Project { name: "Synthetisches Luftbildpaar".into(), ..Default::default() };
    p.images.clear();
    for k in 0..n {
        let img = render(&cam, &poses[k], &targets, 2.0, 11 + k as u64);
        let name = format!("bild{}.png", k + 1);
        img.save_png(&dir.join(&name))?;
        p.images.push(ImageEntry { path: name, camera: cam.clone() });
    }
    // Passpunkte (absolute Koordinaten) und Bildmessungen mit Rauschen 0,3 px
    let mut seed = 99u64;
    let mut gauss = || {
        seed += 1;
        let u1 = hash(seed as i64, 1, 5).max(1e-12);
        let u2 = hash(seed as i64, 2, 5);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    for (i, &(x, y)) in targets.iter().enumerate() {
        let check = i >= ctrl.len();
        let id = if check { format!("K{}", i - ctrl.len() + 1) } else { format!("P{}", i + 1) };
        let z = surface(x, y);
        p.gcps.push(Gcp { id: id.clone(), x: x + E0, y: y + N0, z, sxy: 0.02, sz: 0.02, check });
        for k in 0..n {
            if let Some(px) = cam.project_cam(&poses[k].to_cam(&Vector3::new(x, y, z))) {
                if cam.in_image(&px, 5.0) {
                    p.marks.push(Mark { id: id.clone(), image: k, u: px.x + 0.3 * gauss(), v: px.y + 0.3 * gauss() });
                }
            }
        }
    }
    p.settings = Settings { dense_scale: 1.0, cell: 0.5, export_cloud: false, sigma_px: 0.3, ..Default::default() };
    let path = dir.join("projekt.toml");
    p.save(&path)?;
    Ok(path)
}
