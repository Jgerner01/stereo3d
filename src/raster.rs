//! Rasterung der Punktwolke (DOM), einfacher Bodenfilter (DGM) und Exporte.

use anyhow::Result;
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Grid {
    /// linke untere Ecke des Rasters (Zellrand), absolut
    pub x0: f64,
    pub y0: f64,
    pub cell: f64,
    pub nx: usize,
    pub ny: usize,
    /// Zeile 0 = Süden (y0); NaN = keine Daten
    pub z: Vec<f32>,
}

impl Grid {
    pub fn idx(&self, ix: usize, iy: usize) -> usize {
        iy * self.nx + ix
    }

    /// Median der Höhen je Zelle. `pts`: absolute Koordinaten.
    pub fn from_points(pts: &[[f64; 3]], cell: f64) -> Option<Grid> {
        if pts.is_empty() {
            return None;
        }
        let (mut xmin, mut ymin, mut xmax, mut ymax) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for p in pts {
            xmin = xmin.min(p[0]);
            ymin = ymin.min(p[1]);
            xmax = xmax.max(p[0]);
            ymax = ymax.max(p[1]);
        }
        let x0 = (xmin / cell).floor() * cell;
        let y0 = (ymin / cell).floor() * cell;
        let nx = ((xmax - x0) / cell).floor() as usize + 1;
        let ny = ((ymax - y0) / cell).floor() as usize + 1;
        if nx * ny > 400_000_000 {
            return None;
        }
        let mut bins: Vec<Vec<f32>> = vec![Vec::new(); nx * ny];
        // Höhen relativ zu einem Bezug speichern (f32-Genauigkeit)
        for p in pts {
            let ix = ((p[0] - x0) / cell) as usize;
            let iy = ((p[1] - y0) / cell) as usize;
            bins[iy * nx + ix].push(p[2] as f32);
        }
        let z = bins
            .into_iter()
            .map(|mut b| {
                if b.is_empty() {
                    f32::NAN
                } else {
                    let n = b.len();
                    b.select_nth_unstable_by(n / 2, |a, c| a.partial_cmp(c).unwrap());
                    b[n / 2]
                }
            })
            .collect();
        Some(Grid { x0, y0, cell, nx, ny, z })
    }

    /// Lücken bis zu `max_cells` Zellen Abstand per inverser Distanzgewichtung füllen.
    pub fn fill_gaps(&mut self, max_cells: usize) {
        let src = self.z.clone();
        let r = max_cells as isize;
        for iy in 0..self.ny {
            for ix in 0..self.nx {
                let i = self.idx(ix, iy);
                if src[i].is_finite() {
                    continue;
                }
                let (mut sw, mut sz) = (0f64, 0f64);
                let mut quadrants = [false; 4];
                for dy in -r..=r {
                    for dx in -r..=r {
                        let (x, y) = (ix as isize + dx, iy as isize + dy);
                        if x < 0 || y < 0 || x >= self.nx as isize || y >= self.ny as isize {
                            continue;
                        }
                        let v = src[y as usize * self.nx + x as usize];
                        if !v.is_finite() {
                            continue;
                        }
                        let d2 = (dx * dx + dy * dy) as f64;
                        if d2 > (r * r) as f64 {
                            continue;
                        }
                        let w = 1.0 / d2;
                        sw += w;
                        sz += w * v as f64;
                        quadrants[((dx >= 0) as usize) + 2 * ((dy >= 0) as usize)] = true;
                    }
                }
                // nur interpolieren, nicht extrapolieren
                if sw > 0.0 && quadrants.iter().filter(|q| **q).count() >= 3 {
                    self.z[i] = (sz / sw) as f32;
                }
            }
        }
    }

    /// Progressiver morphologischer Filter (Zhang et al. 2003), vereinfachte Form.
    /// `max_window` [m], `slope` [m/m], `dh0`, `dh_max` [m]. Liefert DGM (Nicht-Boden = NaN, danach gefüllt).
    pub fn ground_filter(&self, max_window: f64, slope: f64, dh0: f64, dh_max: f64) -> Grid {
        let mut surf = self.clone();
        // Lücken vorläufig mit Minimum der Umgebung schließen, damit Morphologie arbeitet
        let zmin = self.z.iter().filter(|v| v.is_finite()).fold(f32::MAX, |a, &b| a.min(b));
        for v in surf.z.iter_mut() {
            if !v.is_finite() {
                *v = zmin;
            }
        }
        let mut ground = vec![true; self.z.len()];
        let mut wprev = 0.0;
        let mut k = 0;
        loop {
            let half = 1usize << k; // Fensterhalbbreite in Zellen: 1, 2, 4, …
            let wsize = (2 * half + 1) as f64 * self.cell;
            if wsize > max_window {
                break;
            }
            let opened = dilate(&erode(&surf.z, self.nx, self.ny, half), self.nx, self.ny, half);
            let dh = if k == 0 { dh0 } else { (slope * (wsize - wprev) + dh0).min(dh_max) };
            for i in 0..ground.len() {
                if surf.z[i] - opened[i] > dh as f32 {
                    ground[i] = false;
                }
            }
            surf.z = opened;
            wprev = wsize;
            k += 1;
        }
        let mut out = self.clone();
        for i in 0..ground.len() {
            if !ground[i] {
                out.z[i] = f32::NAN;
            }
        }
        out.fill_gaps((max_window / self.cell).ceil() as usize);
        out
    }

    pub fn write_ascii_grid(&self, path: &Path) -> Result<()> {
        let mut f = BufWriter::new(std::fs::File::create(path)?);
        writeln!(f, "ncols {}", self.nx)?;
        writeln!(f, "nrows {}", self.ny)?;
        writeln!(f, "xllcorner {:.4}", self.x0)?;
        writeln!(f, "yllcorner {:.4}", self.y0)?;
        writeln!(f, "cellsize {}", self.cell)?;
        writeln!(f, "NODATA_value -9999")?;
        for iy in (0..self.ny).rev() {
            let mut line = String::with_capacity(self.nx * 9);
            for ix in 0..self.nx {
                let v = self.z[self.idx(ix, iy)];
                if v.is_finite() {
                    line.push_str(&format!("{:.3} ", v));
                } else {
                    line.push_str("-9999 ");
                }
            }
            writeln!(f, "{}", line.trim_end())?;
        }
        Ok(())
    }

    /// Zellmitten als XYZ (nur gültige Zellen) – z. B. als Eingang für DGM-Programme.
    pub fn write_xyz(&self, path: &Path) -> Result<()> {
        let mut f = BufWriter::new(std::fs::File::create(path)?);
        for iy in 0..self.ny {
            for ix in 0..self.nx {
                let v = self.z[self.idx(ix, iy)];
                if v.is_finite() {
                    let x = self.x0 + (ix as f64 + 0.5) * self.cell;
                    let y = self.y0 + (iy as f64 + 0.5) * self.cell;
                    writeln!(f, "{:.3} {:.3} {:.3}", x, y, v)?;
                }
            }
        }
        Ok(())
    }

    /// Bilineare Höhe an Position (absolut); None außerhalb/ohne Daten.
    pub fn height_at(&self, x: f64, y: f64) -> Option<f64> {
        let fx = (x - self.x0) / self.cell - 0.5;
        let fy = (y - self.y0) / self.cell - 0.5;
        if fx < 0.0 || fy < 0.0 {
            return None;
        }
        let (ix, iy) = (fx as usize, fy as usize);
        if ix + 1 >= self.nx || iy + 1 >= self.ny {
            return None;
        }
        let (tx, ty) = (fx - ix as f64, fy - iy as f64);
        let a = self.z[self.idx(ix, iy)] as f64;
        let b = self.z[self.idx(ix + 1, iy)] as f64;
        let c = self.z[self.idx(ix, iy + 1)] as f64;
        let d = self.z[self.idx(ix + 1, iy + 1)] as f64;
        let v = a * (1.0 - tx) * (1.0 - ty) + b * tx * (1.0 - ty) + c * (1.0 - tx) * ty + d * tx * ty;
        v.is_finite().then_some(v)
    }
}

fn morph(z: &[f32], nx: usize, ny: usize, half: usize, take_min: bool) -> Vec<f32> {
    // separabel: erst Zeilen, dann Spalten (quadratisches Fenster)
    let pick = |a: f32, b: f32| if take_min { a.min(b) } else { a.max(b) };
    let mut tmp = vec![0f32; z.len()];
    for y in 0..ny {
        for x in 0..nx {
            let mut v = z[y * nx + x];
            for xx in x.saturating_sub(half)..=(x + half).min(nx - 1) {
                v = pick(v, z[y * nx + xx]);
            }
            tmp[y * nx + x] = v;
        }
    }
    let mut out = vec![0f32; z.len()];
    for y in 0..ny {
        for x in 0..nx {
            let mut v = tmp[y * nx + x];
            for yy in y.saturating_sub(half)..=(y + half).min(ny - 1) {
                v = pick(v, tmp[yy * nx + x]);
            }
            out[y * nx + x] = v;
        }
    }
    out
}

fn erode(z: &[f32], nx: usize, ny: usize, half: usize) -> Vec<f32> {
    morph(z, nx, ny, half, true)
}
fn dilate(z: &[f32], nx: usize, ny: usize, half: usize) -> Vec<f32> {
    morph(z, nx, ny, half, false)
}
