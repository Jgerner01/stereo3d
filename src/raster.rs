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

    /// GeoTIFF (Float32, 1 Band, unkomprimiert, NoData -9999). `epsg` = 0: kein CRS-Eintrag.
    pub fn write_geotiff(&self, path: &Path, epsg: u32) -> Result<()> {
        std::fs::write(path, self.geotiff_bytes(epsg))?;
        Ok(())
    }

    pub fn geotiff_bytes(&self, epsg: u32) -> Vec<u8> {
        // Pixeldaten: Zeile 0 = Norden
        let mut data = Vec::with_capacity(self.nx * self.ny * 4);
        for iy in (0..self.ny).rev() {
            for ix in 0..self.nx {
                let v = self.z[self.idx(ix, iy)];
                data.extend_from_slice(&(if v.is_finite() { v } else { -9999.0 }).to_le_bytes());
            }
        }
        let top = self.y0 + self.ny as f64 * self.cell;
        let mut extra: Vec<u8> = Vec::new(); // Daten außerhalb der IFD-Einträge
        // (tag, type, count, value-or-offset-marker)
        struct E(u16, u16, u32, Vec<u8>);
        let u16b = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let f64b = |v: &[f64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let mut keys: Vec<u16> = vec![1, 1, 0, 0];
        let mut add = |id: u16, val: u16| {
            keys.extend_from_slice(&[id, 0, 1, val]);
        };
        add(1024, 1); // GTModelType = projected
        add(1025, 1); // GTRasterType = PixelIsArea
        if epsg > 0 && epsg < 32767 {
            add(3072, epsg as u16); // ProjectedCSType
        } else {
            add(3072, 32767); // user-defined
        }
        keys[3] = ((keys.len() - 4) / 4) as u16;
        let strip_bytes = data.len() as u32;
        let mut entries = vec![
            E(256, 4, 1, (self.nx as u32).to_le_bytes().to_vec()),
            E(257, 4, 1, (self.ny as u32).to_le_bytes().to_vec()),
            E(258, 3, 1, u16b(&[32])),
            E(259, 3, 1, u16b(&[1])),
            E(262, 3, 1, u16b(&[1])),
            E(273, 4, 1, Vec::new()), // StripOffsets, später gesetzt
            E(277, 3, 1, u16b(&[1])),
            E(278, 4, 1, (self.ny as u32).to_le_bytes().to_vec()),
            E(279, 4, 1, strip_bytes.to_le_bytes().to_vec()),
            E(339, 3, 1, u16b(&[3])), // SampleFormat = IEEE float
            E(33550, 12, 3, f64b(&[self.cell, self.cell, 0.0])),
            E(33922, 12, 6, f64b(&[0.0, 0.0, 0.0, self.x0, top, 0.0])),
            E(34735, 3, keys.len() as u32, u16b(&keys)),
            E(42113, 2, 8, b"-9999   ".to_vec()),
        ];
        entries.sort_by_key(|e| e.0);
        let n = entries.len();
        let ifd_off = 8usize;
        let ifd_len = 2 + n * 12 + 4;
        let mut extra_off = ifd_off + ifd_len;
        // Bilddaten nach den Extras, daher zuerst Extras bestimmen
        let mut placed: Vec<(u16, u16, u32, [u8; 4])> = Vec::new();
        let mut strip_slot = None;
        for (i, e) in entries.iter().enumerate() {
            let mut val = [0u8; 4];
            if e.0 == 273 {
                strip_slot = Some(i);
            } else if e.3.len() <= 4 {
                val[..e.3.len()].copy_from_slice(&e.3);
            } else {
                val.copy_from_slice(&(extra_off as u32).to_le_bytes());
                extra.extend_from_slice(&e.3);
                extra_off += e.3.len();
            }
            placed.push((e.0, e.1, e.2, val));
        }
        let data_off = extra_off;
        if let Some(i) = strip_slot {
            placed[i].3 = (data_off as u32).to_le_bytes();
        }
        let mut out = Vec::with_capacity(data_off + data.len());
        out.extend_from_slice(b"II");
        out.extend_from_slice(&42u16.to_le_bytes());
        out.extend_from_slice(&(ifd_off as u32).to_le_bytes());
        out.extend_from_slice(&(n as u16).to_le_bytes());
        for (tag, ty, cnt, val) in &placed {
            out.extend_from_slice(&tag.to_le_bytes());
            out.extend_from_slice(&ty.to_le_bytes());
            out.extend_from_slice(&cnt.to_le_bytes());
            out.extend_from_slice(val);
        }
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&extra);
        out.extend_from_slice(&data);
        out
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
