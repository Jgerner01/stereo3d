//! Einfache Grauwertbilder (f32) mit bilinearer Interpolation.

use anyhow::{Context, Result};
use std::path::Path;

#[derive(Clone, Debug, Default)]
pub struct Gray {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Gray {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![0.0; w * h] }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let img = image::open(path).with_context(|| format!("Bild nicht lesbar: {}", path.display()))?;
        let g = img.to_luma32f();
        let (w, h) = (g.width() as usize, g.height() as usize);
        Ok(Self { w, h, data: g.into_raw().into_iter().map(|v| v * 255.0).collect() })
    }

    pub fn save_png(&self, path: &Path) -> Result<()> {
        let buf: Vec<u8> = self.data.iter().map(|v| v.round().clamp(0.0, 255.0) as u8).collect();
        image::GrayImage::from_raw(self.w as u32, self.h as u32, buf)
            .unwrap()
            .save(path)
            .with_context(|| format!("Schreiben fehlgeschlagen: {}", path.display()))?;
        Ok(())
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.w + x]
    }

    #[inline]
    pub fn at(&self, x: isize, y: isize) -> f32 {
        let x = x.clamp(0, self.w as isize - 1) as usize;
        let y = y.clamp(0, self.h as isize - 1) as usize;
        self.data[y * self.w + x]
    }

    /// Bilineare Interpolation; None außerhalb des Bildes.
    #[inline]
    pub fn sample(&self, x: f64, y: f64) -> Option<f32> {
        if !(x >= 0.0 && y >= 0.0 && x <= (self.w - 1) as f64 && y <= (self.h - 1) as f64) {
            return None;
        }
        let x0 = (x.floor() as usize).min(self.w - 2);
        let y0 = (y.floor() as usize).min(self.h - 2);
        let fx = (x - x0 as f64) as f32;
        let fy = (y - y0 as f64) as f32;
        let i = y0 * self.w + x0;
        let a = self.data[i];
        let b = self.data[i + 1];
        let c = self.data[i + self.w];
        let d = self.data[i + self.w + 1];
        Some(a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy)
    }

    /// Wert und Gradient (bilinear, zentrale Differenzen).
    #[inline]
    pub fn sample_grad(&self, x: f64, y: f64) -> Option<(f32, f32, f32)> {
        let v = self.sample(x, y)?;
        let gx = (self.sample(x + 0.5, y)? - self.sample(x - 0.5, y)?) as f32;
        let gy = (self.sample(x, y + 0.5)? - self.sample(x, y - 0.5)?) as f32;
        Some((v, gx, gy))
    }

    /// Verkleinerung um ganzzahligen Faktor (Mittelwert über k×k).
    pub fn downsample(&self, k: usize) -> Gray {
        if k <= 1 {
            return self.clone();
        }
        let (w, h) = (self.w / k, self.h / k);
        let mut out = Gray::new(w, h);
        let inv = 1.0 / (k * k) as f32;
        for y in 0..h {
            for x in 0..w {
                let mut s = 0.0;
                for yy in 0..k {
                    let row = (y * k + yy) * self.w + x * k;
                    for xx in 0..k {
                        s += self.data[row + xx];
                    }
                }
                out.data[y * w + x] = s * inv;
            }
        }
        out
    }

    pub fn to_u8(&self) -> Vec<u8> {
        self.data.iter().map(|v| v.round().clamp(0.0, 255.0) as u8).collect()
    }
}
