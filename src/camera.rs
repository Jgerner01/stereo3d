//! Innere Orientierung: Lochkamera + Brown-Verzeichnung (OpenCV-Konvention).
//!
//! Bildkoordinaten in Pixel, Ursprung Mitte des linken oberen Pixels,
//! u nach rechts, v nach unten. Kamerakoordinaten: x rechts, y unten,
//! z in Blickrichtung (Computer-Vision-Konvention). Die Umrechnung in die
//! photogrammetrische Konvention (ω, φ, κ) erfolgt in `geom`.

use nalgebra::{Matrix2, Vector2, Vector3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Camera {
    /// Bildgröße in Pixel
    pub width: u32,
    pub height: u32,
    /// Kamerakonstante in Pixel (x- und y-Richtung)
    pub fx: f64,
    pub fy: f64,
    /// Bildhauptpunkt in Pixel (absolut, nicht relativ zur Bildmitte)
    pub cx: f64,
    pub cy: f64,
    /// radial-symmetrische Verzeichnung
    #[serde(default)]
    pub k1: f64,
    #[serde(default)]
    pub k2: f64,
    #[serde(default)]
    pub k3: f64,
    /// radial-asymmetrische/tangentiale Verzeichnung
    #[serde(default)]
    pub p1: f64,
    #[serde(default)]
    pub p2: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            width: 6000,
            height: 4000,
            fx: 5000.0,
            fy: 5000.0,
            cx: 2999.5,
            cy: 1999.5,
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }
}

impl Camera {
    /// Aus Kamerakonstante [mm], Pixelgröße [µm] und Hauptpunktlage relativ zur
    /// Bildmitte [mm] (x rechts, y oben – photogrammetrische Konvention).
    pub fn from_metric(width: u32, height: u32, c_mm: f64, pixel_um: f64, x0_mm: f64, y0_mm: f64) -> Self {
        let f = c_mm / (pixel_um * 1e-3);
        Self {
            width,
            height,
            fx: f,
            fy: f,
            cx: (width as f64 - 1.0) / 2.0 + x0_mm / (pixel_um * 1e-3),
            cy: (height as f64 - 1.0) / 2.0 - y0_mm / (pixel_um * 1e-3),
            ..Default::default()
        }
    }

    pub fn has_distortion(&self) -> bool {
        self.k1 != 0.0 || self.k2 != 0.0 || self.k3 != 0.0 || self.p1 != 0.0 || self.p2 != 0.0
    }

    /// Verzeichnung auf normierte Koordinaten anwenden.
    pub fn distort_norm(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let rad = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3));
        let xd = x * rad + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let yd = y * rad + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        (xd, yd)
    }

    /// Verzeichnung entfernen (Newton-Iteration, numerische Jacobi-Matrix).
    pub fn undistort_norm(&self, xd: f64, yd: f64) -> (f64, f64) {
        if !self.has_distortion() {
            return (xd, yd);
        }
        let (mut x, mut y) = (xd, yd);
        for _ in 0..30 {
            let (fx, fy) = self.distort_norm(x, y);
            let r = Vector2::new(fx - xd, fy - yd);
            if r.norm() < 1e-15 {
                break;
            }
            let h = 1e-7;
            let (ax, ay) = self.distort_norm(x + h, y);
            let (bx, by) = self.distort_norm(x, y + h);
            let j = Matrix2::new((ax - fx) / h, (bx - fx) / h, (ay - fy) / h, (by - fy) / h);
            match j.try_inverse() {
                Some(ji) => {
                    let d = ji * r;
                    x -= d.x;
                    y -= d.y;
                }
                None => break,
            }
        }
        (x, y)
    }

    /// Pixel → verzeichnungsfreie normierte Koordinaten (x/z, y/z).
    pub fn pixel_to_norm(&self, u: f64, v: f64) -> Vector2<f64> {
        let xd = (u - self.cx) / self.fx;
        let yd = (v - self.cy) / self.fy;
        let (x, y) = self.undistort_norm(xd, yd);
        Vector2::new(x, y)
    }

    /// Normierte Koordinaten → Pixel (mit Verzeichnung).
    pub fn norm_to_pixel(&self, x: f64, y: f64) -> Vector2<f64> {
        let (xd, yd) = self.distort_norm(x, y);
        Vector2::new(self.fx * xd + self.cx, self.fy * yd + self.cy)
    }

    /// Punkt im Kamerasystem → Pixel; None, wenn hinter der Kamera.
    pub fn project_cam(&self, p: &Vector3<f64>) -> Option<Vector2<f64>> {
        if p.z <= 1e-9 {
            return None;
        }
        Some(self.norm_to_pixel(p.x / p.z, p.y / p.z))
    }

    /// Mittlere Kamerakonstante (für Umrechnung σ_px ↔ σ_norm).
    pub fn f_mean(&self) -> f64 {
        0.5 * (self.fx + self.fy)
    }

    /// Kamera für ein um Faktor `s` skaliertes Bild (Pixelmitte-Konvention beachtet).
    pub fn scaled(&self, s: f64, width: u32, height: u32) -> Camera {
        Camera {
            width,
            height,
            fx: self.fx * s,
            fy: self.fy * s,
            cx: (self.cx + 0.5) * s - 0.5,
            cy: (self.cy + 0.5) * s - 0.5,
            ..self.clone()
        }
    }

    pub fn in_image(&self, p: &Vector2<f64>, margin: f64) -> bool {
        p.x >= margin
            && p.y >= margin
            && p.x <= self.width as f64 - 1.0 - margin
            && p.y <= self.height as f64 - 1.0 - margin
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn undistort_roundtrip() {
        let c = Camera { k1: -0.12, k2: 0.03, k3: -0.004, p1: 0.0007, p2: -0.0003, ..Default::default() };
        for &(x, y) in &[(0.3, -0.2), (-0.55, 0.38), (0.0, 0.0), (0.6, 0.4)] {
            let (xd, yd) = c.distort_norm(x, y);
            let (xu, yu) = c.undistort_norm(xd, yd);
            assert!((xu - x).abs() < 1e-12 && (yu - y).abs() < 1e-12);
        }
    }
}
