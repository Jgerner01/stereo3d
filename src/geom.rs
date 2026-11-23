//! Äußere Orientierung, Rotationen, Winkelkonventionen.
//!
//! Intern: p_cam = R · (X − C), R = Welt→Kamera (CV-Konvention, z in Blickrichtung).
//! Ausgabe: photogrammetrische Drehwinkel ω, φ, κ (Rx·Ry·Rz, Kamera-z entgegen
//! der Blickrichtung) in gon.

use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};

pub const RHO_GON: f64 = 200.0 / std::f64::consts::PI;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Pose {
    /// Rotation Welt → Kamera (Zeilen = Kameraachsen im Weltsystem)
    pub r: Matrix3<f64>,
    /// Projektionszentrum (im reduzierten Weltsystem, siehe `Project::offset`)
    pub c: Vector3<f64>,
}

impl Default for Pose {
    fn default() -> Self {
        Self { r: Matrix3::identity(), c: Vector3::zeros() }
    }
}

impl Pose {
    pub fn to_cam(&self, x: &Vector3<f64>) -> Vector3<f64> {
        self.r * (x - self.c)
    }
    /// Blickrichtung (Kamera-z) im Weltsystem
    pub fn view_dir(&self) -> Vector3<f64> {
        self.r.row(2).transpose()
    }
    /// Inkrementelle Verbesserung: R ← exp([δ]×)·R, C ← C + dc
    pub fn update(&mut self, dtheta: &Vector3<f64>, dc: &Vector3<f64>) {
        self.r = exp_so3(dtheta) * self.r;
        self.c += dc;
        self.r = orthonormalize(&self.r);
    }
}

pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

pub fn exp_so3(w: &Vector3<f64>) -> Matrix3<f64> {
    let th = w.norm();
    let k = skew(w);
    if th < 1e-12 {
        return Matrix3::identity() + k;
    }
    Matrix3::identity() + k * (th.sin() / th) + k * k * ((1.0 - th.cos()) / (th * th))
}

pub fn orthonormalize(m: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = m.svd(true, true);
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    let mut r = u * vt;
    if r.determinant() < 0.0 {
        let mut u2 = u;
        u2.column_mut(2).neg_mut();
        r = u2 * vt;
    }
    r
}

const FLIP: Matrix3<f64> = Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0);

/// Photogrammetrische Drehmatrix R_photo = Rx(ω)·Ry(φ)·Rz(κ) (Kraus), Winkel in rad.
pub fn r_photo_from_opk(om: f64, ph: f64, ka: f64) -> Matrix3<f64> {
    let (so, co) = om.sin_cos();
    let (sp, cp) = ph.sin_cos();
    let (sk, ck) = ka.sin_cos();
    let rx = Matrix3::new(1.0, 0.0, 0.0, 0.0, co, -so, 0.0, so, co);
    let ry = Matrix3::new(cp, 0.0, sp, 0.0, 1.0, 0.0, -sp, 0.0, cp);
    let rz = Matrix3::new(ck, -sk, 0.0, sk, ck, 0.0, 0.0, 0.0, 1.0);
    rx * ry * rz
}

/// ω, φ, κ [rad] aus R_photo.
pub fn opk_from_r_photo(r: &Matrix3<f64>) -> (f64, f64, f64) {
    let ph = r[(0, 2)].clamp(-1.0, 1.0).asin();
    let om = (-r[(1, 2)]).atan2(r[(2, 2)]);
    let ka = (-r[(0, 1)]).atan2(r[(0, 0)]);
    (om, ph, ka)
}

/// CV-Rotation (Welt→Kamera) aus photogrammetrischen Winkeln [rad].
pub fn r_cv_from_opk(om: f64, ph: f64, ka: f64) -> Matrix3<f64> {
    FLIP * r_photo_from_opk(om, ph, ka).transpose()
}

/// Photogrammetrische Winkel [rad] aus CV-Rotation.
pub fn opk_from_r_cv(r: &Matrix3<f64>) -> (f64, f64, f64) {
    opk_from_r_photo(&(r.transpose() * FLIP))
}

pub fn gon(rad: f64) -> f64 {
    let mut g = rad * RHO_GON;
    if g < 0.0 {
        g += 400.0;
    }
    g
}

pub fn gon_signed(rad: f64) -> f64 {
    rad * RHO_GON
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn opk_roundtrip() {
        let (o, p, k) = (0.05, -0.12, 2.3);
        let r = r_cv_from_opk(o, p, k);
        let (o2, p2, k2) = opk_from_r_cv(&r);
        assert!((o - o2).abs() < 1e-12 && (p - p2).abs() < 1e-12 && (k - k2).abs() < 1e-12);
        // Nadiraufnahme (ω=φ=κ=0): Kamera schaut nach −Z
        let r0 = r_cv_from_opk(0.0, 0.0, 0.0);
        assert!((r0.row(2).transpose() - Vector3::new(0.0, 0.0, -1.0)).norm() < 1e-12);
    }
}
