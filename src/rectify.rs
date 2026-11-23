//! Epipolarnormalisierung (Normalbildfall) nach Fusiello.
//!
//! Beide Normalbilder teilen Rotation R_n und Kamerakonstante f; nur die
//! Hauptpunkt-Abszissen cx[0], cx[1] unterscheiden sich, damit beide Bilder
//! möglichst klein bleiben. Zuordnung erfolgt mit d' = u_l − u_r;
//! Tiefe im Normalsystem: Z = f·B / (d' − (cx_l − cx_r)).

use crate::camera::Camera;
use crate::geom::Pose;
use crate::imgutil::Gray;
use nalgebra::{Matrix3, Vector2, Vector3};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Rectification {
    /// Indizes der Originalbilder: [links, rechts]
    pub idx: [usize; 2],
    pub r: Matrix3<f64>,
    pub f: f64,
    pub cx: [f64; 2],
    pub cy: f64,
    pub size: [(usize, usize); 2],
    pub baseline: f64,
    pub c_left: Vector3<f64>,
    pub scale: f64,
}

impl Rectification {
    /// `scale`: Maßstab Normalbild / Originalbild (1.0 = volle Auflösung).
    pub fn compute(cams: &[Camera; 2], poses: &[Pose; 2], scale: f64) -> Rectification {
        let mut idx = [0usize, 1];
        let b = poses[1].c - poses[0].c;
        // Bild, dessen x-Achse zur Basis zeigt, wird linkes Bild
        if b.dot(&poses[0].r.row(0).transpose()) < 0.0 {
            idx = [1, 0];
        }
        let (pl, pr) = (&poses[idx[0]], &poses[idx[1]]);
        let base = pr.c - pl.c;
        let ex = base.normalize();
        let zm = (pl.view_dir() + pr.view_dir()).normalize();
        let ey = zm.cross(&ex).normalize();
        let ez = ex.cross(&ey);
        let r = Matrix3::from_rows(&[ex.transpose(), ey.transpose(), ez.transpose()]);
        let f = scale * 0.5 * (cams[0].f_mean() + cams[1].f_mean());

        let mut rect = Rectification {
            idx,
            r,
            f,
            cx: [0.0; 2],
            cy: 0.0,
            size: [(0, 0); 2],
            baseline: base.norm(),
            c_left: pl.c,
            scale,
        };
        // Bildränder abbilden
        let mut umin = [f64::MAX; 2];
        let mut umax = [f64::MIN; 2];
        let (mut vmin, mut vmax) = (f64::MAX, f64::MIN);
        for s in 0..2 {
            let cam = &cams[idx[s]];
            let pose = &poses[idx[s]];
            let (w, h) = (cam.width as f64 - 1.0, cam.height as f64 - 1.0);
            let n = 64;
            for k in 0..=n {
                let t = k as f64 / n as f64;
                for (u, v) in [(t * w, 0.0), (t * w, h), (0.0, t * h), (w, t * h)] {
                    if let Some(p) = rect.orig_to_rect_raw(cam, pose, u, v) {
                        umin[s] = umin[s].min(p.x);
                        umax[s] = umax[s].max(p.x);
                        vmin = vmin.min(p.y);
                        vmax = vmax.max(p.y);
                    }
                }
            }
        }
        let maxdim = 2.5 * scale * cams[0].width.max(cams[0].height) as f64;
        for s in 0..2 {
            let wid = (umax[s] - umin[s]).min(maxdim);
            rect.cx[s] = -umin[s];
            rect.size[s] = (wid.ceil() as usize + 1, 0);
        }
        let hgt = (vmax - vmin).min(maxdim);
        rect.cy = -vmin;
        for s in 0..2 {
            rect.size[s].1 = hgt.ceil() as usize + 1;
        }
        rect
    }

    /// Normalbildkoordinaten ohne Hauptpunktverschiebung.
    fn orig_to_rect_raw(&self, cam: &Camera, pose: &Pose, u: f64, v: f64) -> Option<Vector2<f64>> {
        let n = cam.pixel_to_norm(u, v);
        let ray = self.r * pose.r.transpose() * Vector3::new(n.x, n.y, 1.0);
        if ray.z <= 1e-6 {
            return None;
        }
        Some(Vector2::new(self.f * ray.x / ray.z, self.f * ray.y / ray.z))
    }

    /// Originalpixel (Bild `side`: 0 = links, 1 = rechts) → Normalbildpixel.
    pub fn orig_to_rect(&self, side: usize, cam: &Camera, pose: &Pose, u: f64, v: f64) -> Option<Vector2<f64>> {
        self.orig_to_rect_raw(cam, pose, u, v).map(|p| Vector2::new(p.x + self.cx[side], p.y + self.cy))
    }

    /// Normalbildpixel → Originalpixel.
    pub fn rect_to_orig(&self, side: usize, cam: &Camera, pose: &Pose, u: f64, v: f64) -> Option<Vector2<f64>> {
        let ray_n = Vector3::new((u - self.cx[side]) / self.f, (v - self.cy) / self.f, 1.0);
        let pc = pose.r * self.r.transpose() * ray_n;
        cam.project_cam(&pc)
    }

    /// Normalbild erzeugen. `img` ist das Originalbild zu `cam`.
    pub fn resample(&self, side: usize, cam: &Camera, pose: &Pose, img: &Gray) -> Gray {
        // Vorfilterung bei Verkleinerung (Box-Filter, ganzzahlig)
        let k = (1.0 / self.scale).floor().max(1.0) as usize;
        let (src, scam) = if k >= 2 {
            let d = img.downsample(k);
            let c = cam.scaled(1.0 / k as f64, d.w as u32, d.h as u32);
            (d, c)
        } else {
            (img.clone(), cam.clone())
        };
        let (w, h) = self.size[side];
        let mut out = Gray::new(w, h);
        let m = pose.r * self.r.transpose();
        out.data.par_chunks_mut(w).enumerate().for_each(|(v, row)| {
            for (u, px) in row.iter_mut().enumerate() {
                let ray = Vector3::new((u as f64 - self.cx[side]) / self.f, (v as f64 - self.cy) / self.f, 1.0);
                let pc = m * ray;
                *px = match scam.project_cam(&pc) {
                    Some(p) => src.sample(p.x, p.y).unwrap_or(0.0),
                    None => 0.0,
                };
            }
        });
        out
    }

    /// Gültigkeitsmaske (Pixel liegt im Originalbild)
    pub fn valid_mask(&self, side: usize, cam: &Camera, pose: &Pose) -> Vec<bool> {
        let (w, h) = self.size[side];
        let m = pose.r * self.r.transpose();
        let mut out = vec![false; w * h];
        out.par_chunks_mut(w).enumerate().for_each(|(v, row)| {
            for (u, px) in row.iter_mut().enumerate() {
                let ray = Vector3::new((u as f64 - self.cx[side]) / self.f, (v as f64 - self.cy) / self.f, 1.0);
                *px = cam.project_cam(&(m * ray)).is_some_and(|p| cam.in_image(&p, 1.0));
            }
        });
        out
    }

    #[inline]
    pub fn disp_offset(&self) -> f64 {
        self.cx[0] - self.cx[1]
    }

    /// Normalbild-Tiefe → d'
    pub fn depth_to_disp(&self, z: f64) -> f64 {
        self.f * self.baseline / z + self.disp_offset()
    }

    /// Objektpunkt (reduziertes Weltsystem) → (u_l, v, d')
    pub fn world_to_rect(&self, x: &Vector3<f64>) -> Option<(f64, f64, f64)> {
        let p = self.r * (x - self.c_left);
        if p.z <= 0.0 {
            return None;
        }
        let u = self.f * p.x / p.z + self.cx[0];
        let v = self.f * p.y / p.z + self.cy;
        Some((u, v, self.depth_to_disp(p.z)))
    }

    /// Normalbild-Messung → Objektpunkt (reduziertes Weltsystem). Gibt außerdem Z_n zurück.
    pub fn triangulate(&self, u_l: f64, v: f64, d: f64) -> Option<(Vector3<f64>, f64)> {
        let dd = d - self.disp_offset();
        if dd <= 1e-6 {
            return None;
        }
        let z = self.f * self.baseline / dd;
        let p = Vector3::new((u_l - self.cx[0]) * z / self.f, (v - self.cy) * z / self.f, z);
        Some((self.c_left + self.r.transpose() * p, z))
    }
}
