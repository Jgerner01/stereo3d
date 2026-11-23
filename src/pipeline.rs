//! Auswerteablauf: Orientierung → dichte Zuordnung → DOM/DGM, sowie Einzelpunktmessung.

use crate::adjust::{self, BundleOptions, Obs, PointPrior};
use crate::camera::Camera;
use crate::geom::{gon, gon_signed, opk_from_r_cv, Pose};
use crate::imgutil::Gray;
use crate::lsm::{harris_grid, lsm, ncc};
use crate::project::{MeasuredPoint, Project, StoredOrientation};
use crate::raster::Grid;
use crate::rectify::Rectification;
use crate::sgm::{match_pair, Disparity, SgmParams};
use anyhow::{bail, Result};
use nalgebra::{DMatrix, Matrix3, Vector2, Vector3};
use rayon::prelude::*;
use std::fmt::Write as _;
use std::io::Write as _;

pub type Log<'a> = &'a (dyn Fn(String) + Sync);

#[derive(Clone, Debug)]
pub struct Oriented {
    pub offset: Vector3<f64>,
    pub poses: [Pose; 2],
    pub cov: DMatrix<f64>,
    pub sigma0: f64,
    pub support: Vec<Vector3<f64>>,
}

impl Oriented {
    pub fn to_stored(&self) -> StoredOrientation {
        let rr = |r: &Matrix3<f64>| {
            let mut a = [0.0; 9];
            for i in 0..3 {
                for j in 0..3 {
                    a[3 * i + j] = r[(i, j)];
                }
            }
            a
        };
        StoredOrientation {
            offset: self.offset.into(),
            c: [self.poses[0].c.into(), self.poses[1].c.into()],
            r: [rr(&self.poses[0].r), rr(&self.poses[1].r)],
            cov: self.cov.transpose().as_slice().to_vec(),
            sigma0: self.sigma0,
            support_points: self.support.iter().map(|p| (*p).into()).collect(),
        }
    }
    pub fn from_stored(s: &StoredOrientation) -> Oriented {
        let pose = |k: usize| Pose { r: Matrix3::from_row_slice(&s.r[k]), c: Vector3::from(s.c[k]) };
        let n = (s.cov.len() as f64).sqrt() as usize;
        Oriented {
            offset: Vector3::from(s.offset),
            poses: [pose(0), pose(1)],
            cov: DMatrix::from_row_slice(n, n, &s.cov),
            sigma0: s.sigma0,
            support: s.support_points.iter().map(|p| Vector3::from(*p)).collect(),
        }
    }
}

pub fn load_images(p: &Project) -> Result<[Gray; 2]> {
    let a = Gray::load(&p.resolve(&p.images[0].path))?;
    let b = Gray::load(&p.resolve(&p.images[1].path))?;
    for (k, g) in [&a, &b].iter().enumerate() {
        let c = &p.images[k].camera;
        if c.width as usize != g.w || c.height as usize != g.h {
            bail!(
                "Bild {}: Größe {}×{} passt nicht zur Kamera ({}×{})",
                k + 1,
                g.w,
                g.h,
                c.width,
                c.height
            );
        }
    }
    Ok([a, b])
}

fn cams(p: &Project) -> [Camera; 2] {
    [p.images[0].camera.clone(), p.images[1].camera.clone()]
}

/// Ausschnitt eines Normalbilds direkt aus dem Originalbild resampeln.
pub fn rect_patch(rect: &Rectification, side: usize, cam: &Camera, pose: &Pose, img: &Gray, u0: isize, v0: isize, w: usize, h: usize) -> Gray {
    let mut g = Gray::new(w, h);
    for y in 0..h {
        for x in 0..w {
            if let Some(p) = rect.rect_to_orig(side, cam, pose, (u0 + x as isize) as f64, (v0 + y as isize) as f64) {
                g.data[y * w + x] = img.sample(p.x, p.y).unwrap_or(0.0);
            }
        }
    }
    g
}

/// Disparitätsbereich d' aus Stützpunkten (mit Zuschlag).
fn disp_range(rect: &Rectification, pts: &[Vector3<f64>], margin: f64) -> Option<(f64, f64)> {
    let ds: Vec<f64> = pts.iter().filter_map(|x| rect.world_to_rect(x).map(|r| r.2)).collect();
    if ds.is_empty() {
        return None;
    }
    let lo = ds.iter().cloned().fold(f64::MAX, f64::min);
    let hi = ds.iter().cloned().fold(f64::MIN, f64::max);
    let off = rect.disp_offset();
    // Zuschlag: Anteil der Spannweite + 6 % der mittleren Tiefe (Bauwerke/Vegetation über Stützpunkten)
    let extra = margin * (hi - lo) + 0.06 * (0.5 * (hi + lo) - off).abs() + 3.0;
    Some((lo - extra, hi + extra))
}

/// Subpixelgenaue Zuordnung im Normalbild-Maßstab 1 per LSM.
/// `from`: Seite des Templates; (ua, va) Template-Mitte; (ub, vb) Startwert in der anderen Seite.
#[allow(clippy::too_many_arguments)]
fn refine_lsm(
    rect: &Rectification,
    cams: &[Camera; 2],
    poses: &[Pose; 2],
    imgs: &[Gray; 2],
    from: usize,
    ua: f64,
    va: f64,
    ub: f64,
    vb: f64,
    half: usize,
) -> Option<crate::lsm::LsmResult> {
    let other = 1 - from;
    let ia = rect.idx[from];
    let ib = rect.idx[other];
    let h = half as isize;
    let m = h + 3;
    let (pa_u, pa_v) = (ua.round() as isize - m, va.round() as isize - m);
    let pa = rect_patch(rect, from, &cams[ia], &poses[ia], &imgs[ia], pa_u, pa_v, (2 * m + 1) as usize, (2 * m + 1) as usize);
    let mb = m + 6;
    let (pb_u, pb_v) = (ub.round() as isize - mb, vb.round() as isize - mb);
    let pb = rect_patch(rect, other, &cams[ib], &poses[ib], &imgs[ib], pb_u, pb_v, (2 * mb + 1) as usize, (2 * mb + 1) as usize);
    let r = lsm(&pa, ua - pa_u as f64, va - pa_v as f64, &pb, ub - pb_u as f64, vb - pb_v as f64, h)?;
    Some(crate::lsm::LsmResult { x: r.x + pb_u as f64, y: r.y + pb_v as f64, ..r })
}

// ---------------------------------------------------------------------------
// Orientierung
// ---------------------------------------------------------------------------

pub fn orient(p: &Project, imgs: &[Gray; 2], log: Log) -> Result<(Oriented, String)> {
    let cams = cams(p);
    let ctrl: Vec<_> = p.gcps.iter().filter(|g| !g.check).collect();
    if ctrl.len() < 3 {
        bail!("Mindestens 3 Passpunkte (besser ≥ 5) erforderlich");
    }
    let offset = ctrl.iter().fold(Vector3::zeros(), |a, g| a + Vector3::new(g.x, g.y, g.z)) / ctrl.len() as f64;
    let red = |g: &crate::project::Gcp| Vector3::new(g.x, g.y, g.z) - offset;

    // 1. Rückwärtsschnitte
    let mut poses: Vec<Pose> = Vec::new();
    for k in 0..2 {
        let mut xn = Vec::new();
        let mut xw = Vec::new();
        for g in &ctrl {
            if let Some(m) = p.mark(&g.id, k) {
                xn.push(cams[k].pixel_to_norm(m.u, m.v));
                xw.push(red(g));
            }
        }
        let pose = adjust::resection_init(&xn, &xw)?;
        let (_, rms) = adjust::resection_refine(pose.clone(), &xn, &xw)?;
        log(format!("Rückwärtsschnitt Bild {}: {} Passpunkte, RMS {:.2} px", k + 1, xn.len(), rms * cams[k].f_mean()));
        poses.push(pose);
    }
    let mut poses2 = [poses[0].clone(), poses[1].clone()];

    // 2. Verknüpfungspunkte
    let ctrl_pts: Vec<Vector3<f64>> = ctrl.iter().map(|g| red(g)).collect();
    let ties = tie_points(p, &cams, &poses2, imgs, &ctrl_pts, log)?;
    log(format!("Verknüpfungspunkte: {}", ties.len()));

    // 3. Bündelausgleich
    let mut points = Vec::new();
    let mut priors = Vec::new();
    let mut obs = Vec::new();
    for g in &ctrl {
        let marks: Vec<_> = (0..2).filter_map(|k| p.mark(&g.id, k)).collect();
        if marks.is_empty() {
            continue;
        }
        let j = points.len();
        points.push(red(g));
        priors.push(Some(PointPrior { xyz: red(g), sigma: Vector3::new(g.sxy, g.sxy, g.sz) }));
        for m in marks {
            obs.push(Obs { img: m.image, pt: j, xn: cams[m.image].pixel_to_norm(m.u, m.v), sigma: p.settings.sigma_px / cams[m.image].f_mean() });
        }
    }
    let n_ctrl = points.len();
    for t in &ties {
        let xa = cams[0].pixel_to_norm(t[0].x, t[0].y);
        let xb = cams[1].pixel_to_norm(t[1].x, t[1].y);
        let Some(x) = adjust::midpoint(&[&poses2[0], &poses2[1]], &[xa, xb]) else { continue };
        let j = points.len();
        points.push(x);
        priors.push(None);
        obs.push(Obs { img: 0, pt: j, xn: xa, sigma: p.settings.tie_sigma_px / cams[0].f_mean() });
        obs.push(Obs { img: 1, pt: j, xn: xb, sigma: p.settings.tie_sigma_px / cams[1].f_mean() });
    }
    let res = adjust::bundle_adjust(&poses2, &points, &priors, &obs, &BundleOptions::default())?;
    poses2 = [res.poses[0].clone(), res.poses[1].clone()];
    let n_rej = res.rejected.iter().filter(|r| **r).count();
    log(format!("Bündelausgleich: σ0 = {:.3}, Redundanz {}, verworfen {}", res.sigma0, res.redundancy, n_rej));

    // 4. Protokoll
    let mut rep = String::new();
    let _ = writeln!(rep, "ORIENTIERUNG – Bündelausgleich");
    let _ = writeln!(rep, "Passpunkte: {}, Verknüpfungspunkte: {} (verworfene Beobachtungen: {})", n_ctrl, points.len() - n_ctrl, n_rej);
    let _ = writeln!(rep, "σ0 a posteriori = {:.3}  (a priori 1; σ_Bild Passp. {:.2} px, Verkn. {:.2} px)", res.sigma0, p.settings.sigma_px, p.settings.tie_sigma_px);
    let _ = writeln!(rep, "Redundanz r = {}, Iterationen {}\n", res.redundancy, res.iterations);
    let _ = writeln!(rep, "Äußere Orientierung (ω, φ, κ photogrammetrisch, gon):");
    let _ = writeln!(rep, "{:>4} {:>14} {:>14} {:>11} {:>10} {:>10} {:>10} | σX0[m] σY0[m] σZ0[m] σω[mgon] σφ[mgon] σκ[mgon]", "Bild", "X0", "Y0", "Z0", "ω", "φ", "κ");
    for k in 0..2 {
        let ps = &poses2[k];
        let (om, ph, ka) = opk_from_r_cv(&ps.r);
        let cc = ps.c + offset;
        let s = opk_sigma(ps, &res.pose_cov, k);
        let _ = writeln!(
            rep,
            "{:>4} {:>14.3} {:>14.3} {:>11.3} {:>10.4} {:>10.4} {:>10.4} | {:.3}  {:.3}  {:.3}  {:.2}  {:.2}  {:.2}",
            k + 1,
            cc.x,
            cc.y,
            cc.z,
            gon_signed(om),
            gon_signed(ph),
            gon(ka),
            s[3],
            s[4],
            s[5],
            s[0] * 1000.0,
            s[1] * 1000.0,
            s[2] * 1000.0
        );
    }
    let _ = writeln!(rep, "\nPasspunkte – Verbesserungen Objekt [m] und Bild [px]:");
    let _ = writeln!(rep, "{:>8} {:>8} {:>8} {:>8} | {:>7} {:>7} {:>7} {:>7}", "Id", "vX", "vY", "vZ", "vx1", "vy1", "vx2", "vy2");
    let mut j = 0;
    for g in &ctrl {
        if !(0..2).any(|k| p.mark(&g.id, k).is_some()) {
            continue;
        }
        let v = res.points[j] - red(g);
        let mut img_res = [f64::NAN; 4];
        for (i, o) in obs.iter().enumerate() {
            if o.pt == j {
                let f = cams[o.img].f_mean();
                img_res[2 * o.img] = res.residuals[i].x * f;
                img_res[2 * o.img + 1] = res.residuals[i].y * f;
            }
        }
        let _ = writeln!(rep, "{:>8} {:>8.3} {:>8.3} {:>8.3} | {:>7.2} {:>7.2} {:>7.2} {:>7.2}", g.id, v.x, v.y, v.z, img_res[0], img_res[1], img_res[2], img_res[3]);
        j += 1;
    }
    // Restparallaxen Verknüpfungspunkte
    let mut s2 = 0.0;
    let mut n2 = 0;
    for (i, o) in obs.iter().enumerate() {
        if o.pt >= n_ctrl && !res.rejected[i] {
            s2 += (res.residuals[i] * cams[o.img].f_mean()).norm_squared();
            n2 += 1;
        }
    }
    if n2 > 0 {
        let _ = writeln!(rep, "\nRMS Bildverbesserungen Verknüpfungspunkte: {:.3} px", (s2 / n2 as f64).sqrt());
    }
    let ori = Oriented {
        offset,
        poses: poses2.clone(),
        cov: res.pose_cov.clone(),
        sigma0: res.sigma0,
        support: res.points.clone(),
    };
    // Kontrollpunkte
    let checks: Vec<_> = p.gcps.iter().filter(|g| g.check).collect();
    if !checks.is_empty() {
        let _ = writeln!(rep, "\nKontrollpunkte (Vorwärtsschnitt − Sollkoordinate) [m]:");
        let _ = writeln!(rep, "{:>8} {:>8} {:>8} {:>8} | {:>7} {:>7} {:>7}", "Id", "dX", "dY", "dZ", "σX", "σY", "σZ");
        let (mut sx, mut sy, mut sz, mut n) = (0.0, 0.0, 0.0, 0);
        for g in checks {
            let (Some(a), Some(b)) = (p.mark(&g.id, 0), p.mark(&g.id, 1)) else { continue };
            let Some(it) = intersect_px(&cams, &ori, [a.u, a.v], [b.u, b.v], [p.settings.sigma_px; 2]) else { continue };
            let d = it.x - red(g);
            let _ = writeln!(
                rep,
                "{:>8} {:>8.3} {:>8.3} {:>8.3} | {:>7.3} {:>7.3} {:>7.3}",
                g.id,
                d.x,
                d.y,
                d.z,
                it.cov[(0, 0)].sqrt(),
                it.cov[(1, 1)].sqrt(),
                it.cov[(2, 2)].sqrt()
            );
            sx += d.x * d.x;
            sy += d.y * d.y;
            sz += d.z * d.z;
            n += 1;
        }
        if n > 0 {
            let nf = n as f64;
            let _ = writeln!(rep, "{:>8} {:>8.3} {:>8.3} {:>8.3}", "RMS", (sx / nf).sqrt(), (sy / nf).sqrt(), (sz / nf).sqrt());
        }
    }
    Ok((ori, rep))
}

/// σ der Orientierungsparameter: [ω, φ, κ (rad), X0, Y0, Z0 (m)]
fn opk_sigma(pose: &Pose, cov: &DMatrix<f64>, k: usize) -> [f64; 6] {
    // Jacobi-Matrix d(ω,φ,κ)/dδθ numerisch
    let base = opk_from_r_cv(&pose.r);
    let mut j = Matrix3::zeros();
    let h = 1e-7;
    for c in 0..3 {
        let mut d = Vector3::zeros();
        d[c] = h;
        let mut p2 = pose.clone();
        p2.update(&d, &Vector3::zeros());
        let o = opk_from_r_cv(&p2.r);
        let wrap = |a: f64| (a + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI;
        j[(0, c)] = wrap(o.0 - base.0) / h;
        j[(1, c)] = wrap(o.1 - base.1) / h;
        j[(2, c)] = wrap(o.2 - base.2) / h;
    }
    let crot = Matrix3::from_fn(|r, c| cov[(6 * k + r, 6 * k + c)]);
    let co = j * crot * j.transpose();
    [
        co[(0, 0)].sqrt(),
        co[(1, 1)].sqrt(),
        co[(2, 2)].sqrt(),
        cov[(6 * k + 3, 6 * k + 3)].sqrt(),
        cov[(6 * k + 4, 6 * k + 4)].sqrt(),
        cov[(6 * k + 5, 6 * k + 5)].sqrt(),
    ]
}

/// Vorwärtsschnitt aus Originalpixeln beider Bilder (Ergebnis reduziert).
pub fn intersect_px(cams: &[Camera; 2], o: &Oriented, a: [f64; 2], b: [f64; 2], sigma_px: [f64; 2]) -> Option<adjust::Intersection> {
    let xa = cams[0].pixel_to_norm(a[0], a[1]);
    let xb = cams[1].pixel_to_norm(b[0], b[1]);
    adjust::intersect(
        &[&o.poses[0], &o.poses[1]],
        &[0, 1],
        &[xa, xb],
        &[sigma_px[0] / cams[0].f_mean(), sigma_px[1] / cams[1].f_mean()],
        Some(&o.cov),
    )
}

/// Verknüpfungspunkte: Harris im verkleinerten Normalbild, NCC entlang der Epipolarlinie,
/// Rückwärtsprüfung, LSM-Verfeinerung im vollen Maßstab. Rückgabe: Originalpixel [Bild 1, Bild 2].
fn tie_points(
    p: &Project,
    cams: &[Camera; 2],
    poses: &[Pose; 2],
    imgs: &[Gray; 2],
    support: &[Vector3<f64>],
    log: Log,
) -> Result<Vec<[Vector2<f64>; 2]>> {
    let maxdim = cams[0].width.max(cams[0].height) as f64;
    let s = (1600.0 / maxdim).min(1.0);
    let rt = Rectification::compute(cams, poses, s);
    let r1 = Rectification::compute(cams, poses, 1.0);
    let (il, ir) = (rt.idx[0], rt.idx[1]);
    let left = rt.resample(0, &cams[il], &poses[il], &imgs[il]);
    let right = rt.resample(1, &cams[ir], &poses[ir], &imgs[ir]);
    let mask = rt.valid_mask(0, &cams[il], &poses[il]);
    let Some((dmin, dmax)) = disp_range(&rt, support, p.settings.depth_margin.max(0.5)) else {
        bail!("Tiefenbereich nicht bestimmbar");
    };
    log(format!("Verknüpfung: Normalbilder {}×{} px (1:{:.1}), d' = {:.0} … {:.0}", left.w, left.h, 1.0 / s, dmin, dmax));
    let corners = harris_grid(&left, p.settings.tie_grid[0], p.settings.tie_grid[1], 10, Some(&mask));
    let half = 6isize;
    let vy = 3isize;
    let (dmin_i, dmax_i) = (dmin.floor() as isize, dmax.ceil() as isize);
    let found: Vec<[Vector2<f64>; 2]> = corners
        .par_iter()
        .filter_map(|&(x, y, _)| {
            let (x, y) = (x as isize, y as isize);
            let mut best = (-2f32, 0isize, 0isize);
            let mut cands = Vec::new();
            for dv in -vy..=vy {
                for d in dmin_i..=dmax_i {
                    let xr = x - d;
                    if let Some(c) = ncc(&left, x, y, &right, xr as f64, (y + dv) as f64, half) {
                        cands.push((c, d, dv));
                        if c > best.0 {
                            best = (c, d, dv);
                        }
                    }
                }
            }
            if best.0 < 0.75 {
                return None;
            }
            let second = cands
                .iter()
                .filter(|c| (c.1 - best.1).abs() > 2 || (c.2 - best.2).abs() > 2)
                .map(|c| c.0)
                .fold(-1f32, f32::max);
            if second > best.0 - 0.08 {
                return None;
            }
            // Rückwärtsprüfung
            let (xr, yr) = (x - best.1, y + best.2);
            let mut back = (-2f32, 0isize);
            for d in dmin_i..=dmax_i {
                if let Some(c) = ncc(&right, xr, yr, &left, (xr + d) as f64, y as f64, half) {
                    if c > back.0 {
                        back = (c, d);
                    }
                }
            }
            if (back.1 - best.1).abs() > 1 {
                return None;
            }
            // Übergang auf Maßstab 1 und LSM
            let ol = rt.rect_to_orig(0, &cams[il], &poses[il], x as f64, y as f64)?;
            let or = rt.rect_to_orig(1, &cams[ir], &poses[ir], xr as f64, yr as f64)?;
            let a = r1.orig_to_rect(0, &cams[il], &poses[il], ol.x, ol.y)?;
            let b = r1.orig_to_rect(1, &cams[ir], &poses[ir], or.x, or.y)?;
            let lr = refine_lsm(&r1, cams, poses, imgs, 0, a.x, a.y, b.x, b.y, 8)?;
            if lr.rho < 0.8 || lr.cov[(0, 0)].sqrt() > 0.5 || (lr.x - b.x).abs() > 4.0 / s || (lr.y - b.y).abs() > 4.0 / s {
                return None;
            }
            let pl = r1.rect_to_orig(0, &cams[il], &poses[il], a.x, a.y)?;
            let pr = r1.rect_to_orig(1, &cams[ir], &poses[ir], lr.x, lr.y)?;
            let mut out = [Vector2::zeros(); 2];
            out[il] = pl;
            out[ir] = pr;
            Some(out)
        })
        .collect();
    Ok(found)
}

// ---------------------------------------------------------------------------
// Dichte Zuordnung und Produkte
// ---------------------------------------------------------------------------

pub struct DenseResult {
    /// absolute Koordinaten
    pub pts: Vec<[f64; 3]>,
    pub gray: Vec<u8>,
    /// Bodenpixelgröße der dichten Zuordnung [m]
    pub gsd: f64,
    pub disp: Disparity,
    pub left_rect: Gray,
    pub right_rect: Gray,
    pub rect: Rectification,
}

pub fn dense(p: &Project, o: &Oriented, imgs: &[Gray; 2], log: Log) -> Result<DenseResult> {
    let cams = cams(p);
    let s = p.settings.dense_scale.clamp(0.05, 1.0);
    let rect = Rectification::compute(&cams, &o.poses, s);
    let (il, ir) = (rect.idx[0], rect.idx[1]);
    log(format!("Normalbilder {}×{} / {}×{} px (Maßstab {:.2})", rect.size[0].0, rect.size[0].1, rect.size[1].0, rect.size[1].1, s));
    let left = rect.resample(0, &cams[il], &o.poses[il], &imgs[il]);
    let right = rect.resample(1, &cams[ir], &o.poses[ir], &imgs[ir]);
    let mask = rect.valid_mask(0, &cams[il], &o.poses[il]);
    let Some((dmin, dmax)) = disp_range(&rect, &o.support, p.settings.depth_margin) else {
        bail!("Tiefenbereich nicht bestimmbar");
    };
    let sp = SgmParams { p1: p.settings.sgm_p1, p2: p.settings.sgm_p2, uniqueness: p.settings.sgm_uniqueness, speckle_size: ((150.0 * s * s) as usize).max(30), ..Default::default() };
    let disp = match_pair(&left, &right, Some(&mask), dmin, dmax, &sp, log);
    let mut pts = Vec::new();
    let mut gray = Vec::new();
    let mut zsum = 0.0;
    for v in 0..disp.h {
        for u in 0..disp.w {
            let d = disp.d[v * disp.w + u];
            if !d.is_finite() {
                continue;
            }
            if let Some((x, z)) = rect.triangulate(u as f64, v as f64, d as f64) {
                let a = x + o.offset;
                pts.push([a.x, a.y, a.z]);
                gray.push(left.get(u, v).clamp(0.0, 255.0) as u8);
                zsum += z;
            }
        }
    }
    let valid = pts.len() as f64 / mask.iter().filter(|m| **m).count().max(1) as f64;
    let gsd = if pts.is_empty() { 0.0 } else { zsum / pts.len() as f64 / rect.f };
    log(format!("Dichte Zuordnung: {} Punkte ({:.0} % des Überlappungsbereichs), Bodenpixel {:.3} m", pts.len(), 100.0 * valid, gsd));
    Ok(DenseResult { pts, gray, gsd, disp, left_rect: left, right_rect: right, rect })
}

/// DOM, DGM und Exporte schreiben. Rückgabe: Protokolltext.
pub fn products(p: &Project, d: &DenseResult, log: Log) -> Result<(Grid, Grid, String)> {
    let out = p.out_dir();
    std::fs::create_dir_all(&out)?;
    let cell = if p.settings.cell > 0.0 { p.settings.cell } else { nice_cell(2.0 * d.gsd) };
    let Some(mut dom) = Grid::from_points(&d.pts, cell) else { bail!("Keine Punkte für Raster") };
    dom.fill_gaps(p.settings.fill_cells);
    let dgm = dom.ground_filter(p.settings.dgm_window, p.settings.dgm_slope, p.settings.dgm_dh0, p.settings.dgm_dhmax);
    dom.write_ascii_grid(&out.join("dom.asc"))?;
    dgm.write_ascii_grid(&out.join("dgm.asc"))?;
    dgm.write_xyz(&out.join("dgm.xyz"))?;
    d.left_rect.save_png(&out.join("normalbild_links.png"))?;
    d.right_rect.save_png(&out.join("normalbild_rechts.png"))?;
    save_disp_png(&d.disp, &out.join("disparitaet.png"))?;
    if p.settings.export_cloud {
        let mut f = std::io::BufWriter::new(std::fs::File::create(out.join("punktwolke.xyz"))?);
        for (q, g) in d.pts.iter().zip(&d.gray) {
            writeln!(f, "{:.3} {:.3} {:.3} {}", q[0], q[1], q[2], g)?;
        }
    }
    let nd = dom.z.iter().filter(|v| v.is_finite()).count();
    let ng = dgm.z.iter().filter(|v| v.is_finite()).count();
    let rep = format!(
        "PRODUKTE\nRasterweite {cell} m, DOM {}×{} Zellen ({} belegt), DGM {} belegt\nσZ (theor., 0,25 px Disparität): {:.3} m\nDateien: {}",
        dom.nx,
        dom.ny,
        nd,
        ng,
        0.25 * d.gsd * mean_depth_ratio(d),
        out.display()
    );
    log(format!("Raster geschrieben: {}", out.display()));
    Ok((dom, dgm, rep))
}

/// Z/B (Höhen-Basis-Verhältnis) der dichten Zuordnung
fn mean_depth_ratio(d: &DenseResult) -> f64 {
    let dd: f64 = d.disp.d.iter().filter(|v| v.is_finite()).map(|&v| v as f64 - d.rect.disp_offset()).sum::<f64>();
    let n = d.disp.d.iter().filter(|v| v.is_finite()).count().max(1) as f64;
    let mean_disp = dd / n;
    d.rect.f / mean_disp.max(1e-9)
}

fn nice_cell(c: f64) -> f64 {
    let steps = [0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0];
    *steps.iter().find(|&&s| s >= c * 0.999).unwrap_or(&10.0)
}

pub fn save_disp_png(d: &Disparity, path: &std::path::Path) -> Result<()> {
    let v: Vec<f32> = d.d.iter().cloned().filter(|x| x.is_finite()).collect();
    let (lo, hi) = v.iter().fold((f32::MAX, f32::MIN), |a, &b| (a.0.min(b), a.1.max(b)));
    let g = Gray {
        w: d.w,
        h: d.h,
        data: d.d.iter().map(|&x| if x.is_finite() { 20.0 + 235.0 * (x - lo) / (hi - lo).max(1e-6) } else { 0.0 }).collect(),
    };
    g.save_png(path)
}

// ---------------------------------------------------------------------------
// Einzelpunktmessung
// ---------------------------------------------------------------------------

/// Ergebnis einer Einzelpunktmessung (Koordinaten absolut).
#[derive(Clone, Debug)]
pub struct PointResult {
    pub px: [Vector2<f64>; 2],
    pub xyz: Vector3<f64>,
    pub sigma: Vector3<f64>,
    pub sigma_meas: Vector3<f64>,
    pub rho: f64,
    pub sigma_match_px: f64,
}

/// Punkt in Bild `a` (Originalpixel) messen; Gegenpunkt automatisch (oder ab Startwert `guess`).
pub fn measure_point(p: &Project, o: &Oriented, imgs: &[Gray; 2], a: usize, ua: f64, va: f64, guess: Option<Vector2<f64>>) -> Result<PointResult> {
    let cams = cams(p);
    let b = 1 - a;
    let rect = Rectification::compute(&cams, &o.poses, 1.0);
    let side_a = if rect.idx[0] == a { 0 } else { 1 };
    let side_b = 1 - side_a;
    let Some(ra) = rect.orig_to_rect(side_a, &cams[a], &o.poses[a], ua, va) else { bail!("Punkt nicht abbildbar") };
    let half = p.settings.lsm_half.max(4);
    let (ub0, vb0) = if let Some(g) = guess {
        let Some(rb) = rect.orig_to_rect(side_b, &cams[b], &o.poses[b], g.x, g.y) else { bail!("Startwert nicht abbildbar") };
        (rb.x, rb.y)
    } else {
        let Some((dmin, dmax)) = disp_range(&rect, &o.support, p.settings.depth_margin.max(0.5)) else { bail!("Tiefenbereich unbekannt") };
        // Streifen im anderen Normalbild: u_b = u_a ∓ d
        let (u_lo, u_hi) = if side_a == 0 { (ra.x - dmax, ra.x - dmin) } else { (ra.x + dmin, ra.x + dmax) };
        if u_hi - u_lo > 30000.0 {
            bail!("Suchbereich zu groß – Startwert im zweiten Bild anklicken");
        }
        let h = half as isize;
        let vy = 2isize;
        let su0 = u_lo.floor() as isize - h - 1;
        let sv0 = ra.y.round() as isize - h - vy;
        let sw = (u_hi - u_lo).ceil() as usize + 2 * half + 3;
        let sh = 2 * half + 1 + 2 * vy as usize;
        let strip = rect_patch(&rect, side_b, &cams[b], &o.poses[b], &imgs[b], su0, sv0, sw, sh);
        let ta0 = (ra.x.round() as isize - h, ra.y.round() as isize - h);
        let tpl = rect_patch(&rect, side_a, &cams[a], &o.poses[a], &imgs[a], ta0.0, ta0.1, 2 * half + 1, 2 * half + 1);
        let mut best = (-2f32, 0isize, 0isize);
        let mut all = Vec::new();
        for dv in 0..=(2 * vy) {
            for x in h..(sw as isize - h) {
                if let Some(c) = ncc(&tpl, h, h, &strip, x as f64, (h + dv) as f64, h) {
                    all.push((c, x));
                    if c > best.0 {
                        best = (c, x, dv);
                    }
                }
            }
        }
        if best.0 < 0.5 {
            bail!("Keine Zuordnung gefunden (NCC {:.2}) – Gegenpunkt manuell setzen", best.0);
        }
        let second = all.iter().filter(|c| (c.1 - best.1).abs() > 3).map(|c| c.0).fold(-1f32, f32::max);
        if second > best.0 - 0.03 {
            bail!("Mehrdeutige Zuordnung (NCC {:.2} / {:.2}) – Gegenpunkt manuell setzen", best.0, second);
        }
        // ganzzahlige Rundung von ra berücksichtigen
        let off_u = ra.x - ra.x.round();
        let off_v = ra.y - ra.y.round();
        ((su0 + best.1) as f64 + off_u, (sv0 + h + best.2) as f64 + off_v)
    };
    let r = refine_lsm(&rect, &cams, &o.poses, imgs, side_a, ra.x, ra.y, ub0, vb0, half)
        .ok_or_else(|| anyhow::anyhow!("LSM nicht konvergiert – Gegenpunkt manuell setzen"))?;
    let Some(pb) = rect.rect_to_orig(side_b, &cams[b], &o.poses[b], r.x, r.y) else { bail!("Rückabbildung fehlgeschlagen") };
    let sig_match = (0.5 * (r.cov[(0, 0)] + r.cov[(1, 1)])).sqrt().max(0.03);
    let mut px = [Vector2::zeros(); 2];
    px[a] = Vector2::new(ua, va);
    px[b] = pb;
    let mut sp = [0.1; 2];
    sp[b] = (sig_match * sig_match + 0.05f64.powi(2)).sqrt();
    let Some(it) = intersect_px(&cams, o, [px[0].x, px[0].y], [px[1].x, px[1].y], sp) else { bail!("Vorwärtsschnitt fehlgeschlagen") };
    Ok(PointResult {
        px,
        xyz: it.x + o.offset,
        sigma: Vector3::new(it.cov[(0, 0)].sqrt(), it.cov[(1, 1)].sqrt(), it.cov[(2, 2)].sqrt()),
        sigma_meas: Vector3::new(it.cov_meas[(0, 0)].sqrt(), it.cov_meas[(1, 1)].sqrt(), it.cov_meas[(2, 2)].sqrt()),
        rho: r.rho,
        sigma_match_px: sig_match,
    })
}

pub fn to_measured(id: &str, r: &PointResult) -> MeasuredPoint {
    MeasuredPoint {
        id: id.into(),
        ul: r.px[0].x,
        vl: r.px[0].y,
        ur: r.px[1].x,
        vr: r.px[1].y,
        x: r.xyz.x,
        y: r.xyz.y,
        z: r.xyz.z,
        sx: r.sigma.x,
        sy: r.sigma.y,
        sz: r.sigma.z,
        rho: r.rho,
    }
}

pub fn export_points(p: &Project, path: &std::path::Path) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "Id;X;Y;Z;sX;sY;sZ;rho;u1;v1;u2;v2")?;
    for m in &p.points {
        writeln!(
            f,
            "{};{:.3};{:.3};{:.3};{:.4};{:.4};{:.4};{:.3};{:.2};{:.2};{:.2};{:.2}",
            m.id, m.x, m.y, m.z, m.sx, m.sy, m.sz, m.rho, m.ul, m.vl, m.ur, m.vr
        )?;
    }
    Ok(())
}
