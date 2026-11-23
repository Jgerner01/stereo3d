//! Ausgleichung: Rückwärtsschnitt, Bündelausgleich, Vorwärtsschnitt mit
//! Kovarianzfortpflanzung. Beobachtungen werden als verzeichnungsfreie
//! normierte Bildkoordinaten (x/z, y/z) eingeführt; σ entsprechend σ_px / f.

use crate::geom::{orthonormalize, skew, Pose};
use anyhow::{bail, Result};
use nalgebra::{DMatrix, DVector, Matrix2x3, Matrix3, Matrix6, SMatrix, Vector2, Vector3};

#[derive(Clone, Debug)]
pub struct Obs {
    pub img: usize,
    pub pt: usize,
    /// normierte Bildkoordinaten (verzeichnungsfrei)
    pub xn: Vector2<f64>,
    /// a-priori-σ in normierten Einheiten
    pub sigma: f64,
}

#[derive(Clone, Debug)]
pub struct PointPrior {
    pub xyz: Vector3<f64>,
    pub sigma: Vector3<f64>,
}

#[derive(Clone, Debug)]
pub struct BundleResult {
    pub poses: Vec<Pose>,
    pub points: Vec<Vector3<f64>>,
    pub sigma0: f64,
    pub redundancy: usize,
    /// gemeinsame Kovarianz aller Orientierungsparameter (6·m × 6·m), σ0²-skaliert
    pub pose_cov: DMatrix<f64>,
    pub point_cov: Vec<Matrix3<f64>>,
    /// Verbesserungen je Beobachtung (normiert)
    pub residuals: Vec<Vector2<f64>>,
    pub rejected: Vec<bool>,
    pub iterations: usize,
}

type M26 = SMatrix<f64, 2, 6>;

/// Projektion und Jacobi-Matrizen bzgl. Pose (δθ, dC) und Punkt.
#[inline]
pub fn project_jac(pose: &Pose, x: &Vector3<f64>) -> Option<(Vector2<f64>, M26, Matrix2x3<f64>)> {
    let pc = pose.to_cam(x);
    if pc.z <= 1e-9 {
        return None;
    }
    let iz = 1.0 / pc.z;
    let n = Vector2::new(pc.x * iz, pc.y * iz);
    let dn = Matrix2x3::new(iz, 0.0, -pc.x * iz * iz, 0.0, iz, -pc.y * iz * iz);
    let d_theta = dn * (-skew(&pc));
    let d_c = dn * (-pose.r);
    let d_x = dn * pose.r;
    let mut jp = M26::zeros();
    jp.fixed_view_mut::<2, 3>(0, 0).copy_from(&d_theta);
    jp.fixed_view_mut::<2, 3>(0, 3).copy_from(&d_c);
    Some((n, jp, d_x))
}

// ---------------------------------------------------------------------------
// Näherungswerte Rückwärtsschnitt
// ---------------------------------------------------------------------------

fn pca(points: &[Vector3<f64>]) -> (Vector3<f64>, Matrix3<f64>, Vector3<f64>) {
    let n = points.len() as f64;
    let m = points.iter().fold(Vector3::zeros(), |a, p| a + p) / n;
    let mut c = Matrix3::zeros();
    for p in points {
        let d = p - m;
        c += d * d.transpose();
    }
    let eig = c.symmetric_eigen();
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| eig.eigenvalues[b].partial_cmp(&eig.eigenvalues[a]).unwrap());
    let mut e = Matrix3::zeros();
    let mut ev = Vector3::zeros();
    for (k, &i) in idx.iter().enumerate() {
        e.set_column(k, &eig.eigenvectors.column(i));
        ev[k] = eig.eigenvalues[i];
    }
    if e.determinant() < 0.0 {
        e.column_mut(2).neg_mut();
    }
    (m, e, ev)
}

fn null_vector(a: &DMatrix<f64>) -> DVector<f64> {
    // kleinster Eigenvektor von AᵀA (robuster als SVD bei 2n<cols nicht nötig)
    let ata = a.transpose() * a;
    let eig = ata.symmetric_eigen();
    let mut imin = 0;
    for i in 1..eig.eigenvalues.len() {
        if eig.eigenvalues[i] < eig.eigenvalues[imin] {
            imin = i;
        }
    }
    eig.eigenvectors.column(imin).into_owned()
}

fn init_dlt(xn: &[Vector2<f64>], xw: &[Vector3<f64>]) -> Option<Pose> {
    let (m, _, ev) = pca(xw);
    let s = (ev.sum() / xw.len() as f64).sqrt().max(1e-9);
    let n = xn.len();
    let mut a = DMatrix::zeros(2 * n, 12);
    for i in 0..n {
        let p = (xw[i] - m) / s;
        let (x, y) = (xn[i].x, xn[i].y);
        let r1 = [p.x, p.y, p.z, 1.0, 0.0, 0.0, 0.0, 0.0, -x * p.x, -x * p.y, -x * p.z, -x];
        let r2 = [0.0, 0.0, 0.0, 0.0, p.x, p.y, p.z, 1.0, -y * p.x, -y * p.y, -y * p.z, -y];
        for j in 0..12 {
            a[(2 * i, j)] = r1[j];
            a[(2 * i + 1, j)] = r2[j];
        }
    }
    let h = null_vector(&a);
    let mut mp = DMatrix::from_row_slice(3, 4, h.as_slice());
    // Denormierung: X' = (X − m)/s
    let mut t = DMatrix::<f64>::identity(4, 4);
    for k in 0..3 {
        t[(k, k)] = 1.0 / s;
        t[(k, 3)] = -m[k] / s;
    }
    mp = mp * t;
    let mut a3 = Matrix3::from_fn(|i, j| mp[(i, j)]);
    let mut t3 = Vector3::new(mp[(0, 3)], mp[(1, 3)], mp[(2, 3)]);
    if a3.determinant() < 0.0 {
        a3 = -a3;
        t3 = -t3;
    }
    let sv = a3.svd(false, false).singular_values;
    let lam = sv.mean();
    let r = orthonormalize(&(a3 / lam));
    let t = t3 / lam;
    let c = -r.transpose() * t;
    Some(Pose { r, c })
}

fn init_homography(xn: &[Vector2<f64>], xw: &[Vector3<f64>]) -> Option<Pose> {
    let (o, e, ev) = pca(xw);
    let s = ((ev[0] + ev[1]) / xw.len() as f64).sqrt().max(1e-9);
    let n = xn.len();
    let mut a = DMatrix::zeros(2 * n, 9);
    for i in 0..n {
        let d = xw[i] - o;
        let pa = e.column(0).dot(&d) / s;
        let pb = e.column(1).dot(&d) / s;
        let (x, y) = (xn[i].x, xn[i].y);
        let r1 = [pa, pb, 1.0, 0.0, 0.0, 0.0, -x * pa, -x * pb, -x];
        let r2 = [0.0, 0.0, 0.0, pa, pb, 1.0, -y * pa, -y * pb, -y];
        for j in 0..9 {
            a[(2 * i, j)] = r1[j];
            a[(2 * i + 1, j)] = r2[j];
        }
    }
    let h = null_vector(&a);
    let mut hm = Matrix3::from_row_slice(h.as_slice());
    hm = hm * Matrix3::new(1.0 / s, 0.0, 0.0, 0.0, 1.0 / s, 0.0, 0.0, 0.0, 1.0);
    let h1 = hm.column(0).into_owned();
    let h2 = hm.column(1).into_owned();
    let h3 = hm.column(2).into_owned();
    let mut lam = 0.5 * (h1.norm() + h2.norm());
    if h3.z / lam < 0.0 {
        lam = -lam;
    }
    let r1 = h1 / lam;
    let r2 = h2 / lam;
    let r3 = r1.cross(&r2);
    let rp = orthonormalize(&Matrix3::from_columns(&[r1, r2, r3]));
    let t = h3 / lam;
    let r = orthonormalize(&(rp * e.transpose()));
    let c = o - r.transpose() * t;
    Some(Pose { r, c })
}

/// Näherungswerte der äußeren Orientierung aus ≥4 Passpunkten.
pub fn resection_init(xn: &[Vector2<f64>], xw: &[Vector3<f64>]) -> Result<Pose> {
    if xn.len() < 4 {
        bail!("Rückwärtsschnitt: mindestens 4 Passpunkte je Bild erforderlich ({} vorhanden)", xn.len());
    }
    let (_, _, ev) = pca(xw);
    let planar = ev[2] < 1e-3 * ev[0];
    let mut cands = Vec::new();
    if xn.len() >= 6 && !planar {
        if let Some(p) = init_dlt(xn, xw) {
            cands.push(p);
        }
    }
    if let Some(p) = init_homography(xn, xw) {
        cands.push(p);
    }
    // besten Kandidaten nach Refinement wählen
    let mut best: Option<(f64, Pose)> = None;
    for p in cands {
        if let Ok((p2, rms)) = resection_refine(p, xn, xw) {
            if best.as_ref().map_or(true, |b| rms < b.0) {
                best = Some((rms, p2));
            }
        }
    }
    best.map(|b| b.1).ok_or_else(|| anyhow::anyhow!("Rückwärtsschnitt nicht konvergiert"))
}

/// Gauß-Newton/LM für eine Pose bei festen Objektpunkten. Rückgabe: Pose, RMS (normiert).
pub fn resection_refine(mut pose: Pose, xn: &[Vector2<f64>], xw: &[Vector3<f64>]) -> Result<(Pose, f64)> {
    let mut lambda = 1e-4;
    let cost = |p: &Pose| -> f64 {
        let mut s = 0.0;
        for (o, x) in xn.iter().zip(xw) {
            match project_jac(p, x) {
                Some((n, _, _)) => s += (o - n).norm_squared(),
                None => s += 1.0,
            }
        }
        s
    };
    let mut c0 = cost(&pose);
    for _ in 0..100 {
        let mut h = Matrix6::zeros();
        let mut g = nalgebra::Vector6::zeros();
        for (o, x) in xn.iter().zip(xw) {
            if let Some((n, jp, _)) = project_jac(&pose, x) {
                let r = o - n;
                h += jp.transpose() * jp;
                g += jp.transpose() * r;
            }
        }
        let mut improved = false;
        for _ in 0..10 {
            let mut hl = h;
            for k in 0..6 {
                hl[(k, k)] *= 1.0 + lambda;
            }
            let Some(d) = hl.lu().solve(&g) else { break };
            let mut p2 = pose.clone();
            p2.update(&Vector3::new(d[0], d[1], d[2]), &Vector3::new(d[3], d[4], d[5]));
            let c1 = cost(&p2);
            if c1 < c0 {
                let rel = (c0 - c1) / c0.max(1e-300);
                pose = p2;
                c0 = c1;
                lambda = (lambda * 0.3).max(1e-9);
                improved = true;
                if rel < 1e-12 {
                    return Ok((pose, (c0 / xn.len() as f64).sqrt()));
                }
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    Ok((pose, (c0 / xn.len() as f64).sqrt()))
}

// ---------------------------------------------------------------------------
// Bündelausgleich
// ---------------------------------------------------------------------------

pub struct BundleOptions {
    pub max_iter: usize,
    /// Ausreißergrenze in Vielfachen von σ0
    pub reject_k: f64,
    pub reject_rounds: usize,
}

impl Default for BundleOptions {
    fn default() -> Self {
        Self { max_iter: 50, reject_k: 3.5, reject_rounds: 4 }
    }
}

pub fn bundle_adjust(
    poses: &[Pose],
    points: &[Vector3<f64>],
    priors: &[Option<PointPrior>],
    obs: &[Obs],
    opt: &BundleOptions,
) -> Result<BundleResult> {
    let mut rejected = vec![false; obs.len()];
    let mut res = bundle_core(poses, points, priors, obs, &rejected, opt.max_iter)?;
    for _ in 0..opt.reject_rounds {
        let mut changed = false;
        // größte normierte Verbesserung je Runde verwerfen (wie Data-Snooping, vereinfacht)
        let mut worst: Option<(usize, f64)> = None;
        for (i, o) in obs.iter().enumerate() {
            if rejected[i] {
                continue;
            }
            let w = res.residuals[i].norm() / (o.sigma * std::f64::consts::SQRT_2);
            if w > opt.reject_k * res.sigma0.max(0.5) && worst.map_or(true, |b| w > b.1) {
                worst = Some((i, w));
            }
        }
        if let Some((i, w)) = worst {
            // alle Beobachtungen nahe am Maximum gleichzeitig (effizienter bei vielen Verknüpfungspunkten)
            for (j, o) in obs.iter().enumerate() {
                if rejected[j] {
                    continue;
                }
                let wj = res.residuals[j].norm() / (o.sigma * std::f64::consts::SQRT_2);
                if wj > 0.7 * w && wj > opt.reject_k * res.sigma0.max(0.5) && priors[o.pt].is_none() {
                    rejected[j] = true;
                    changed = true;
                }
            }
            if !rejected[i] {
                rejected[i] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        res = bundle_core(&res.poses, &res.points, priors, obs, &rejected, opt.max_iter)?;
    }
    res.rejected = rejected;
    Ok(res)
}

struct Normal {
    hcc: DMatrix<f64>,
    gc: DVector<f64>,
    hpp: Vec<Matrix3<f64>>,
    gp: Vec<Vector3<f64>>,
    hcp: Vec<Vec<(usize, SMatrix<f64, 6, 3>)>>,
    cost: f64,
}

fn build_normal(
    poses: &[Pose],
    points: &[Vector3<f64>],
    priors: &[Option<PointPrior>],
    obs: &[Obs],
    rejected: &[bool],
) -> Normal {
    let m = poses.len();
    let np = points.len();
    let mut nm = Normal {
        hcc: DMatrix::zeros(6 * m, 6 * m),
        gc: DVector::zeros(6 * m),
        hpp: vec![Matrix3::zeros(); np],
        gp: vec![Vector3::zeros(); np],
        hcp: vec![Vec::new(); np],
        cost: 0.0,
    };
    for (i, o) in obs.iter().enumerate() {
        if rejected[i] {
            continue;
        }
        let Some((n, jp, jx)) = project_jac(&poses[o.img], &points[o.pt]) else {
            nm.cost += 1e6;
            continue;
        };
        let w = 1.0 / (o.sigma * o.sigma);
        let r = o.xn - n;
        nm.cost += w * r.norm_squared();
        let hc = jp.transpose() * jp * w;
        let k = 6 * o.img;
        let mut blk = nm.hcc.view_mut((k, k), (6, 6));
        blk += hc;
        let gcv = jp.transpose() * r * w;
        let mut gv = nm.gc.rows_mut(k, 6);
        gv += gcv;
        nm.hpp[o.pt] += jx.transpose() * jx * w;
        nm.gp[o.pt] += jx.transpose() * r * w;
        let hcpb: SMatrix<f64, 6, 3> = jp.transpose() * jx * w;
        if let Some(e) = nm.hcp[o.pt].iter_mut().find(|e| e.0 == o.img) {
            e.1 += hcpb;
        } else {
            nm.hcp[o.pt].push((o.img, hcpb));
        }
    }
    for (j, pr) in priors.iter().enumerate() {
        if let Some(pr) = pr {
            let r = pr.xyz - points[j];
            for k in 0..3 {
                let w = 1.0 / (pr.sigma[k] * pr.sigma[k]);
                nm.hpp[j][(k, k)] += w;
                nm.gp[j][k] += w * r[k];
                nm.cost += w * r[k] * r[k];
            }
        }
    }
    nm
}

fn bundle_core(
    poses0: &[Pose],
    points0: &[Vector3<f64>],
    priors: &[Option<PointPrior>],
    obs: &[Obs],
    rejected: &[bool],
    max_iter: usize,
) -> Result<BundleResult> {
    let m = poses0.len();
    let np = points0.len();
    let mut poses = poses0.to_vec();
    let mut points = points0.to_vec();
    let mut lambda = 1e-6;
    let mut nm = build_normal(&poses, &points, priors, obs, rejected);
    let mut it = 0;
    while it < max_iter {
        it += 1;
        let mut accepted = false;
        for _ in 0..12 {
            // Schur-Komplement
            let mut s = nm.hcc.clone();
            for k in 0..6 * m {
                s[(k, k)] *= 1.0 + lambda;
                s[(k, k)] += 1e-12;
            }
            let mut rhs = nm.gc.clone();
            let mut hpp_inv = Vec::with_capacity(np);
            for j in 0..np {
                let mut h = nm.hpp[j];
                for k in 0..3 {
                    h[(k, k)] *= 1.0 + lambda;
                }
                let hi = h.try_inverse().unwrap_or_else(Matrix3::zeros);
                for (a, ca) in &nm.hcp[j] {
                    let t = ca * hi;
                    let mut r = rhs.rows_mut(6 * a, 6);
                    r -= t * nm.gp[j];
                    for (b, cb) in &nm.hcp[j] {
                        let mut blk = s.view_mut((6 * a, 6 * b), (6, 6));
                        blk -= t * cb.transpose();
                    }
                }
                hpp_inv.push(hi);
            }
            let Some(dc) = s.clone().cholesky().map(|c| c.solve(&rhs)).or_else(|| s.lu().solve(&rhs)) else {
                bail!("Bündelausgleich: Normalgleichung singulär (Datum/Passpunkte prüfen)");
            };
            let mut np_poses = poses.clone();
            for a in 0..m {
                let d = dc.rows(6 * a, 6);
                np_poses[a].update(&Vector3::new(d[0], d[1], d[2]), &Vector3::new(d[3], d[4], d[5]));
            }
            let mut np_points = points.clone();
            for j in 0..np {
                let mut g = nm.gp[j];
                for (a, ca) in &nm.hcp[j] {
                    g -= ca.transpose() * dc.rows(6 * a, 6);
                }
                np_points[j] += hpp_inv[j] * g;
            }
            let nm2 = build_normal(&np_poses, &np_points, priors, obs, rejected);
            if nm2.cost <= nm.cost {
                let rel = (nm.cost - nm2.cost) / nm.cost.max(1e-300);
                poses = np_poses;
                points = np_points;
                nm = nm2;
                lambda = (lambda * 0.2).max(1e-12);
                accepted = true;
                if rel < 1e-10 || dc.amax() < 1e-12 {
                    it = max_iter + it;
                }
                break;
            }
            lambda *= 10.0;
        }
        if !accepted {
            break;
        }
    }

    // Statistik und Kovarianzen (ohne Dämpfung)
    let n_obs = 2 * rejected.iter().filter(|r| !**r).count() + 3 * priors.iter().filter(|p| p.is_some()).count();
    let n_unk = 6 * m + 3 * np;
    let redundancy = n_obs.saturating_sub(n_unk);
    let sigma0 = if redundancy > 0 { (nm.cost / redundancy as f64).sqrt() } else { 1.0 };
    let mut s = nm.hcc.clone();
    let mut hpp_inv = Vec::with_capacity(np);
    for j in 0..np {
        let hi = nm.hpp[j].try_inverse().unwrap_or_else(Matrix3::zeros);
        for (a, ca) in &nm.hcp[j] {
            let t = ca * hi;
            for (b, cb) in &nm.hcp[j] {
                let mut blk = s.view_mut((6 * a, 6 * b), (6, 6));
                blk -= t * cb.transpose();
            }
        }
        hpp_inv.push(hi);
    }
    let s_inv = s.try_inverse().unwrap_or_else(|| DMatrix::zeros(6 * m, 6 * m));
    let s02 = sigma0 * sigma0;
    let pose_cov = &s_inv * s02;
    let mut point_cov = Vec::with_capacity(np);
    for j in 0..np {
        let hi = hpp_inv[j];
        // Q_pp = Hpp⁻¹ + Hpp⁻¹ Hpc S⁻¹ Hcp Hpp⁻¹
        let mut w = DMatrix::zeros(6 * m, 3);
        for (a, ca) in &nm.hcp[j] {
            let t = ca * hi;
            let mut v = w.view_mut((6 * a, 0), (6, 3));
            v += t;
        }
        let add = w.transpose() * &s_inv * &w;
        let q = hi + Matrix3::from_fn(|r, c| add[(r, c)]);
        point_cov.push(q * s02);
    }
    let residuals = obs
        .iter()
        .map(|o| match project_jac(&poses[o.img], &points[o.pt]) {
            Some((n, _, _)) => o.xn - n,
            None => Vector2::new(f64::NAN, f64::NAN),
        })
        .collect();
    Ok(BundleResult {
        poses,
        points,
        sigma0,
        redundancy,
        pose_cov,
        point_cov,
        residuals,
        rejected: rejected.to_vec(),
        iterations: if it > max_iter { it - max_iter } else { it },
    })
}

// ---------------------------------------------------------------------------
// Vorwärtsschnitt
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Intersection {
    pub x: Vector3<f64>,
    /// Gesamtkovarianz (Bildmessung + Orientierung)
    pub cov: Matrix3<f64>,
    /// nur aus Bildmessung
    pub cov_meas: Matrix3<f64>,
    /// Verbesserungen [normiert] je Bild
    pub residuals: Vec<Vector2<f64>>,
}

/// Mittelpunkt-Näherung zweier/mehrerer Strahlen.
pub fn midpoint(poses: &[&Pose], xn: &[Vector2<f64>]) -> Option<Vector3<f64>> {
    let mut a = Matrix3::zeros();
    let mut b = Vector3::zeros();
    for (p, n) in poses.iter().zip(xn) {
        let d = (p.r.transpose() * Vector3::new(n.x, n.y, 1.0)).normalize();
        let m = Matrix3::identity() - d * d.transpose();
        a += m;
        b += m * p.c;
    }
    a.try_inverse().map(|ai| ai * b)
}

/// Vorwärtsschnitt mit Kovarianzfortpflanzung.
/// `img_idx` ordnet die Beobachtungen den Bildern in `pose_cov` (6·m × 6·m) zu.
pub fn intersect(
    poses: &[&Pose],
    img_idx: &[usize],
    xn: &[Vector2<f64>],
    sigma_n: &[f64],
    pose_cov: Option<&DMatrix<f64>>,
) -> Option<Intersection> {
    let mut x = midpoint(poses, xn)?;
    let k = poses.len();
    for _ in 0..20 {
        let mut n = Matrix3::zeros();
        let mut g = Vector3::zeros();
        for i in 0..k {
            let (pn, _, jx) = project_jac(poses[i], &x)?;
            let w = 1.0 / (sigma_n[i] * sigma_n[i]);
            n += jx.transpose() * jx * w;
            g += jx.transpose() * (xn[i] - pn) * w;
        }
        let d = n.try_inverse()? * g;
        x += d;
        if d.norm() < 1e-10 * (1.0 + x.norm()) {
            break;
        }
    }
    let mut n = Matrix3::zeros();
    let mut res = Vec::new();
    let mut jxs = Vec::new();
    let mut jps = Vec::new();
    for i in 0..k {
        let (pn, jp, jx) = project_jac(poses[i], &x)?;
        let w = 1.0 / (sigma_n[i] * sigma_n[i]);
        n += jx.transpose() * jx * w;
        res.push(xn[i] - pn);
        jxs.push(jx * w);
        jps.push(jp);
    }
    let ni = n.try_inverse()?;
    let cov_meas = ni;
    let mut cov = ni;
    if let Some(pc) = pose_cov {
        // dX/dEO = −N⁻¹ Σ Aᵢᵀ Pᵢ Bᵢ
        let m6 = pc.nrows();
        let mut gm = DMatrix::zeros(3, m6);
        for i in 0..k {
            let t = -(ni * jxs[i].transpose() * jps[i]);
            let mut v = gm.view_mut((0, 6 * img_idx[i]), (3, 6));
            v += t;
        }
        let add = &gm * pc * gm.transpose();
        cov += Matrix3::from_fn(|r, c| add[(r, c)]);
    }
    Some(Intersection { x, cov, cov_meas, residuals: res })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::r_cv_from_opk;

    fn synth() -> (Vec<Pose>, Vec<Vector3<f64>>) {
        let p1 = Pose { r: r_cv_from_opk(0.01, -0.02, 0.03), c: Vector3::new(-60.0, 3.0, 500.0) };
        let p2 = Pose { r: r_cv_from_opk(-0.015, 0.01, 0.02), c: Vector3::new(60.0, -2.0, 505.0) };
        let mut pts = Vec::new();
        for i in 0..7 {
            for j in 0..5 {
                let x = -150.0 + 50.0 * i as f64;
                let y = -100.0 + 50.0 * j as f64;
                pts.push(Vector3::new(x, y, 5.0 * ((x * 0.03).sin() + (y * 0.02).cos())));
            }
        }
        (vec![p1, p2], pts)
    }

    #[test]
    fn resection_and_bundle() {
        let (poses, pts) = synth();
        let xn: Vec<Vector2<f64>> = pts.iter().map(|x| project_jac(&poses[0], x).unwrap().0).collect();
        let p = resection_init(&xn[..6], &pts[..6]).unwrap();
        assert!((p.c - poses[0].c).norm() < 1e-6, "{:?}", p.c);
        // Bündel: 6 Passpunkte, Rest Verknüpfungspunkte mit Startfehler
        let mut obs = Vec::new();
        for (j, x) in pts.iter().enumerate() {
            for i in 0..2 {
                obs.push(Obs { img: i, pt: j, xn: project_jac(&poses[i], x).unwrap().0, sigma: 1e-4 });
            }
        }
        let priors: Vec<_> = (0..pts.len())
            .map(|j| if j % 5 == 0 { Some(PointPrior { xyz: pts[j], sigma: Vector3::repeat(0.01) }) } else { None })
            .collect();
        let start: Vec<_> = pts.iter().map(|p| p + Vector3::new(1.0, -1.0, 3.0)).collect();
        let mut sp = poses.clone();
        sp[0].c += Vector3::new(2.0, 1.0, -3.0);
        sp[1].update(&Vector3::new(0.01, 0.0, -0.01), &Vector3::new(-1.0, 2.0, 1.0));
        let r = bundle_adjust(&sp, &start, &priors, &obs, &BundleOptions::default()).unwrap();
        for (a, b) in r.points.iter().zip(&pts) {
            assert!((a - b).norm() < 1e-5);
        }
    }
}
