//! Komplette Auswertung auf dem synthetischen Luftbildpaar (Orientierung, SGM, DOM/DGM).
use stereo3d::{pipeline, project::Project, synth};

#[test]
fn synthetic_pair_full_pipeline() {
    let dir = std::env::temp_dir().join("stereo3d_e2e_test");
    let _ = std::fs::remove_dir_all(&dir);
    let proj = synth::generate(&dir).unwrap();
    let p = Project::load(&proj).unwrap();
    let imgs = pipeline::load_images(&p).unwrap();
    let quiet = |_: String| {};
    let (o, rep) = pipeline::orient(&p, &imgs, &quiet).unwrap();
    assert!(rep.contains("σ0"), "Protokoll fehlt");

    // Kontrollpunkte: Vorwärtsschnitt gegen Soll
    for g in p.gcps.iter().filter(|g| g.check) {
        let (Some(a), Some(b)) = (p.mark(&g.id, 0), p.mark(&g.id, 1)) else { continue };
        let _ = (a, b);
        let m = p.mark(&g.id, 0).unwrap();
        let r = pipeline::measure_point(&p, &o, &imgs, 0, m.u, m.v, None).unwrap();
        assert!((r.xyz.x - g.x).abs() < 0.3, "dX {}", r.xyz.x - g.x);
        assert!((r.xyz.y - g.y).abs() < 0.3, "dY {}", r.xyz.y - g.y);
        assert!((r.xyz.z - g.z).abs() < 0.6, "dZ {}", r.xyz.z - g.z);
    }

    let d = pipeline::dense(&p, &o, &imgs, &quiet).unwrap();
    assert!(d.pts.len() > 100_000);
    let (dom, dgm, _) = pipeline::products(&p, &d, &quiet).unwrap();
    // DOM-RMS gegen Soll (ohne Gebäuderänder)
    let (mut s2, mut n) = (0.0, 0usize);
    for iy in 0..dom.ny {
        for ix in 0..dom.nx {
            let z = dom.z[dom.idx(ix, iy)];
            if !z.is_finite() { continue; }
            let x = dom.x0 + (ix as f64 + 0.5) * dom.cell - synth::E0;
            let y = dom.y0 + (iy as f64 + 0.5) * dom.cell - synth::N0;
            let edge = synth::BUILDINGS.iter().any(|&(x0, x1, y0, y1, _)| x > x0 - 1.5 && x < x1 + 1.5 && y > y0 - 1.5 && y < y1 + 1.5);
            if edge { continue; }
            let dz = z as f64 - synth::surface(x, y);
            s2 += dz * dz;
            n += 1;
        }
    }
    let rms = (s2 / n as f64).sqrt();
    assert!(rms < 0.3, "DOM RMS {rms}");
    assert!(dgm.z.iter().filter(|v| v.is_finite()).count() > 0);
    let out = p.out_dir();
    for f in ["dom.asc", "dgm.asc", "dom.tif", "dgm.tif", "dgm.xyz"] {
        assert!(out.join(f).exists(), "{f} fehlt");
    }
}
