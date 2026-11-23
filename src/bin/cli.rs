//! Kommandozeile für Stapelverarbeitung und Prüfung.
//!
//!   stereo3d-cli synth <verzeichnis>            synthetisches Testpaar erzeugen
//!   stereo3d-cli run <projekt.toml> [--check]   Orientierung, SGM, DOM/DGM (+ Soll-Ist bei synth.)
//!   stereo3d-cli measure <projekt.toml> <bild 1|2> <u> <v>

use anyhow::{bail, Result};
use stereo3d::pipeline::{self, Oriented};
use stereo3d::project::Project;
use stereo3d::synth;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let log = |s: String| eprintln!("  {s}");
    match args.get(1).map(|s| s.as_str()) {
        Some("synth") => {
            let p = synth::generate(&PathBuf::from(args.get(2).map(|s| s.as_str()).unwrap_or("synth")))?;
            println!("Projekt: {}", p.display());
        }
        Some("run") => {
            let path = PathBuf::from(&args[2]);
            let check = args.iter().any(|a| a == "--check");
            let mut p = Project::load(&path)?;
            let t = Instant::now();
            let imgs = pipeline::load_images(&p)?;
            let (o, rep) = pipeline::orient(&p, &imgs, &log)?;
            println!("{rep}");
            p.orientation = Some(o.to_stored());
            p.save(&path)?;
            eprintln!("Orientierung: {:.1} s", t.elapsed().as_secs_f64());
            let t = Instant::now();
            let d = pipeline::dense(&p, &o, &imgs, &log)?;
            eprintln!("Dichte Zuordnung: {:.1} s", t.elapsed().as_secs_f64());
            let (dom, dgm, rep2) = pipeline::products(&p, &d, &log)?;
            println!("\n{rep2}");
            std::fs::write(p.out_dir().join("protokoll.txt"), format!("{rep}\n{rep2}\n"))?;
            if check {
                synth_check(&p, &o, &imgs, &d, &dom, &dgm)?;
            }
        }
        Some("measure") => {
            let p = Project::load(&PathBuf::from(&args[2]))?;
            let Some(so) = &p.orientation else { bail!("Projekt ist nicht orientiert") };
            let o = Oriented::from_stored(so);
            let imgs = pipeline::load_images(&p)?;
            let a: usize = args[3].parse::<usize>()? - 1;
            let r = pipeline::measure_point(&p, &o, &imgs, a, args[4].parse()?, args[5].parse()?, None)?;
            println!(
                "X {:.3}  Y {:.3}  Z {:.3}   σ {:.3} {:.3} {:.3} m   ρ {:.3}  Gegenpunkt Bild {}: {:.2} {:.2}",
                r.xyz.x, r.xyz.y, r.xyz.z, r.sigma.x, r.sigma.y, r.sigma.z, r.rho, 2 - a, r.px[1 - a].x, r.px[1 - a].y
            );
        }
        _ => eprintln!("Aufruf: stereo3d-cli synth <dir> | run <projekt.toml> [--check] | measure <projekt.toml> <1|2> <u> <v>"),
    }
    Ok(())
}

fn stats(v: &mut [f64]) -> String {
    if v.is_empty() {
        return "keine Werte".into();
    }
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let rms = (v.iter().map(|x| x * x).sum::<f64>() / n).sqrt();
    v.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
    let p68 = v[(0.68 * n) as usize].abs();
    let p95 = v[((0.95 * n) as usize).min(v.len() - 1)].abs();
    format!("n={} Mittel {:+.3} m, RMS {:.3} m, |Δ|68% {:.3} m, |Δ|95% {:.3} m", v.len(), mean, rms, p68, p95)
}

fn synth_check(
    p: &Project,
    o: &Oriented,
    imgs: &[stereo3d::imgutil::Gray; 2],
    d: &pipeline::DenseResult,
    dom: &stereo3d::raster::Grid,
    dgm: &stereo3d::raster::Grid,
) -> Result<()> {
    println!("\nSOLL-IST-VERGLEICH (synthetisch)");
    // Punktwolke: Höhenabweichung, Kanten (Gebäuderand ±1,5 m) ausgenommen
    let near_edge = |x: f64, y: f64| {
        synth::BUILDINGS.iter().any(|&(x0, x1, y0, y1, _)| {
            let inside_ext = x > x0 - 1.5 && x < x1 + 1.5 && y > y0 - 1.5 && y < y1 + 1.5;
            let inside_int = x > x0 + 1.5 && x < x1 - 1.5 && y > y0 + 1.5 && y < y1 - 1.5;
            inside_ext && !inside_int
        })
    };
    let mut dz = Vec::new();
    for q in d.pts.iter().step_by(7) {
        let (x, y) = (q[0] - synth::E0, q[1] - synth::N0);
        if near_edge(x, y) {
            continue;
        }
        dz.push(q[2] - synth::surface(x, y));
    }
    println!("Punktwolke  Z: {}", stats(&mut dz));
    let grid_cmp = |g: &stereo3d::raster::Grid, f: &dyn Fn(f64, f64) -> f64| {
        let mut v = Vec::new();
        for iy in 0..g.ny {
            for ix in 0..g.nx {
                let z = g.z[g.idx(ix, iy)];
                if !z.is_finite() {
                    continue;
                }
                let x = g.x0 + (ix as f64 + 0.5) * g.cell - synth::E0;
                let y = g.y0 + (iy as f64 + 0.5) * g.cell - synth::N0;
                if near_edge(x, y) {
                    continue;
                }
                v.push(z as f64 - f(x, y));
            }
        }
        v
    };
    println!("DOM         Z: {}", stats(&mut grid_cmp(dom, &synth::surface)));
    println!("DGM         Z: {}", stats(&mut grid_cmp(dgm, &synth::ground)));
    // Einzelpunktmessung an Kontrollpunkten (automatische Zuordnung ab Bild 1)
    println!("Einzelpunkte (LSM) an Kontrollpunkten, Klick = Messung Bild 1:");
    for g in p.gcps.iter().filter(|g| g.check) {
        let Some(m) = p.mark(&g.id, 0) else { continue };
        match pipeline::measure_point(p, o, imgs, 0, m.u, m.v, None) {
            Ok(r) => println!(
                "  {:>4}: dX {:+.3} dY {:+.3} dZ {:+.3} m   σ {:.3} {:.3} {:.3} m   ρ {:.3}",
                g.id,
                r.xyz.x - g.x,
                r.xyz.y - g.y,
                r.xyz.z - g.z,
                r.sigma.x,
                r.sigma.y,
                r.sigma.z,
                r.rho
            ),
            Err(e) => println!("  {:>4}: {e}", g.id),
        }
    }
    Ok(())
}
