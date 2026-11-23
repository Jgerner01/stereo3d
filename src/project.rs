//! Projektdatei (TOML): Bilder, Kameras, Passpunkte, Bildmessungen, Einstellungen,
//! Orientierungsergebnis und gemessene Einzelpunkte.

use crate::camera::Camera;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ImageEntry {
    pub path: String,
    pub camera: Camera,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gcp {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    #[serde(default = "d_sxy")]
    pub sxy: f64,
    #[serde(default = "d_sxy")]
    pub sz: f64,
    /// Kontrollpunkt: nicht in der Ausgleichung, nur zur Genauigkeitsprüfung
    #[serde(default)]
    pub check: bool,
}
fn d_sxy() -> f64 {
    0.01
}

/// Bildmessung eines Passpunkts
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mark {
    pub id: String,
    pub image: usize,
    pub u: f64,
    pub v: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    /// Messgenauigkeit Passpunkte im Bild [px]
    pub sigma_px: f64,
    /// a-priori-Genauigkeit Verknüpfungspunkte [px]
    pub tie_sigma_px: f64,
    /// Rasterzellen für Verknüpfungspunkte (x, y)
    pub tie_grid: [usize; 2],
    /// Maßstab der dichten Zuordnung (1.0 = volle Auflösung)
    pub dense_scale: f64,
    /// Zuschlag zum Tiefenbereich aus Pass-/Verknüpfungspunkten (Anteil)
    pub depth_margin: f64,
    pub sgm_p1: u16,
    pub sgm_p2: u16,
    pub sgm_uniqueness: f32,
    /// Rasterweite DOM/DGM [m]; 0 = automatisch (≈ 2 × Bodenpixel der dichten Zuordnung)
    pub cell: f64,
    /// Lückenfüllung DOM [Zellen]
    pub fill_cells: usize,
    /// Bodenfilter: max. Fenster [m], Geländeneigung [m/m], dh0 [m], dh_max [m]
    pub dgm_window: f64,
    pub dgm_slope: f64,
    pub dgm_dh0: f64,
    pub dgm_dhmax: f64,
    /// LSM-Fenster Einzelpunktmessung (Halbbreite) [px]
    pub lsm_half: usize,
    pub export_cloud: bool,
    pub out_dir: String,
    /// EPSG-Code des Objektkoordinatensystems für GeoTIFF (0 = unbekannt/benutzerdefiniert)
    #[serde(default)]
    pub epsg: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            sigma_px: 0.5,
            tie_sigma_px: 0.3,
            tie_grid: [24, 16],
            dense_scale: 0.5,
            depth_margin: 0.3,
            sgm_p1: 8,
            sgm_p2: 96,
            sgm_uniqueness: 0.03,
            cell: 0.0,
            fill_cells: 3,
            dgm_window: 40.0,
            dgm_slope: 0.3,
            dgm_dh0: 0.3,
            dgm_dhmax: 3.0,
            lsm_half: 10,
            export_cloud: false,
            out_dir: "ergebnis".into(),
            epsg: 0,
        }
    }
}

/// Ergebnis der Orientierung (absolute Koordinaten)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredOrientation {
    /// Reduktion (Schwerpunkt der Passpunkte)
    pub offset: [f64; 3],
    /// je Bild: Projektionszentrum (reduziert) und Rotation Welt→Kamera (zeilenweise)
    pub c: [[f64; 3]; 2],
    pub r: [[f64; 9]; 2],
    /// Kovarianz der Orientierungsparameter (12×12, zeilenweise)
    pub cov: Vec<f64>,
    pub sigma0: f64,
    /// Objektpunkte (reduziert) zur Ableitung des Tiefenbereichs
    pub support_points: Vec<[f64; 3]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeasuredPoint {
    pub id: String,
    pub ul: f64,
    pub vl: f64,
    pub ur: f64,
    pub vr: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub sx: f64,
    pub sy: f64,
    pub sz: f64,
    pub rho: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Project {
    pub name: String,
    pub images: [ImageEntry; 2],
    #[serde(default)]
    pub gcps: Vec<Gcp>,
    #[serde(default)]
    pub marks: Vec<Mark>,
    #[serde(default)]
    pub settings: Settings,
    #[serde(default)]
    pub orientation: Option<StoredOrientation>,
    #[serde(default)]
    pub points: Vec<MeasuredPoint>,
    /// Verzeichnis der Projektdatei (nicht gespeichert)
    #[serde(skip)]
    pub base_dir: PathBuf,
}

impl Project {
    pub fn load(path: &Path) -> Result<Project> {
        let s = std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?;
        let mut p: Project = toml::from_str(&s).context("Projektdatei fehlerhaft")?;
        p.base_dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        Ok(p)
    }

    pub fn save(&mut self, path: &Path) -> Result<()> {
        self.base_dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        std::fs::write(path, toml::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn resolve(&self, p: &str) -> PathBuf {
        let pb = PathBuf::from(p);
        if pb.is_absolute() { pb } else { self.base_dir.join(pb) }
    }

    pub fn out_dir(&self) -> PathBuf {
        self.resolve(&self.settings.out_dir)
    }

    pub fn mark(&self, id: &str, image: usize) -> Option<&Mark> {
        self.marks.iter().find(|m| m.id == id && m.image == image)
    }

    pub fn set_mark(&mut self, id: &str, image: usize, u: f64, v: f64) {
        if let Some(m) = self.marks.iter_mut().find(|m| m.id == id && m.image == image) {
            m.u = u;
            m.v = v;
        } else {
            self.marks.push(Mark { id: id.into(), image, u, v });
        }
    }

    /// Passpunktliste einlesen: „Id X Y Z [σxy σz] [K]“, Trennzeichen Leerzeichen, Tab, ; oder ,
    /// (Dezimalkomma wird bei ;-Trennung akzeptiert). „K“/„check“ markiert Kontrollpunkte.
    pub fn import_gcps(&mut self, path: &Path) -> Result<usize> {
        let s = std::fs::read_to_string(path)?;
        let mut n = 0;
        for line in s.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let semi = line.contains(';');
            let toks: Vec<String> = if semi {
                line.split(';').map(|t| t.trim().replace(',', ".")).collect()
            } else {
                line.split(|c: char| c.is_whitespace() || c == ',').filter(|t| !t.is_empty()).map(|t| t.to_string()).collect()
            };
            if toks.len() < 4 {
                continue;
            }
            let num = |i: usize| toks.get(i).and_then(|t| t.parse::<f64>().ok());
            let (Some(x), Some(y), Some(z)) = (num(1), num(2), num(3)) else { continue };
            let check = toks.iter().skip(4).any(|t| t.eq_ignore_ascii_case("k") || t.eq_ignore_ascii_case("check"));
            let g = Gcp { id: toks[0].clone(), x, y, z, sxy: num(4).unwrap_or(0.01), sz: num(5).unwrap_or(0.01), check };
            if let Some(e) = self.gcps.iter_mut().find(|e| e.id == g.id) {
                *e = g;
            } else {
                self.gcps.push(g);
            }
            n += 1;
        }
        Ok(n)
    }
}
