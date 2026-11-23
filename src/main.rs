//! stereo3d – grafische Oberfläche (egui)

mod viewer;

use eframe::egui::{self, Color32, Pos2, RichText, Vec2};
use nalgebra::Vector2;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};
use stereo3d::geom::{gon, gon_signed, opk_from_r_cv};
use stereo3d::imgutil::Gray;
use stereo3d::pipeline::{self, Oriented, PointResult};
use stereo3d::project::{Gcp, Project};
use stereo3d::raster::Grid;
use stereo3d::rectify::Rectification;
use viewer::{Overlay, Viewer};

fn main() -> eframe::Result {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1600.0, 950.0]).with_title("stereo3d – Stereoauswertung"),
        ..Default::default()
    };
    let start = std::env::args().nth(1).map(PathBuf::from);
    eframe::run_native("stereo3d", opts, Box::new(move |cc| Ok(Box::new(App::new(cc, start)))))
}

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Projekt,
    Kamera,
    Passpunkte,
    Orientierung,
    Auswertung,
    Einzelpunkte,
}

#[derive(PartialEq, Clone, Copy)]
enum Mode {
    Navigieren,
    Passpunkt,
    Einzelpunkt,
}

#[derive(PartialEq, Clone, Copy)]
enum View {
    Bilder,
    Anaglyphe,
    Dom,
    Dgm,
}

struct AnaPair {
    left: Gray,
    right: Gray,
    /// mittlere Parallaxe (Pixel des Vorschau-Normalbilds)
    d0: f32,
}

struct DenseOut {
    report: String,
    dom: Grid,
    dgm: Grid,
}

enum Job {
    Loaded(anyhow::Result<[Gray; 2]>),
    Oriented(anyhow::Result<(Oriented, String, AnaPair)>),
    Dense(anyhow::Result<DenseOut>),
}

struct App {
    project: Project,
    path: Option<PathBuf>,
    imgs: Option<Arc<[Gray; 2]>>,
    views: [Viewer; 2],
    big: Viewer,
    oriented: Option<Oriented>,
    report: String,
    dense_report: String,
    tab: Tab,
    mode: Mode,
    view: View,
    link: bool,
    sel_gcp: Option<usize>,
    last_click: Option<(usize, f64, f64)>,
    last_result: Option<PointResult>,
    point_counter: usize,
    log: Arc<Mutex<Vec<String>>>,
    job: Option<Receiver<Job>>,
    ana: Option<AnaPair>,
    ana_shift: f32,
    ana_dirty: bool,
    dom: Option<(Grid, Grid)>,
    big_dirty: bool,
    status: String,
    new_gcp: String,
    pending_load: bool,
}

impl App {
    fn new(cc: &eframe::CreationContext, start: Option<PathBuf>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let mut app = App {
            project: Project { name: "Neues Projekt".into(), ..Default::default() },
            path: None,
            imgs: None,
            views: [Viewer::default(), Viewer::default()],
            big: Viewer::default(),
            oriented: None,
            report: String::new(),
            dense_report: String::new(),
            tab: Tab::Projekt,
            mode: Mode::Navigieren,
            view: View::Bilder,
            link: false,
            sel_gcp: None,
            last_click: None,
            last_result: None,
            point_counter: 1,
            log: Arc::new(Mutex::new(Vec::new())),
            job: None,
            ana: None,
            ana_shift: 0.0,
            ana_dirty: false,
            dom: None,
            big_dirty: false,
            status: String::new(),
            new_gcp: String::new(),
            pending_load: false,
        };
        if let Some(p) = start {
            app.open(&p);
        }
        app
    }

    fn logf(&self, s: impl Into<String>) {
        self.log.lock().unwrap().push(s.into());
    }

    fn busy(&self) -> bool {
        self.job.is_some()
    }

    fn spawn<F>(&mut self, ctx: &egui::Context, f: F)
    where
        F: FnOnce(&(dyn Fn(String) + Sync)) -> Job + Send + 'static,
    {
        let (tx, rx) = channel();
        let log = self.log.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let l = move |s: String| log.lock().unwrap().push(s);
            let r = f(&l);
            let _ = tx.send(r);
            ctx.request_repaint();
        });
        self.job = Some(rx);
    }

    fn open(&mut self, p: &std::path::Path) {
        match Project::load(p) {
            Ok(pr) => {
                self.oriented = pr.orientation.as_ref().map(Oriented::from_stored);
                self.point_counter = pr.points.len() + 1;
                self.project = pr;
                self.path = Some(p.to_path_buf());
                self.imgs = None;
                self.ana = None;
                self.dom = None;
                self.report.clear();
                self.dense_report.clear();
                self.status = format!("Projekt geladen: {}", p.display());
                self.pending_load = true;
            }
            Err(e) => self.status = format!("Fehler: {e:#}"),
        }
    }

    fn save(&mut self, ask: bool) {
        let path = if ask || self.path.is_none() {
            rfd::FileDialog::new().add_filter("Projekt", &["toml"]).set_file_name("projekt.toml").save_file()
        } else {
            self.path.clone()
        };
        if let Some(p) = path {
            if let Some(o) = &self.oriented {
                self.project.orientation = Some(o.to_stored());
            }
            match self.project.save(&p) {
                Ok(()) => {
                    self.status = format!("Gespeichert: {}", p.display());
                    self.path = Some(p);
                }
                Err(e) => self.status = format!("Fehler beim Speichern: {e:#}"),
            }
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_job(&ctx);
        if self.pending_load && !self.busy() {
            self.pending_load = false;
            self.start_load(&ctx);
        }

        egui::Panel::top("menu").show(ui, |ui| self.menu_bar(ui, &ctx));
        egui::Panel::bottom("log").resizable(true).default_size(110.0).show(ui, |ui| self.log_panel(ui));
        egui::Panel::left("side").resizable(true).default_size(470.0).show(ui, |ui| self.side_panel(ui, &ctx));
        egui::CentralPanel::default().show(ui, |ui| self.central(ui, &ctx));
        if self.busy() {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
    }
}

impl App {
    fn poll_job(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.job else { return };
        let Ok(job) = rx.try_recv() else { return };
        self.job = None;
        match job {
            Job::Loaded(Ok(imgs)) => {
                for k in 0..2 {
                    self.views[k].set_image(ctx, &format!("bild{k}"), &imgs[k]);
                    let c = &mut self.project.images[k].camera;
                    if c.width as usize != imgs[k].w || c.height as usize != imgs[k].h {
                        c.width = imgs[k].w as u32;
                        c.height = imgs[k].h as u32;
                        c.cx = (imgs[k].w as f64 - 1.0) / 2.0;
                        c.cy = (imgs[k].h as f64 - 1.0) / 2.0;
                        let msg = format!("Bild {}: Kameragröße auf {}×{} gesetzt – Kameradaten prüfen!", k + 1, c.width, c.height);
                        self.log.lock().unwrap().push(msg);
                    }
                }
                self.imgs = Some(Arc::new(imgs));
                self.status = "Bilder geladen".into();
            }
            Job::Oriented(Ok((o, rep, ana))) => {
                self.project.orientation = Some(o.to_stored());
                self.oriented = Some(o);
                self.report = rep;
                self.ana_shift = ana.d0;
                self.ana = Some(ana);
                self.ana_dirty = true;
                self.status = "Orientierung berechnet".into();
            }
            Job::Dense(Ok(d)) => {
                self.dense_report = d.report;
                self.dom = Some((d.dom, d.dgm));
                self.big_dirty = true;
                self.status = "DOM/DGM berechnet".into();
            }
            Job::Loaded(Err(e)) | Job::Oriented(Err(e)) | Job::Dense(Err(e)) => {
                self.status = format!("Fehler: {e:#}");
                self.logf(self.status.clone());
            }
        }
    }

    fn start_load(&mut self, ctx: &egui::Context) {
        if self.project.images.iter().any(|i| i.path.is_empty()) {
            return;
        }
        let paths = [self.project.resolve(&self.project.images[0].path), self.project.resolve(&self.project.images[1].path)];
        self.spawn(ctx, move |log| {
            log("Bilder werden geladen …".into());
            let r = (|| Ok([Gray::load(&paths[0])?, Gray::load(&paths[1])?]))();
            Job::Loaded(r)
        });
    }

    fn start_orient(&mut self, ctx: &egui::Context) {
        let Some(imgs) = self.imgs.clone() else {
            self.status = "Zuerst Bilder laden".into();
            return;
        };
        let p = self.project.clone();
        self.spawn(ctx, move |log| {
            let r = pipeline::orient(&p, &imgs, log).map(|(o, rep)| {
                let cams = [p.images[0].camera.clone(), p.images[1].camera.clone()];
                let s = (1800.0 / cams[0].width.max(cams[0].height) as f64).min(1.0);
                let rect = Rectification::compute(&cams, &o.poses, s);
                let (il, ir) = (rect.idx[0], rect.idx[1]);
                let left = rect.resample(0, &cams[il], &o.poses[il], &imgs[il]);
                let right = rect.resample(1, &cams[ir], &o.poses[ir], &imgs[ir]);
                let ds: Vec<f64> = o.support.iter().filter_map(|x| rect.world_to_rect(x).map(|r| r.2)).collect();
                let d0 = (ds.iter().sum::<f64>() / ds.len().max(1) as f64) as f32;
                log(format!("{rep}"));
                (o, rep, AnaPair { left, right, d0 })
            });
            Job::Oriented(r)
        });
    }

    fn start_dense(&mut self, ctx: &egui::Context) {
        let (Some(imgs), Some(o)) = (self.imgs.clone(), self.oriented.clone()) else {
            self.status = "Zuerst Bilder laden und orientieren".into();
            return;
        };
        let p = self.project.clone();
        self.spawn(ctx, move |log| {
            let r = (|| {
                let d = pipeline::dense(&p, &o, &imgs, log)?;
                let (dom, dgm, rep) = pipeline::products(&p, &d, log)?;
                Ok(DenseOut { report: rep, dom, dgm })
            })();
            Job::Dense(r)
        });
    }

    fn menu_bar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal(|ui| {
            if ui.button("Öffnen …").clicked() {
                if let Some(p) = rfd::FileDialog::new().add_filter("Projekt", &["toml"]).pick_file() {
                    self.open(&p);
                }
            }
            if ui.button("Speichern").clicked() {
                self.save(false);
            }
            if ui.button("Speichern unter …").clicked() {
                self.save(true);
            }
            ui.separator();
            ui.label("Modus:");
            ui.selectable_value(&mut self.mode, Mode::Navigieren, "Navigieren");
            ui.selectable_value(&mut self.mode, Mode::Passpunkt, "Passpunkt messen");
            ui.selectable_value(&mut self.mode, Mode::Einzelpunkt, "Einzelpunkt messen");
            ui.separator();
            ui.label("Ansicht:");
            ui.selectable_value(&mut self.view, View::Bilder, "Bildpaar");
            if ui.selectable_value(&mut self.view, View::Anaglyphe, "Anaglyphe").clicked() {
                self.ana_dirty = true;
            }
            if ui.selectable_value(&mut self.view, View::Dom, "DOM").clicked() || ui.selectable_value(&mut self.view, View::Dgm, "DGM").clicked() {
                self.big_dirty = true;
            }
            ui.separator();
            ui.checkbox(&mut self.link, "Ansichten koppeln");
            if ui.button("Einpassen").clicked() {
                for v in self.views.iter_mut() {
                    v.need_fit = true;
                }
                self.big.need_fit = true;
            }
            if self.busy() {
                ui.separator();
                ui.spinner();
                ui.label("Berechnung läuft …");
            }
            let _ = ctx;
        });
    }

    fn log_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(RichText::new(&self.status).strong());
        });
        egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
            for l in self.log.lock().unwrap().iter().rev().take(400).collect::<Vec<_>>().into_iter().rev() {
                ui.monospace(l);
            }
        });
    }

    fn side_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal_wrapped(|ui| {
            for (t, n) in [
                (Tab::Projekt, "Projekt"),
                (Tab::Kamera, "Kamera"),
                (Tab::Passpunkte, "Passpunkte"),
                (Tab::Orientierung, "Orientierung"),
                (Tab::Auswertung, "DOM/DGM"),
                (Tab::Einzelpunkte, "Einzelpunkte"),
            ] {
                ui.selectable_value(&mut self.tab, t, n);
            }
        });
        ui.separator();
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| match self.tab {
            Tab::Projekt => self.tab_projekt(ui, ctx),
            Tab::Kamera => self.tab_kamera(ui),
            Tab::Passpunkte => self.tab_gcp(ui),
            Tab::Orientierung => self.tab_orient(ui, ctx),
            Tab::Auswertung => self.tab_dense(ui, ctx),
            Tab::Einzelpunkte => self.tab_points(ui),
        });
    }

    fn tab_projekt(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal(|ui| {
            ui.label("Bezeichnung");
            ui.text_edit_singleline(&mut self.project.name);
        });
        ui.add_space(6.0);
        for k in 0..2 {
            ui.horizontal(|ui| {
                ui.label(format!("Bild {}:", k + 1));
                ui.text_edit_singleline(&mut self.project.images[k].path);
                if ui.button("…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("Bilder", &["png", "jpg", "jpeg", "tif", "tiff", "bmp"]).pick_file() {
                        self.project.images[k].path = p.to_string_lossy().into();
                    }
                }
            });
        }
        if ui.add_enabled(!self.busy(), egui::Button::new("Bilder laden")).clicked() {
            self.start_load(ctx);
        }
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label("Ergebnisverzeichnis");
            ui.text_edit_singleline(&mut self.project.settings.out_dir);
        });
        ui.add_space(10.0);
        ui.label(RichText::new("Arbeitsablauf").strong());
        ui.label(
            "1. Bilder laden, Kameradaten (Kalibrierung) eintragen\n\
             2. Passpunkte importieren (Id X Y Z [σxy σz] [K])\n\
             3. Modus „Passpunkt messen“: Punkt in der Liste wählen,\n   in beiden Bildern anklicken (Lupe beachten, Rechtsklick löscht)\n\
             4. Orientierung berechnen, Protokoll prüfen (σ0, Kontrollpunkte)\n\
             5. DOM/DGM berechnen\n\
             6. Modus „Einzelpunkt messen“: Klick in Bild 1 oder 2,\n   Gegenpunkt per NCC+LSM; Rechtsklick im anderen Bild = Startwert",
        );
    }

    fn tab_kamera(&mut self, ui: &mut egui::Ui) {
        for k in 0..2 {
            ui.label(RichText::new(format!("Kamera Bild {}", k + 1)).strong());
            let c = &mut self.project.images[k].camera;
            egui::Grid::new(format!("cam{k}")).num_columns(4).show(ui, |ui| {
                ui.label("Breite/Höhe [px]");
                ui.label(format!("{} × {}", c.width, c.height));
                ui.end_row();
                ui.label("fx [px]");
                ui.add(egui::DragValue::new(&mut c.fx).speed(0.1).max_decimals(3));
                ui.label("fy [px]");
                ui.add(egui::DragValue::new(&mut c.fy).speed(0.1).max_decimals(3));
                ui.end_row();
                ui.label("cx [px]");
                ui.add(egui::DragValue::new(&mut c.cx).speed(0.1).max_decimals(3));
                ui.label("cy [px]");
                ui.add(egui::DragValue::new(&mut c.cy).speed(0.1).max_decimals(3));
                ui.end_row();
                ui.label("k1");
                ui.add(egui::DragValue::new(&mut c.k1).speed(1e-4).max_decimals(8));
                ui.label("k2");
                ui.add(egui::DragValue::new(&mut c.k2).speed(1e-4).max_decimals(8));
                ui.end_row();
                ui.label("k3");
                ui.add(egui::DragValue::new(&mut c.k3).speed(1e-4).max_decimals(8));
                ui.label("p1");
                ui.add(egui::DragValue::new(&mut c.p1).speed(1e-5).max_decimals(8));
                ui.end_row();
                ui.label("p2");
                ui.add(egui::DragValue::new(&mut c.p2).speed(1e-5).max_decimals(8));
                ui.end_row();
            });
            ui.add_space(6.0);
        }
        if ui.button("Kamera 1 → Kamera 2 übernehmen").clicked() {
            self.project.images[1].camera = self.project.images[0].camera.clone();
        }
        ui.add_space(8.0);
        ui.label(
            RichText::new(
                "Konvention: Brown/OpenCV, normierte Bildkoordinaten; Hauptpunkt absolut in Pixel \
                 (Ursprung Mitte des linken oberen Pixels). Agisoft: cx = W/2 − 0,5 + cx_Agisoft. \
                 Metrische Kalibrierung: f[px] = c[mm] / Pixelgröße[mm].",
            )
            .small(),
        );
        egui::CollapsingHeader::new("Umrechnung aus metrischen Werten").show(ui, |ui| {
            let id = ui.scope_id().with("metric");
            let mut v: [f64; 4] = ui.data_mut(|d| *d.get_temp_mut_or(id, [100.5, 4.6, 0.0, 0.0]));
            egui::Grid::new("metric_g").show(ui, |ui| {
                ui.label("c [mm]");
                ui.add(egui::DragValue::new(&mut v[0]).speed(0.01));
                ui.label("Pixel [µm]");
                ui.add(egui::DragValue::new(&mut v[1]).speed(0.01));
                ui.end_row();
                ui.label("x0 [mm]");
                ui.add(egui::DragValue::new(&mut v[2]).speed(0.001));
                ui.label("y0 [mm]");
                ui.add(egui::DragValue::new(&mut v[3]).speed(0.001));
                ui.end_row();
            });
            if ui.button("Für beide Kameras übernehmen").clicked() {
                for k in 0..2 {
                    let c = &self.project.images[k].camera;
                    let m = stereo3d::camera::Camera::from_metric(c.width, c.height, v[0], v[1], v[2], v[3]);
                    let c = &mut self.project.images[k].camera;
                    c.fx = m.fx;
                    c.fy = m.fy;
                    c.cx = m.cx;
                    c.cy = m.cy;
                }
            }
            ui.data_mut(|d| d.insert_temp(id, v));
        });
    }

    fn tab_gcp(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Passpunkte importieren …").clicked() {
                if let Some(p) = rfd::FileDialog::new().add_filter("Koordinaten", &["txt", "csv", "dat", "xyz"]).pick_file() {
                    match self.project.import_gcps(&p) {
                        Ok(n) => self.status = format!("{n} Punkte importiert"),
                        Err(e) => self.status = format!("Fehler: {e:#}"),
                    }
                }
            }
            ui.text_edit_singleline(&mut self.new_gcp).on_hover_text("Neuer Punkt: Id");
            if ui.button("+").clicked() && !self.new_gcp.trim().is_empty() {
                self.project.gcps.push(Gcp { id: self.new_gcp.trim().into(), x: 0.0, y: 0.0, z: 0.0, sxy: 0.01, sz: 0.01, check: false });
                self.new_gcp.clear();
            }
        });
        ui.horizontal(|ui| {
            ui.label("σ Bildmessung [px]");
            ui.add(egui::DragValue::new(&mut self.project.settings.sigma_px).speed(0.01).range(0.05..=5.0));
        });
        ui.label(RichText::new("Auswahl → Modus „Passpunkt messen“ → in beiden Bildern klicken. K = Kontrollpunkt.").small());
        let mut remove = None;
        egui::Grid::new("gcps").striped(true).num_columns(9).show(ui, |ui| {
            for h in ["", "Id", "X", "Y", "Z", "σxy", "σz", "K", "B1 B2"] {
                ui.label(RichText::new(h).strong());
            }
            ui.end_row();
            for (i, g) in self.project.gcps.iter_mut().enumerate() {
                let sel = self.sel_gcp == Some(i);
                if ui.selectable_label(sel, if sel { ">" } else { "·" }).clicked() {
                    self.sel_gcp = Some(i);
                    self.mode = Mode::Passpunkt;
                }
                ui.add(egui::TextEdit::singleline(&mut g.id).desired_width(50.0));
                ui.add(egui::DragValue::new(&mut g.x).speed(0.001).max_decimals(3));
                ui.add(egui::DragValue::new(&mut g.y).speed(0.001).max_decimals(3));
                ui.add(egui::DragValue::new(&mut g.z).speed(0.001).max_decimals(3));
                ui.add(egui::DragValue::new(&mut g.sxy).speed(0.001).max_decimals(3));
                ui.add(egui::DragValue::new(&mut g.sz).speed(0.001).max_decimals(3));
                ui.checkbox(&mut g.check, "");
                let m: String = (0..2).map(|k| if self.project.marks.iter().any(|m| m.id == g.id && m.image == k) { "B+ " } else { "-- " }).collect();
                ui.horizontal(|ui| {
                    ui.monospace(m);
                    if ui.small_button("x").on_hover_text("Punkt löschen").clicked() {
                        remove = Some(i);
                    }
                });
                ui.end_row();
            }
        });
        if let Some(i) = remove {
            let id = self.project.gcps.remove(i).id;
            self.project.marks.retain(|m| m.id != id);
            self.sel_gcp = None;
        }
    }

    fn tab_orient(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let s = &mut self.project.settings;
        egui::Grid::new("os").show(ui, |ui| {
            ui.label("σ Passpunkte im Bild [px]");
            ui.add(egui::DragValue::new(&mut s.sigma_px).speed(0.01).range(0.05..=5.0));
            ui.end_row();
            ui.label("σ Verknüpfungspunkte [px]");
            ui.add(egui::DragValue::new(&mut s.tie_sigma_px).speed(0.01).range(0.02..=5.0));
            ui.end_row();
            ui.label("Raster Verknüpfungspunkte");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut s.tie_grid[0]).range(2..=80));
                ui.label("×");
                ui.add(egui::DragValue::new(&mut s.tie_grid[1]).range(2..=80));
            });
            ui.end_row();
            ui.label("Tiefenzuschlag (Anteil)");
            ui.add(egui::DragValue::new(&mut s.depth_margin).speed(0.01).range(0.0..=5.0));
            ui.end_row();
        });
        if ui.add_enabled(!self.busy(), egui::Button::new(RichText::new("Orientierung berechnen").strong())).clicked() {
            self.start_orient(ctx);
        }
        if let Some(o) = &self.oriented {
            ui.add_space(6.0);
            ui.label(RichText::new("Äußere Orientierung").strong());
            egui::Grid::new("eo").striped(true).show(ui, |ui| {
                for h in ["Bild", "X0", "Y0", "Z0", "ω [gon]", "φ [gon]", "κ [gon]"] {
                    ui.label(RichText::new(h).strong());
                }
                ui.end_row();
                for k in 0..2 {
                    let c = o.poses[k].c + o.offset;
                    let (om, ph, ka) = opk_from_r_cv(&o.poses[k].r);
                    ui.label(format!("{}", k + 1));
                    ui.monospace(format!("{:.3}", c.x));
                    ui.monospace(format!("{:.3}", c.y));
                    ui.monospace(format!("{:.3}", c.z));
                    ui.monospace(format!("{:.4}", gon_signed(om)));
                    ui.monospace(format!("{:.4}", gon_signed(ph)));
                    ui.monospace(format!("{:.4}", gon(ka)));
                    ui.end_row();
                }
            });
        }
        if !self.report.is_empty() {
            ui.add_space(6.0);
            ui.label(RichText::new("Protokoll").strong());
            egui::ScrollArea::horizontal().id_salt("rep_h").show(ui, |ui| {
                ui.add(egui::Label::new(RichText::new(&self.report).monospace().size(11.0)).extend());
            });
        }
    }

    fn tab_dense(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let s = &mut self.project.settings;
        egui::Grid::new("ds").show(ui, |ui| {
            ui.label("Maßstab Zuordnung");
            egui::ComboBox::from_id_salt("scale").selected_text(format!("{:.3}", s.dense_scale)).show_ui(ui, |ui| {
                for v in [1.0, 0.5, 0.25, 0.125] {
                    ui.selectable_value(&mut s.dense_scale, v, format!("{v} (1:{})", 1.0 / v));
                }
            });
            ui.end_row();
            ui.label("SGM P1 / P2");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut s.sgm_p1).range(1..=60));
                ui.add(egui::DragValue::new(&mut s.sgm_p2).range(2..=400));
            });
            ui.end_row();
            ui.label("Eindeutigkeit");
            ui.add(egui::DragValue::new(&mut s.sgm_uniqueness).speed(0.005).range(0.0..=0.3));
            ui.end_row();
            ui.label("Rasterweite [m] (0 = auto)");
            ui.add(egui::DragValue::new(&mut s.cell).speed(0.01).range(0.0..=100.0));
            ui.end_row();
            ui.label("Lückenfüllung [Zellen]");
            ui.add(egui::DragValue::new(&mut s.fill_cells).range(0..=50));
            ui.end_row();
            ui.label("Bodenfilter: Fenster [m]");
            ui.add(egui::DragValue::new(&mut s.dgm_window).speed(0.5).range(1.0..=500.0));
            ui.end_row();
            ui.label("Neigung [m/m] / dh0 / dh_max [m]");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut s.dgm_slope).speed(0.01).range(0.0..=5.0));
                ui.add(egui::DragValue::new(&mut s.dgm_dh0).speed(0.01).range(0.0..=10.0));
                ui.add(egui::DragValue::new(&mut s.dgm_dhmax).speed(0.1).range(0.0..=100.0));
            });
            ui.end_row();
            ui.label("Punktwolke exportieren");
            ui.checkbox(&mut s.export_cloud, "punktwolke.xyz");
            ui.end_row();
        });
        if ui.add_enabled(!self.busy() && self.oriented.is_some(), egui::Button::new(RichText::new("DOM/DGM berechnen").strong())).clicked() {
            self.start_dense(ctx);
        }
        if !self.dense_report.is_empty() {
            ui.add_space(6.0);
            ui.monospace(&self.dense_report);
        }
        ui.add_space(6.0);
        ui.label(
            RichText::new(
                "Ausgabe: dom.asc, dgm.asc (ESRI-ASCII-Grid), dgm.xyz, Normalbilder, Disparitätsbild. \
                 DGM = DOM mit progressivem morphologischem Bodenfilter; Fenster ≥ größte Gebäudeabmessung wählen.",
            )
            .small(),
        );
    }

    fn tab_points(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("LSM-Fenster (Halbbreite) [px]");
            ui.add(egui::DragValue::new(&mut self.project.settings.lsm_half).range(4..=40));
        });
        if let Some(r) = &self.last_result {
            ui.group(|ui| {
                ui.label(RichText::new("Letzte Messung").strong());
                ui.monospace(format!("X {:.3}  Y {:.3}  Z {:.3}", r.xyz.x, r.xyz.y, r.xyz.z));
                ui.monospace(format!("σ gesamt   {:.3} {:.3} {:.3} m", r.sigma.x, r.sigma.y, r.sigma.z));
                ui.monospace(format!("σ Messung  {:.3} {:.3} {:.3} m", r.sigma_meas.x, r.sigma_meas.y, r.sigma_meas.z));
                ui.monospace(format!("ρ {:.3}   σ Zuordnung {:.3} px", r.rho, r.sigma_match_px));
            });
        }
        ui.horizontal(|ui| {
            if ui.button("CSV exportieren …").clicked() {
                if let Some(p) = rfd::FileDialog::new().add_filter("CSV", &["csv"]).set_file_name("punkte.csv").save_file() {
                    match pipeline::export_points(&self.project, &p) {
                        Ok(()) => self.status = format!("Exportiert: {}", p.display()),
                        Err(e) => self.status = format!("Fehler: {e:#}"),
                    }
                }
            }
            if ui.button("Alle löschen").clicked() {
                self.project.points.clear();
            }
        });
        let mut remove = None;
        egui::Grid::new("pts").striped(true).show(ui, |ui| {
            for h in ["Id", "X", "Y", "Z", "σX", "σY", "σZ", "ρ", ""] {
                ui.label(RichText::new(h).strong());
            }
            ui.end_row();
            for (i, m) in self.project.points.iter_mut().enumerate() {
                ui.add(egui::TextEdit::singleline(&mut m.id).desired_width(45.0));
                ui.monospace(format!("{:.3}", m.x));
                ui.monospace(format!("{:.3}", m.y));
                ui.monospace(format!("{:.3}", m.z));
                ui.monospace(format!("{:.3}", m.sx));
                ui.monospace(format!("{:.3}", m.sy));
                ui.monospace(format!("{:.3}", m.sz));
                ui.monospace(format!("{:.2}", m.rho));
                if ui.small_button("x").clicked() {
                    remove = Some(i);
                }
                ui.end_row();
            }
        });
        if let Some(i) = remove {
            self.project.points.remove(i);
        }
    }

    fn overlays(&self, k: usize) -> Vec<Overlay> {
        let mut v = Vec::new();
        for m in self.project.marks.iter().filter(|m| m.image == k) {
            let check = self.project.gcps.iter().any(|g| g.id == m.id && g.check);
            let sel = self.sel_gcp.and_then(|i| self.project.gcps.get(i)).is_some_and(|g| g.id == m.id);
            v.push(Overlay {
                pos: Pos2::new(m.u as f32, m.v as f32),
                label: m.id.clone(),
                color: if sel { Color32::YELLOW } else if check { Color32::from_rgb(80, 200, 255) } else { Color32::from_rgb(255, 80, 80) },
                big: sel,
            });
        }
        for p in &self.project.points {
            let (u, vv) = if k == 0 { (p.ul, p.vl) } else { (p.ur, p.vr) };
            v.push(Overlay { pos: Pos2::new(u as f32, vv as f32), label: p.id.clone(), color: Color32::from_rgb(90, 255, 90), big: false });
        }
        if let Some(r) = &self.last_result {
            v.push(Overlay { pos: Pos2::new(r.px[k].x as f32, r.px[k].y as f32), label: String::new(), color: Color32::from_rgb(255, 0, 255), big: true });
        }
        v
    }

    fn central(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        match self.view {
            View::Bilder => self.central_pair(ui),
            View::Anaglyphe => self.central_anaglyph(ui, ctx),
            View::Dom | View::Dgm => self.central_raster(ui, ctx),
        }
    }

    fn central_pair(&mut self, ui: &mut egui::Ui) {
        let avail = ui.available_size();
        let size = Vec2::new((avail.x - 8.0) / 2.0, avail.y - 22.0);
        let old = [(self.views[0].center, self.views[0].zoom), (self.views[1].center, self.views[1].zoom)];
        let mut outs = Vec::new();
        ui.horizontal(|ui| {
            for k in 0..2 {
                ui.vertical(|ui| {
                    ui.label(RichText::new(format!("Bild {}  {}", k + 1, self.project.images[k].path)).small());
                    let ov = self.overlays(k);
                    let full = self.imgs.clone();
                    let o = self.views[k].show(ui, &ov, full.as_ref().map(|g| &g[k]), size);
                    outs.push(o);
                });
            }
        });
        if self.link {
            for k in 0..2 {
                if outs[k].moved {
                    let o = 1 - k;
                    let dc = self.views[k].center - old[k].0;
                    let fz = self.views[k].zoom / old[k].1;
                    self.views[o].center += dc;
                    self.views[o].zoom *= fz;
                    break;
                }
            }
        }
        for k in 0..2 {
            if let Some(p) = outs[k].clicked {
                self.click(k, p, false);
            }
            if let Some(p) = outs[k].secondary {
                self.click(k, p, true);
            }
        }
    }

    fn click(&mut self, k: usize, p: Pos2, secondary: bool) {
        match self.mode {
            Mode::Navigieren => {}
            Mode::Passpunkt => {
                let Some(g) = self.sel_gcp.and_then(|i| self.project.gcps.get(i)) else {
                    self.status = "Passpunkt in der Liste auswählen".into();
                    return;
                };
                let id = g.id.clone();
                if secondary {
                    self.project.marks.retain(|m| !(m.id == id && m.image == k));
                    self.status = format!("{id}: Messung Bild {} gelöscht", k + 1);
                } else {
                    self.project.set_mark(&id, k, p.x as f64, p.y as f64);
                    self.status = format!("{id}: Bild {} gemessen ({:.1}, {:.1})", k + 1, p.x, p.y);
                    // nächsten Punkt ohne Messung in diesem Bild vorschlagen
                    if self.project.mark(&id, 1 - k).is_some() {
                        if let Some(n) = self.project.gcps.iter().position(|g| self.project.mark(&g.id, 0).is_none() || self.project.mark(&g.id, 1).is_none()) {
                            self.sel_gcp = Some(n);
                        }
                    }
                }
            }
            Mode::Einzelpunkt => {
                let (Some(o), Some(imgs)) = (&self.oriented, &self.imgs) else {
                    self.status = "Zuerst orientieren".into();
                    return;
                };
                let (a, ua, va, guess) = if secondary {
                    // Startwert im anderen Bild für die letzte Messung
                    let Some((a, ua, va)) = self.last_click else { return };
                    if a == k {
                        return;
                    }
                    (a, ua, va, Some(Vector2::new(p.x as f64, p.y as f64)))
                } else {
                    (k, p.x as f64, p.y as f64, None)
                };
                match pipeline::measure_point(&self.project, o, imgs, a, ua, va, guess) {
                    Ok(r) => {
                        let replace = secondary && self.project.points.last().is_some();
                        let id = if replace { self.project.points.last().unwrap().id.clone() } else { format!("N{}", self.point_counter) };
                        let m = pipeline::to_measured(&id, &r);
                        if replace {
                            *self.project.points.last_mut().unwrap() = m;
                        } else {
                            self.project.points.push(m);
                            self.point_counter += 1;
                        }
                        self.status = format!(
                            "{id}: X {:.3} Y {:.3} Z {:.3}  σ {:.3}/{:.3}/{:.3} m  ρ {:.3}",
                            r.xyz.x, r.xyz.y, r.xyz.z, r.sigma.x, r.sigma.y, r.sigma.z, r.rho
                        );
                        self.last_result = Some(r);
                        self.last_click = Some((a, ua, va));
                        self.tab = Tab::Einzelpunkte;
                    }
                    Err(e) => {
                        self.status = format!("Einzelpunkt: {e:#}");
                        self.last_click = Some((a, ua, va));
                    }
                }
            }
        }
    }

    fn central_anaglyph(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let Some(a) = &self.ana else {
            ui.label("Anaglyphe verfügbar nach der Orientierung (Normalbilder).");
            return;
        };
        ui.horizontal(|ui| {
            ui.label("Parallaxe (Grundeinstellung) [px]");
            if ui.add(egui::Slider::new(&mut self.ana_shift, (a.d0 - 150.0)..=(a.d0 + 150.0))).changed() {
                self.ana_dirty = true;
            }
            ui.label("Rot-Cyan-Brille; Messung im Bildpaar-Modus");
        });
        if self.ana_dirty {
            let (w, h) = (a.left.w, a.left.h);
            let mut rgb = vec![0u8; w * h * 3];
            let sh = self.ana_shift.round() as isize;
            for y in 0..h {
                for x in 0..w {
                    let l = a.left.get(x, y).clamp(0.0, 255.0) as u8;
                    let xr = x as isize - sh;
                    let r = if xr >= 0 && (xr as usize) < a.right.w && y < a.right.h { a.right.get(xr as usize, y).clamp(0.0, 255.0) as u8 } else { 0 };
                    let i = 3 * (y * w + x);
                    rgb[i] = l;
                    rgb[i + 1] = r;
                    rgb[i + 2] = r;
                }
            }
            self.big.set_rgb(ctx, "anaglyphe", w, h, &rgb);
            self.ana_dirty = false;
        }
        let size = ui.available_size();
        self.big.show(ui, &[], None, size);
    }

    fn central_raster(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let Some((dom, dgm)) = &self.dom else {
            ui.label("Noch kein DOM/DGM berechnet.");
            return;
        };
        let g = if self.view == View::Dom { dom } else { dgm };
        if self.big_dirty {
            let rgb = hillshade(g);
            self.big.set_rgb(ctx, "raster", g.nx, g.ny, &rgb);
            self.big.need_fit = true;
            self.big_dirty = false;
        }
        let size = ui.available_size() - Vec2::new(0.0, 22.0);
        let out = self.big.show(ui, &[], None, size);
        if let Some(h) = out.hover {
            let (ix, iy) = (h.x.round() as isize, h.y.round() as isize);
            if ix >= 0 && iy >= 0 && (ix as usize) < g.nx && (iy as usize) < g.ny {
                let row = g.ny - 1 - iy as usize;
                let z = g.z[g.idx(ix as usize, row)];
                let x = g.x0 + (ix as f64 + 0.5) * g.cell;
                let y = g.y0 + (row as f64 + 0.5) * g.cell;
                ui.monospace(format!("X {:.2}  Y {:.2}  Z {}", x, y, if z.is_finite() { format!("{:.2}", z) } else { "—".into() }));
            }
        }
    }
}

/// Schummerung + Höhenfarbe (Norden oben)
fn hillshade(g: &Grid) -> Vec<u8> {
    let (nx, ny) = (g.nx, g.ny);
    let vals: Vec<f32> = g.z.iter().cloned().filter(|v| v.is_finite()).collect();
    let (lo, hi) = vals.iter().fold((f32::MAX, f32::MIN), |a, &b| (a.0.min(b), a.1.max(b)));
    let mut rgb = vec![0u8; nx * ny * 3];
    let az = 315f32.to_radians();
    let alt = 45f32.to_radians();
    let (lx, ly, lz) = (alt.cos() * az.sin(), alt.cos() * az.cos(), alt.sin());
    for r in 0..ny {
        let iy = ny - 1 - r;
        for ix in 0..nx {
            let z = g.z[g.idx(ix, iy)];
            let o = 3 * (r * nx + ix);
            if !z.is_finite() {
                continue;
            }
            let zx = |x: usize| g.z[g.idx(x, iy)];
            let zy = |y: usize| g.z[g.idx(ix, y)];
            let dzdx = if ix > 0 && ix + 1 < nx { (zx(ix + 1) - zx(ix - 1)) / (2.0 * g.cell as f32) } else { 0.0 };
            let dzdy = if iy > 0 && iy + 1 < ny { (zy(iy + 1) - zy(iy - 1)) / (2.0 * g.cell as f32) } else { 0.0 };
            let (dzdx, dzdy) = (if dzdx.is_finite() { dzdx } else { 0.0 }, if dzdy.is_finite() { dzdy } else { 0.0 });
            let n = (dzdx * dzdx + dzdy * dzdy + 1.0).sqrt();
            let shade = ((-dzdx * lx - dzdy * ly + lz) / n).clamp(0.0, 1.0);
            let t = ((z - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0);
            // Farbverlauf grün → gelb → braun → weiß
            let c = if t < 0.33 {
                let s = t / 0.33;
                [60.0 + 150.0 * s, 140.0 + 70.0 * s, 70.0 + 10.0 * s]
            } else if t < 0.66 {
                let s = (t - 0.33) / 0.33;
                [210.0 - 60.0 * s, 210.0 - 100.0 * s, 80.0 - 20.0 * s]
            } else {
                let s = (t - 0.66) / 0.34;
                [150.0 + 100.0 * s, 110.0 + 140.0 * s, 60.0 + 190.0 * s]
            };
            let f = 0.35 + 0.75 * shade;
            for k in 0..3 {
                rgb[o + k] = (c[k] * f).clamp(0.0, 255.0) as u8;
            }
        }
    }
    rgb
}
