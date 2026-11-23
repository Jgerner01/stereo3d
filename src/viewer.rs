//! Bildbetrachter mit Zoom/Pan, Overlays und Lupe (volle Auflösung).

use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, Sense, Stroke, StrokeKind, TextureHandle, TextureOptions, Vec2};
use stereo3d::imgutil::Gray;

pub struct Overlay {
    /// Originalpixel
    pub pos: Pos2,
    pub label: String,
    pub color: Color32,
    pub big: bool,
}

#[derive(Default)]
pub struct ViewerOut {
    pub clicked: Option<Pos2>,
    pub secondary: Option<Pos2>,
    pub hover: Option<Pos2>,
    #[allow(dead_code)]
    pub rect: Option<Rect>,
    /// Pan/Zoom geändert (für gekoppelte Ansichten)
    pub moved: bool,
}

pub struct Viewer {
    pub tex: Option<TextureHandle>,
    pub tex_scale: f32,
    pub w: f32,
    pub h: f32,
    pub zoom: f32,
    pub center: Vec2,
    pub need_fit: bool,
    loupe: Option<TextureHandle>,
    loupe_key: (i64, i64),
}

impl Default for Viewer {
    fn default() -> Self {
        Self { tex: None, tex_scale: 1.0, w: 1.0, h: 1.0, zoom: 1.0, center: Vec2::ZERO, need_fit: true, loupe: None, loupe_key: (i64::MIN, 0) }
    }
}

impl Viewer {
    pub fn set_image(&mut self, ctx: &egui::Context, name: &str, g: &Gray) {
        let max = ctx.input(|i| i.max_texture_side).min(8192);
        let mut k = 1;
        while g.w / k > max || g.h / k > max {
            k += 1;
        }
        let d = g.downsample(k);
        let img = ColorImage::from_gray([d.w, d.h], &d.to_u8());
        self.tex = Some(ctx.load_texture(name, img, TextureOptions::LINEAR));
        self.tex_scale = 1.0 / k as f32;
        self.w = g.w as f32;
        self.h = g.h as f32;
        self.need_fit = true;
        self.loupe_key = (i64::MIN, 0);
    }

    pub fn set_rgb(&mut self, ctx: &egui::Context, name: &str, w: usize, h: usize, rgb: &[u8]) {
        let img = ColorImage::from_rgb([w, h], rgb);
        let keep = self.tex.is_some() && (self.w - w as f32).abs() < 0.5 && (self.h - h as f32).abs() < 0.5;
        self.tex = Some(ctx.load_texture(name, img, TextureOptions::LINEAR));
        self.tex_scale = 1.0;
        self.w = w as f32;
        self.h = h as f32;
        if !keep {
            self.need_fit = true;
        }
    }

    pub fn to_screen(&self, rect: Rect, p: Pos2) -> Pos2 {
        rect.center() + (p.to_vec2() - self.center) * self.zoom
    }
    pub fn to_orig(&self, rect: Rect, s: Pos2) -> Pos2 {
        ((s - rect.center()) / self.zoom + self.center).to_pos2()
    }

    pub fn fit(&mut self, size: Vec2) {
        self.zoom = (size.x / self.w).min(size.y / self.h) * 0.98;
        self.center = Vec2::new(self.w / 2.0 - 0.5, self.h / 2.0 - 0.5);
    }

    pub fn show(&mut self, ui: &mut egui::Ui, overlays: &[Overlay], full: Option<&Gray>, size: Vec2) -> ViewerOut {
        let (rect, resp) = ui.allocate_exact_size(size, Sense::click_and_drag());
        let mut out = ViewerOut { rect: Some(rect), ..Default::default() };
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_gray(25));
        let Some(tex_id) = self.tex.as_ref().map(|t| t.id()) else {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, "kein Bild", egui::FontId::proportional(16.0), Color32::GRAY);
            return out;
        };
        if self.need_fit {
            self.fit(rect.size());
            self.need_fit = false;
        }
        // Zoom (Mausrad) um den Cursor
        if let Some(hp) = resp.hover_pos() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let f = (scroll * 0.002).exp() * pinch;
            if (f - 1.0).abs() > 1e-4 {
                let before = self.to_orig(rect, hp);
                self.zoom = (self.zoom * f).clamp(0.01, 64.0);
                let after = self.to_orig(rect, hp);
                self.center += before - after;
                out.moved = true;
            }
        }
        if resp.dragged() {
            self.center -= resp.drag_delta() / self.zoom;
            out.moved = true;
        }
        // Bild zeichnen (nur sichtbarer Bereich)
        let half = rect.size() / (2.0 * self.zoom);
        let lo = (self.center - half).max(Vec2::splat(-0.5));
        let hi = (self.center + half).min(Vec2::new(self.w - 0.5, self.h - 0.5));
        if lo.x < hi.x && lo.y < hi.y {
            let uv = Rect::from_min_max(
                Pos2::new((lo.x + 0.5) / self.w, (lo.y + 0.5) / self.h),
                Pos2::new((hi.x + 0.5) / self.w, (hi.y + 0.5) / self.h),
            );
            let srect = Rect::from_min_max(self.to_screen(rect, lo.to_pos2()), self.to_screen(rect, hi.to_pos2()));
            painter.image(tex_id, srect, uv, Color32::WHITE);
        }
        // Overlays
        for o in overlays {
            let s = self.to_screen(rect, o.pos);
            if !rect.expand(20.0).contains(s) {
                continue;
            }
            let r = if o.big { 12.0 } else { 7.0 };
            let st = Stroke::new(if o.big { 2.0 } else { 1.5 }, o.color);
            painter.line_segment([s - Vec2::new(r, 0.0), s - Vec2::new(2.5, 0.0)], st);
            painter.line_segment([s + Vec2::new(2.5, 0.0), s + Vec2::new(r, 0.0)], st);
            painter.line_segment([s - Vec2::new(0.0, r), s - Vec2::new(0.0, 2.5)], st);
            painter.line_segment([s + Vec2::new(0.0, 2.5), s + Vec2::new(0.0, r)], st);
            if !o.label.is_empty() {
                painter.text(s + Vec2::new(r + 2.0, -r), egui::Align2::LEFT_BOTTOM, &o.label, egui::FontId::monospace(12.0), o.color);
            }
        }
        if resp.clicked() {
            out.clicked = resp.interact_pointer_pos().map(|p| self.to_orig(rect, p));
        }
        if resp.secondary_clicked() {
            out.secondary = resp.interact_pointer_pos().map(|p| self.to_orig(rect, p));
        }
        out.hover = resp.hover_pos().map(|p| self.to_orig(rect, p));
        // Lupe (volle Auflösung, 8-fach) in der rechten oberen Ecke
        if let (Some(h), Some(g)) = (out.hover, full) {
            const R: i64 = 15;
            const M: f32 = 8.0;
            let key = (h.x.round() as i64, h.y.round() as i64);
            if key != self.loupe_key || self.loupe.is_none() {
                let n = (2 * R + 1) as usize;
                let mut px = Vec::with_capacity(n * n);
                for y in key.1 - R..=key.1 + R {
                    for x in key.0 - R..=key.0 + R {
                        let v = if x >= 0 && y >= 0 && (x as usize) < g.w && (y as usize) < g.h { g.get(x as usize, y as usize) } else { 0.0 };
                        px.push(v.clamp(0.0, 255.0) as u8);
                    }
                }
                self.loupe = Some(ui.ctx().load_texture("lupe", ColorImage::from_gray([n, n], &px), TextureOptions::NEAREST));
                self.loupe_key = key;
            }
            let side = (2 * R + 1) as f32 * M;
            let lr = Rect::from_min_size(Pos2::new(rect.right() - side - 6.0, rect.top() + 6.0), Vec2::splat(side));
            if let Some(t) = &self.loupe {
                painter.image(t.id(), lr, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
            }
            painter.rect_stroke(lr, 0.0, Stroke::new(1.0, Color32::YELLOW), StrokeKind::Outside);
            // Fadenkreuz an der exakten Cursorposition
            let c = lr.center() + Vec2::new((h.x - key.0 as f32) * M, (h.y - key.1 as f32) * M);
            let st = Stroke::new(1.0, Color32::from_rgb(255, 60, 60));
            painter.line_segment([Pos2::new(lr.left(), c.y), Pos2::new(c.x - 4.0, c.y)], st);
            painter.line_segment([Pos2::new(c.x + 4.0, c.y), Pos2::new(lr.right(), c.y)], st);
            painter.line_segment([Pos2::new(c.x, lr.top()), Pos2::new(c.x, c.y - 4.0)], st);
            painter.line_segment([Pos2::new(c.x, c.y + 4.0), Pos2::new(c.x, lr.bottom())], st);
            // Overlays auch in der Lupe
            for o in overlays {
                let q = lr.center() + Vec2::new((o.pos.x - key.0 as f32) * M, (o.pos.y - key.1 as f32) * M);
                if lr.contains(q) {
                    painter.circle_stroke(q, 6.0, Stroke::new(2.0, o.color));
                }
            }
            painter.text(
                Pos2::new(lr.left(), lr.bottom() + 2.0),
                egui::Align2::LEFT_TOP,
                format!("u {:.1}  v {:.1}", h.x, h.y),
                egui::FontId::monospace(12.0),
                Color32::YELLOW,
            );
        }
        out
    }
}
