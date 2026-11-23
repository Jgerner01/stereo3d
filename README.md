# stereo3d – photogrammetrische Stereoauswertung in Rust

Auswertung eines Stereobildpaares (Luftbild oder terrestrisch) mit kalibrierter Kamera
und Passpunkten: Bündelausgleich, dichte Zuordnung (SGM), DOM/DGM-Raster und
Einzelpunktmessung mit Genauigkeitsangabe.

## Bauen

```
cargo build --release
```

Ergebnis: `target/release/stereo3d` (GUI) und `target/release/stereo3d-cli` (Stapelbetrieb).
Windows: Rust über rustup installieren, gleicher Befehl. Linux: X11/Wayland-Bibliotheken.
Nur den Rechenkern ohne GUI bauen: `cargo build --release --no-default-features`.

## Ablauf (GUI)

1. **Projekt**: Bild 1/2 wählen, „Bilder laden“.
2. **Kamera**: innere Orientierung (Brown/OpenCV): fx, fy, cx, cy [px], k1–k3, p1, p2.
   Der Hauptpunkt wird absolut in Pixeln angegeben (Ursprung = Mitte des linken oberen Pixels).
   Metrische Kalibrierwerte (c [mm], Pixelgröße [µm], x0/y0 [mm]) lassen sich umrechnen.
3. **Passpunkte**: Import einer Textdatei `Id X Y Z [σxy σz] [K]`. `K` = Kontrollpunkt,
   der nicht in die Ausgleichung eingeht. Danach Punkt wählen, Modus „Passpunkt messen“
   und den Punkt in beiden Bildern anklicken. Die Lupe zeigt den Ausschnitt in voller
   Auflösung (8-fach). Rechtsklick löscht die Messung.
4. **Orientierung**: Rückwärtsschnitte (DLT- bzw. Homographie-Näherung, danach LM),
   automatische Verknüpfungspunkte (Harris → NCC entlang der Epipolarlinie → Rückwärtsprüfung
   → LSM im vollen Maßstab) und Bündelausgleich mit Passpunkten als gewichtete
   Beobachtungen. Das Protokoll enthält σ0, Redundanz, die äußere Orientierung (ω, φ, κ in gon)
   mit σ, die Verbesserungen der Passpunkte und die Abweichungen an den Kontrollpunkten.
5. **DOM/DGM**: Epipolarnormalisierung, Census-SGM (8 Pfade, hierarchisch, gekachelt,
   LR-Prüfung, Subpixel-Interpolation, Median- und Speckle-Filter), Vorwärtsschnitt,
   Rasterung (Median je Zelle), progressiver morphologischer Bodenfilter.
   Ausgabe: `dom.asc`, `dgm.asc` (ESRI-ASCII-Grid), `dgm.xyz`, auf Wunsch `punktwolke.xyz`,
   Normalbilder und Disparitätsbild.
6. **Einzelpunkte**: Klick in Bild 1 oder 2 → Gegenpunkt per NCC entlang der Epipolarlinie,
   Feinzuordnung per LSM (affin + radiometrisch), Vorwärtsschnitt mit
   Kovarianzfortpflanzung. σ gesamt = Bildmessung + Kovarianz der äußeren Orientierung
   (12×12 aus dem Bündel). Ein Rechtsklick im anderen Bild setzt einen Startwert, z. B. bei
   wenig Textur. Die Punkte lassen sich als CSV exportieren.

Ansichten: Bildpaar (gekoppelt zoom-/verschiebbar), Anaglyphe der Normalbilder (rot/cyan,
Parallaxe einstellbar) sowie DOM/DGM als Schummerung mit Koordinatenanzeige.

## Kommandozeile

```
stereo3d-cli synth testdaten                      # synthetisches Luftbildpaar erzeugen
stereo3d-cli run testdaten/projekt.toml --check   # komplette Auswertung + Soll-Ist
stereo3d-cli measure projekt.toml 1 1234.5 876.2  # Einzelpunkt aus Bild 1
```

## Prüfung (synthetisches Luftbildpaar)

Aufnahmekonfiguration: 2400×1800 px, c = 3000 px, Verzeichnung k1 = −0,06,
Flughöhe ≈ 450 m über Grund, Basis 122 m (Z/B ≈ 3,7), Bodenpixel 0,15 m.
Passpunktmessungen mit 0,3 px Rauschen; 8 Passpunkte, 4 Kontrollpunkte, 2 Gebäude (12 m und 8 m).

| Größe | Ergebnis |
|---|---|
| σ0 Bündel | 0,45 (117 Verknüpfungspunkte, RMS 0,04 px) |
| Kontrollpunkte (Klickmessung, 0,3 px Rauschen) | RMS X/Y/Z 0,056 / 0,036 / 0,19 m |
| Einzelpunkte LSM an Kontrollpunkten | dZ ≤ 0,08 m bei σZ ≈ 0,08 m |
| Punktwolke SGM, Z gegen Soll | RMS 0,12 m (≈ 0,2 px Disparität), 95 % < 0,22 m |
| DOM 0,5 m | RMS 0,13 m |
| DGM (Gebäude gefiltert) | RMS 0,15 m |

Laufzeit: Orientierung 0,3 s, dichte Zuordnung von 5 Mio. px ≈ 6 s (2 Kerne).

## Grenzen / mögliche Erweiterungen

- Nur zwei Bilder; die Ausgleichung ist bereits für m Bilder ausgelegt, Bildverbände
  sind daher eine überschaubare Erweiterung.
- Keine Simultankalibrierung: die innere Orientierung ist fest.
- Das Raster liegt in der XY-Ebene. Für Fassaden wäre ein Raster in einer frei gewählten
  Bezugsebene nötig.
- Die SGM-Subpixelinterpolation (Parabel) hat einen bekannten Pixel-Locking-Effekt.
  Eine LSM-Verfeinerung je Pixel würde die Genauigkeit weiter steigern, kostet aber Rechenzeit.
- Die Ausgabe erfolgt als ASCII-Grid und nicht als GeoTIFF. Der Lagebezug wird über die
  Passpunktkoordinaten hergestellt.
