use stereo3d::raster::Grid;

fn u16at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn f64at(b: &[u8], o: usize) -> f64 {
    f64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

#[test]
fn from_points_median_and_geotiff_roundtrip() {
    // 3x2 Zellen, 1 m; eine Zelle leer
    let pts = vec![
        [100.2, 200.2, 5.0],
        [100.4, 200.4, 7.0],
        [100.3, 200.3, 6.0], // Median 6
        [101.5, 200.5, 8.0],
        [102.5, 201.5, 9.0],
    ];
    let g = Grid::from_points(&pts, 1.0).unwrap();
    assert_eq!((g.nx, g.ny), (3, 2));
    assert_eq!(g.z[g.idx(0, 0)], 6.0);
    assert!(g.z[g.idx(0, 1)].is_nan());

    let b = g.geotiff_bytes(25832);
    assert_eq!(&b[0..2], b"II");
    assert_eq!(u16at(&b, 2), 42);
    let ifd = u32at(&b, 4) as usize;
    let n = u16at(&b, ifd) as usize;
    let (mut w, mut h, mut off, mut tie, mut scale, mut keys) = (0, 0, 0, 0usize, 0usize, 0usize);
    let mut prev = 0;
    for i in 0..n {
        let e = ifd + 2 + i * 12;
        let tag = u16at(&b, e);
        assert!(tag > prev, "Tags müssen aufsteigend sein");
        prev = tag;
        let val = u32at(&b, e + 8) as usize;
        match tag {
            256 => w = val,
            257 => h = val,
            273 => off = val,
            33922 => tie = val,
            33550 => scale = val,
            34735 => keys = val,
            _ => {}
        }
    }
    assert_eq!((w, h), (3, 2));
    assert_eq!(f64at(&b, scale), 1.0);
    assert_eq!(f64at(&b, tie + 24), g.x0);
    assert_eq!(f64at(&b, tie + 32), g.y0 + 2.0); // obere Kante
    // GeoKey ProjectedCSType = 25832
    let nk = u16at(&b, keys + 6) as usize;
    let found = (0..nk).any(|k| u16at(&b, keys + 8 + k * 8) == 3072 && u16at(&b, keys + 8 + k * 8 + 6) == 25832);
    assert!(found);
    // Pixel: Zeile 0 = Norden = iy 1 → (0,1) NoData, (2,1)=9
    let px = |i: usize| f32::from_le_bytes(b[off + 4 * i..off + 4 * i + 4].try_into().unwrap());
    assert_eq!(px(0), -9999.0);
    assert_eq!(px(2), 9.0);
    assert_eq!(px(3), 6.0);
    assert_eq!(b.len(), off + 3 * 2 * 4);
}
