// Wafer-map codec + colours, shared by the components (pure; tested with `node --test`).
// map_code = "<height>x<width>:<base64 of 2-bit cells>", as written by the batch job
// (wafer_embeddings.serving.encode_map): 0 = no die, 1 = pass, 2 = fail.

export const PALETTE = [
  [255, 255, 255], // no die
  [205, 205, 205], // pass
  [214, 39, 40], // fail
];

function base64ToBytes(b64) {
  if (typeof atob === "function") {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  return Uint8Array.from(Buffer.from(b64, "base64")); // node (tests)
}

export function decodeMap(code) {
  const [shape, b64] = code.split(":", 2);
  const [height, width] = shape.split("x").map(Number);
  const packed = base64ToBytes(b64);
  const cells = new Uint8Array(height * width);
  for (let i = 0; i < cells.length; i++) {
    cells[i] = (packed[i >> 2] >> ((i & 3) * 2)) & 3;
  }
  return { height, width, cells };
}

export function toImageData(map, scale = 1) {
  // RGBA pixels, each die a scale x scale block (for canvas putImageData)
  const w = map.width * scale;
  const h = map.height * scale;
  const px = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const c = PALETTE[Math.min(map.cells[Math.floor(y / scale) * map.width + Math.floor(x / scale)], 2)];
      const o = (y * w + x) * 4;
      px[o] = c[0];
      px[o + 1] = c[1];
      px[o + 2] = c[2];
      px[o + 3] = 255;
    }
  }
  return { width: w, height: h, data: px };
}

export function autoScale(map, targetPx = 160) {
  return Math.max(1, Math.floor(targetPx / Math.max(map.height, map.width)));
}
