// node --test: the JS decoder matches the Python encoder (fixture from encode_map).
import { test } from "node:test";
import assert from "node:assert/strict";
import { autoScale, decodeMap, PALETTE, toImageData } from "./wafer.js";

// encode_map(np.array([[0, 1, 2], [1, 1, 0]], np.uint8)) in wafer_embeddings.serving
const FIXTURE = "2x3:ZAE=";

test("decodeMap matches the Python encoder", () => {
  const m = decodeMap(FIXTURE);
  assert.equal(m.height, 2);
  assert.equal(m.width, 3);
  assert.deepEqual(Array.from(m.cells), [0, 1, 2, 1, 1, 0]);
});

test("toImageData scales and colours dies", () => {
  const img = toImageData(decodeMap(FIXTURE), 2);
  assert.equal(img.width, 6);
  assert.equal(img.height, 4);
  const px = (x, y) => Array.from(img.data.slice((y * 6 + x) * 4, (y * 6 + x) * 4 + 3));
  assert.deepEqual(px(0, 0), PALETTE[0]); // no die
  assert.deepEqual(px(2, 0), PALETTE[1]); // pass
  assert.deepEqual(px(4, 1), PALETTE[2]); // fail
});

test("autoScale keeps maps roughly target-sized", () => {
  assert.equal(autoScale({ height: 40, width: 20 }), 4);
  assert.equal(autoScale({ height: 500, width: 500 }), 1);
});
