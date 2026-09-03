import { HEIGHT, VIEWS, WIDTH } from '../config';

// Mirrors export/export_da3.py: pose_enc is (N, 9) = [t(3), quat xyzw(4), fov_h, fov_w],
// decoded to camera-to-world rotation+translation and pinhole intrinsics.
type Camera = {
  r: Float32Array; // 3x3 row-major
  t: [number, number, number];
  fx: number;
  fy: number;
  cx: number;
  cy: number;
};

function poseEncToCameras(poseEnc: Float32Array): Camera[] {
  const cams: Camera[] = [];
  for (let v = 0; v < VIEWS; v++) {
    const o = v * 9;
    const [i, j, k, r] = [poseEnc[o + 3], poseEnc[o + 4], poseEnc[o + 5], poseEnc[o + 6]];
    const twoS = 2 / (i * i + j * j + k * k + r * r);
    const R = new Float32Array([
      1 - twoS * (j * j + k * k), twoS * (i * j - k * r), twoS * (i * k + j * r),
      twoS * (i * j + k * r), 1 - twoS * (i * i + k * k), twoS * (j * k - i * r),
      twoS * (i * k - j * r), twoS * (j * k + i * r), 1 - twoS * (i * i + j * j),
    ]);
    cams.push({
      r: R,
      t: [poseEnc[o], poseEnc[o + 1], poseEnc[o + 2]],
      fy: HEIGHT / 2 / Math.max(Math.tan(poseEnc[o + 7] / 2), 1e-6),
      fx: WIDTH / 2 / Math.max(Math.tan(poseEnc[o + 8] / 2), 1e-6),
      cx: WIDTH / 2,
      cy: HEIGHT / 2,
    });
  }
  return cams;
}

export type PointCloud = {
  // xyz + per-point world radius in w, and rgba colors in [0,1]
  positions: Float32Array;
  colors: Float32Array;
  count: number;
  center: [number, number, number];
  radius: number;
};

const STRIDE = 1;
const CONF_PERCENTILE = 0.25;
// Depth-edge filter: a pixel whose depth differs from a neighbor by more than
// this ratio sits on an occlusion boundary and would smear as a "flying pixel".
const EDGE_RATIO = 0.08;
// Voxel grid resolution for deduplication, as a fraction of the scene radius.
// Overlapping views produce near-coincident points; one survivor per voxel.
const VOXEL_FRACTION = 1 / 420;

export function buildPointCloud(
  depth: Float32Array,
  conf: Float32Array,
  rgbs: Uint8Array[],
  poseEnc: Float32Array
): PointCloud {
  const cams = poseEncToCameras(poseEnc);
  const plane = WIDTH * HEIGHT;

  // Confidence threshold at a percentile, from a subsample.
  const sample: number[] = [];
  for (let i = 0; i < conf.length; i += 97) sample.push(conf[i]);
  sample.sort((a, b) => a - b);
  const confThresh = sample[Math.floor(sample.length * CONF_PERCENTILE)];

  const maxPoints = Math.ceil(plane / (STRIDE * STRIDE)) * VIEWS;
  const positions = new Float32Array(maxPoints * 4);
  const colors = new Float32Array(maxPoints * 4);
  let count = 0;
  let mx = 0;
  let my = 0;
  let mz = 0;

  for (let v = 0; v < VIEWS; v++) {
    const cam = cams[v];
    const dOff = v * plane;
    const rgb = rgbs[v];
    for (let y = 0; y < HEIGHT; y += STRIDE) {
      for (let x = 0; x < WIDTH; x += STRIDE) {
        const pi = y * WIDTH + x;
        const d = depth[dOff + pi];
        if (conf[dOff + pi] < confThresh) continue;
        const dr = x + STRIDE < WIDTH ? depth[dOff + pi + STRIDE] : d;
        const db = y + STRIDE < HEIGHT ? depth[dOff + pi + STRIDE * WIDTH] : d;
        if (
          Math.abs(dr - d) / d > EDGE_RATIO ||
          Math.abs(db - d) / d > EDGE_RATIO
        ) {
          continue;
        }
        const xc = ((x - cam.cx) / cam.fx) * d;
        const yc = ((y - cam.cy) / cam.fy) * d;
        const R = cam.r;
        const px = R[0] * xc + R[1] * yc + R[2] * d + cam.t[0];
        const py = R[3] * xc + R[4] * yc + R[5] * d + cam.t[1];
        const pz = R[6] * xc + R[7] * yc + R[8] * d + cam.t[2];
        const o = count * 4;
        positions[o] = px;
        positions[o + 1] = py;
        positions[o + 2] = pz;
        // True footprint: one source pixel at depth d covers d/fx world units,
        // so far points get big splats and near points stay crisp.
        positions[o + 3] = (0.5 * STRIDE * d) / cam.fx;
        colors[o] = rgb[pi * 3] / 255;
        colors[o + 1] = rgb[pi * 3 + 1] / 255;
        colors[o + 2] = rgb[pi * 3 + 2] / 255;
        colors[o + 3] = 1;
        mx += px;
        my += py;
        mz += pz;
        count++;
      }
    }
  }

  mx /= count;
  my /= count;
  mz /= count;
  // Robust radius: 90th percentile distance from the centroid.
  const dists: number[] = [];
  for (let i = 0; i < count; i += 13) {
    const o = i * 4;
    const dx = positions[o] - mx;
    const dy = positions[o + 1] - my;
    const dz = positions[o + 2] - mz;
    dists.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
  }
  dists.sort((a, b) => a - b);
  const radius = dists[Math.floor(dists.length * 0.9)] || 1;

  // Voxel dedupe: overlapping views double-layer surfaces; keep one point per
  // voxel. Keys pack the quantized coordinates into a single number.
  const voxel = radius * VOXEL_FRACTION;
  const seen = new Set<number>();
  const outPos = new Float32Array(count * 4);
  const outCol = new Float32Array(count * 4);
  let kept = 0;
  for (let i = 0; i < count; i++) {
    const o = i * 4;
    const ix = Math.round((positions[o] - mx) / voxel) + 4096;
    const iy = Math.round((positions[o + 1] - my) / voxel) + 4096;
    const iz = Math.round((positions[o + 2] - mz) / voxel) + 4096;
    if (ix < 0 || iy < 0 || iz < 0 || ix > 8191 || iy > 8191 || iz > 8191) {
      continue;
    }
    const key = ix + iy * 8192 + iz * 8192 * 8192;
    if (seen.has(key)) continue;
    seen.add(key);
    const ko = kept * 4;
    outPos[ko] = positions[o];
    outPos[ko + 1] = positions[o + 1];
    outPos[ko + 2] = positions[o + 2];
    outPos[ko + 3] = positions[o + 3];
    outCol[ko] = colors[o];
    outCol[ko + 1] = colors[o + 1];
    outCol[ko + 2] = colors[o + 2];
    outCol[ko + 3] = 1;
    kept++;
  }

  // slice, not subarray: the renderer hands the backing ArrayBuffer to the GPU,
  // which must be exactly count * vec4 bytes.
  return {
    positions: outPos.slice(0, kept * 4),
    colors: outCol.slice(0, kept * 4),
    count: kept,
    center: [mx, my, mz],
    radius,
  };
}
