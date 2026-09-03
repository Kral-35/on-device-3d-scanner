# On-device 3D scanner

Take 8 photos of an object and get a 3D scan you can spin with your finger. Runs fully on-device with [react-native-executorch](https://github.com/software-mansion/react-native-executorch): no cloud, no LiDAR, no ARKit session, works in airplane mode. About 3 seconds and around a million points per scan on an iPhone 16 Pro.

The model is [Depth Anything 3](https://github.com/bytedance-seed/depth-anything-3) (ByteDance Seed, ICLR 2026) in its BASE any-view variant: 0.12B parameters, N photos in, per-view depth, confidence and camera poses out, in one forward pass. The SMALL variant (0.08B) ships too, one line in [src/config.ts](src/config.ts) switches to it for a ~2 second scan at slightly lower quality. There is no SfM, no feature matching and no per-scene optimization anywhere in the pipeline; the transformer does all of it. The point cloud is rendered with [TypeGPU](https://typegpu.com) and [react-native-wgpu](https://github.com/software-mansion/react-native-webgpu). No native code in the app.

## How it works

1. A guided camera flow collects 8 photos (or you pick 8 from the library). Circle the object with small steps, or sweep across a scene; neighboring photos should overlap a lot.
2. Skia rasterizes each photo to 420x560 and normalizes with ImageNet statistics.
3. One ExecuTorch graph (`da3_{base|small}_8v_560x420_*.pte`, CoreML on iOS, XNNPACK on Android) maps the stacked `(1, 8, 3, 560, 420)` batch to depth `(1, 8, 560, 420)`, confidence, and a 9-number pose encoding per view: translation, quaternion, and vertical/horizontal field of view. View 0 is the reference frame.
4. TypeScript in [src/pipeline/pointcloud.ts](src/pipeline/pointcloud.ts) decodes the poses to cameras and unprojects every pixel of every view into a shared world space, with three filters: a confidence percentile cut, a depth-edge test that removes "flying pixels" at occlusion boundaries, and a voxel dedupe so overlapping views do not double-layer surfaces.
5. A TypeGPU pipeline renders the surviving points as gaussian splats at interactive framerates: each point carries its true world-space footprint (a pixel at depth `d` covers `d/fx` world units, so far points grow to close their gaps), drawn as billboards with a gaussian alpha falloff. Orbit with a finger (with fling inertia), pinch to zoom, auto-rotate until the first touch.

## Running it

```bash
npm install
npx expo run:ios
```

Requires an Expo dev build (New Architecture; Expo Go does not support react-native-executorch). The model is downloaded once from [Hugging Face](https://huggingface.co/nklockiewicz/react-native-executorch-demo-models) and cached, ~480MB on iOS for the default BASE model. If you want to serve your own export instead, copy `.env.example` to `.env` and point `EXPO_PUBLIC_MODEL_BASE` at your own host.

Change `ios.bundleIdentifier` in [app.json](app.json) to something of your own before building.

## Exporting the model yourself

See [export/export_da3.py](export/export_da3.py). You need `executorch==1.4.1`, the [Depth Anything 3 repo](https://github.com/bytedance-seed/depth-anything-3) in `export/da3-repo`, and the [DA3-SMALL](https://huggingface.co/depth-anything/DA3-SMALL) safetensors in `export/weights/da3-small`.

```bash
python export_da3.py --stage smoke          # eager sanity check
python export_da3.py --stage export         # XNNPACK fp32 .pte, cross-platform CPU
python export_da3.py --stage export-coreml --model base --precision fp16 --compute-units cpu_and_gpu
python export_da3.py --stage verify         # eager vs .pte on test images
```

`--model` (small | base), `--views`, `--height` and `--width` are baked into the graph at export time and must match [src/config.ts](src/config.ts). For the CoreML stage, `--precision fp16` halves the file and `--compute-units cpu_and_gpu` keeps the graph off the Neural Engine, see the notes below for why that matters.

Notes on what the export does to the paper code:

- The reference-view strategy is pinned to "first" so the graph stays free of data-dependent view reordering. Sky masking, quantile clamping and the gaussian-splat branch are post-processing the app does not need, and are skipped.
- DA3's RoPE computes `int(positions.max()) + 1` at runtime, which `torch.export` rejects as data-dependent. The position tables are precomputed for the fixed view count and resolution and baked into the graph.
- The pose encoding to extrinsics/intrinsics conversion happens outside the graph, reimplemented in ~30 lines of TypeScript.
- The released checkpoint is missing a few convolutions of an auxiliary ray branch. The export uses the camera decoder for poses and drops the aux outputs, so that branch may stay randomly initialized.

## Numbers

Full scan, 8 views at 420x560, warm model, iPhone 16 Pro:

| model | backend | inference | download |
|---|---|---|---|
| BASE (default) | CoreML fp16, CPU+GPU | ~3.1s | 480MB |
| SMALL | CoreML fp32 | ~2.0s | 382MB |

The first inference after install is much slower because CoreML compiles the model once and caches it. On the simulator CoreML falls back to CPU (~9.5s for SMALL). A smaller 448x336 SMALL export runs in ~0.9s on the phone if you want speed over detail.

## Notes from the build

- **The Neural Engine silently kills the BASE model.** With default compute units, CoreML hands the graph to the ANE, whose compiler aborts the whole process mid-inference: no crash report, no memory-kill record, the app just vanishes. No amount of memory entitlement helps because it was never about memory. Exporting with `ct.ComputeUnit.CPU_AND_GPU` sidesteps the ANE entirely and the same graph runs happily on the GPU at 3.1s. The SMALL model compiles fine on the ANE, so whatever the compiler dislikes appears only at ViT-B scale. If your CoreML-delegated model dies on a physical device with no trace, try this before anything else.
- **Aim the photos at what you want scanned.** The pipeline reconstructs the whole frame of every photo, so orbiting an object gets you the object and its surroundings, and sweeping across a room corner gets you walls and floor. What was never photographed does not exist in the scan.
- **The confidence output is doing real work.** Distant or reflective regions (a glossy laptop screen, a far wall) get low confidence, and the percentile cut is what keeps them from smearing across the scene. It is a quality dial: raise it for cleaner scans of a single object, lower it to keep more background.
- **Depth edges become "flying pixels" if you keep them.** A pixel whose depth differs from its neighbor by more than 8% sits on an occlusion boundary and unprojects into a streak between foreground and background. Dropping those pixels removes most of the visual noise.
- **Overlapping views double-layer every surface.** Eight photos of the same object produce up to eight coincident sheets of points. A voxel-grid dedupe (one survivor per cell, sized at 1/420 of the scene radius) trims the double-layers while keeping surfaces dense.

## Android

The XNNPACK export and code paths are in place and the app selects them automatically, but I have not benchmarked it on an Android device yet. Expect fp32 CPU inference to be slower than the iOS numbers above.

## Credit and license

Depth Anything 3 is by Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng and Bingyi Kang at ByteDance Seed ([paper](https://arxiv.org/abs/2511.10647), [code](https://github.com/bytedance-seed/depth-anything-3)). The DA3-SMALL code and weights are Apache 2.0, and the exported weights in the Hugging Face repo above inherit that license.

App code is MIT.
