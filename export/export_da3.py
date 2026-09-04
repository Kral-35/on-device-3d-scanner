"""Export Depth Anything 3 SMALL (any-view) to ExecuTorch .pte for react-native-executorch.

One static graph: N view images in, per-view depth + confidence + camera pose encoding out.

  da3: (1, N, 3, H, W) ImageNet-normalized RGB
       -> depth (1, N, H, W), conf (1, N, H, W), pose_enc (1, N, P)

Reference view strategy is fixed to "first" (view 0 is the reference; poses are
relative to it), which keeps the graph free of data-dependent view reordering.
Sky masking, quantile clamping and the GS branch are skipped: they are
data-dependent post-processing and the app does not need them.
pose_enc -> extrinsics/intrinsics conversion happens outside the graph
(pose_encoding_to_extri_intri; reimplemented in the app).

Usage: python export_da3.py --stage smoke|export|verify [--views 4] [--height 336] [--width 448]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE / "da3-repo" / "src"
OUT = HERE / "pte"
sys.path.insert(0, str(REPO))


def load_net(model: str):
    # api.py drags in moviepy/plyfile/trimesh via its export utils; build the
    # net directly from config + safetensors instead.
    from safetensors.torch import load_file

    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY

    config = load_config(MODEL_REGISTRY[f"da3-{model}"])
    net = create_object(config)
    state = load_file(HERE / "weights" / f"da3-{model}" / "model.safetensors")
    state = {k.removeprefix("model."): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    # The released checkpoint lacks a few convs of the aux ray branch; we use
    # cam_dec for poses and drop the aux outputs, so that branch may stay random.
    bad = [k for k in missing if "output_conv2_aux" not in k]
    assert not bad, f"missing non-aux keys: {bad[:5]}"
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    net.eval()
    return net


class DA3Export(torch.nn.Module):
    """Backbone + DualDPT head + camera decoder, nothing data-dependent."""

    def __init__(self, net, views: int, height: int, width: int):
        super().__init__()
        self.backbone = net.backbone
        self.head = net.head
        self.cam_dec = net.cam_dec
        self.height = height
        self.width = width
        self._pin_rope(height, width)
        self._pin_positions(views, height, width)

    def _pin_rope(self, height: int, width: int):
        # rope.forward computes `int(positions.max()) + 1`, which torch.export
        # rejects as data-dependent. With fixed input shapes the position grid
        # is a constant: values run 0..grid-1, +1 shifted for the special
        # tokens, so the table size is max(grid)+1. Pin it.
        from depth_anything_3.model.dinov2.layers.rope import RotaryPositionEmbedding2D

        rope = next(
            (m for m in self.backbone.modules() if isinstance(m, RotaryPositionEmbedding2D)),
            None,
        )
        if rope is None:
            return
        max_position = max(height, width) // 14 + 1

        # Pre-warm the frequency cache eagerly so the einsum/arange chain
        # becomes a stored constant instead of traced graph ops (coremltools
        # mistypes the traced ints). Key must match the traced lookup:
        # feature_dim = head_dim // 2, cpu, float32.
        blk = next(
            m for m in self.backbone.modules() if hasattr(m, "num_heads") and hasattr(m, "qkv")
        )
        head_dim = blk.qkv.in_features // blk.num_heads
        rope._compute_frequency_components(
            head_dim // 2, max_position, torch.device("cpu"), torch.float32
        )

        def pinned_forward(tokens, positions):
            feature_dim = tokens.size(-1) // 2
            cos_comp, sin_comp = rope._compute_frequency_components(
                feature_dim, max_position, tokens.device, tokens.dtype
            )
            vertical, horizontal = tokens.chunk(2, dim=-1)
            vertical = rope._apply_1d_rope(vertical, positions[..., 0], cos_comp, sin_comp)
            horizontal = rope._apply_1d_rope(horizontal, positions[..., 1], cos_comp, sin_comp)
            return torch.cat((vertical, horizontal), dim=-1)

        rope.forward = pinned_forward

    def _pin_positions(self, views: int, height: int, width: int):
        # The RoPE position grids are shape-derived constants; precompute them
        # so the arange/meshgrid integer chain never enters the graph
        # (coremltools mistypes it as int32 into float matmuls).
        vit = next(m for m in self.backbone.modules() if hasattr(m, "_prepare_rope"))
        pos, pos_nodiff = vit._prepare_rope(1, views, height, width, torch.device("cpu"))
        if pos is None:
            return
        self.register_buffer("_rope_pos", pos, persistent=False)
        self.register_buffer("_rope_pos_nodiff", pos_nodiff, persistent=False)

        def pinned_prepare(B, S, H, W, device):
            return self._rope_pos, self._rope_pos_nodiff

        vit._prepare_rope = pinned_prepare

    def forward(self, images: torch.Tensor):
        feats, _ = self.backbone(
            images, cam_token=None, export_feat_layers=[], ref_view_strategy="first"
        )
        out = self.head(feats, self.height, self.width, patch_start_idx=0)
        pose_enc = self.cam_dec(feats[-1][1])
        return out["depth"], out["depth_conf"], pose_enc


def make_input(views, height, width, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand((1, views, 3, height, width), generator=g)


def stage_smoke(args):
    net = load_net(args.model)
    wrapper = DA3Export(net, args.views, args.height, args.width).eval()
    x = make_input(args.views, args.height, args.width)
    t0 = time.time()
    with torch.no_grad():
        outs = wrapper(x)
    dt = time.time() - t0
    for name, o in zip(["depth", "conf", "pose_enc"], outs):
        print(f"{name}: {tuple(o.shape)} {o.dtype} min={o.min():.4f} max={o.max():.4f}")
    print(f"eager forward: {dt:.1f}s for {args.views} views @ {args.height}x{args.width}")


def stage_export(args):
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    net = load_net(args.model)
    wrapper = DA3Export(net, args.views, args.height, args.width).eval()
    x = make_input(args.views, args.height, args.width)

    t0 = time.time()
    with torch.no_grad():
        ep = torch.export.export(wrapper, (x,))
    print(f"torch.export ok in {time.time() - t0:.0f}s")

    t0 = time.time()
    lowered = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    et = lowered.to_executorch()
    OUT.mkdir(exist_ok=True)
    name = f"da3_{args.model}_{args.views}v_{args.height}x{args.width}_xnnpack.pte"
    path = OUT / name
    path.write_bytes(et.buffer)
    print(f"lower+serialize ok in {time.time() - t0:.0f}s -> {path} ({path.stat().st_size / 1e6:.0f}MB)")


def _rebuffer_rope(wrapper, height: int, width: int):
    # The pinned rope forward captures the cached cos/sin tables as lifted
    # tensor constants. The Vulkan partitioner copies those into its
    # partitions, and torch's unlift then fails on them ("lifted_tensor_0 is
    # not a buffer") when to_executorch fake-props the call_delegate node.
    # Registering the tables as real buffers keeps them out of constant
    # lifting entirely.
    from depth_anything_3.model.dinov2.layers.rope import RotaryPositionEmbedding2D

    rope = next(
        (m for m in wrapper.backbone.modules() if isinstance(m, RotaryPositionEmbedding2D)),
        None,
    )
    if rope is None:
        return
    max_position = max(height, width) // 14 + 1
    blk = next(
        m for m in wrapper.backbone.modules() if hasattr(m, "num_heads") and hasattr(m, "qkv")
    )
    head_dim = blk.qkv.in_features // blk.num_heads
    cos, sin = rope._compute_frequency_components(
        head_dim // 2, max_position, torch.device("cpu"), torch.float32
    )
    rope.register_buffer("_pinned_cos", cos, persistent=False)
    rope.register_buffer("_pinned_sin", sin, persistent=False)

    def buffered_forward(tokens, positions):
        vertical, horizontal = tokens.chunk(2, dim=-1)
        vertical = rope._apply_1d_rope(vertical, positions[..., 0], rope._pinned_cos, rope._pinned_sin)
        horizontal = rope._apply_1d_rope(horizontal, positions[..., 1], rope._pinned_cos, rope._pinned_sin)
        return torch.cat((vertical, horizontal), dim=-1)

    rope.forward = buffered_forward


def _patch_vulkan_preprocess():
    # executorch 1.4.1's Vulkan preprocess retraces the partitioned submodule
    # with ExportPass-based transforms, but the partitioner hands it fake
    # tensors from mixed FakeTensorModes, which torch's dispatch rejects.
    # Re-fakify every node's meta under one fresh mode before each pass, on a
    # deepcopy so the parent program keeps its original metadata.
    import copy
    import json
    import os
    import tempfile

    import executorch.backends.vulkan.serialization.vulkan_graph_serialize as vgs
    import executorch.backends.vulkan.vulkan_preprocess as vp
    from executorch.exir._serialize._dataclass import _DataclassEncoder
    from executorch.exir._serialize._flatbuffer import _flatc_compile
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    keep_alive = []

    def normalize_fake_modes(program):
        fm = FakeTensorMode(allow_non_fake_inputs=True)
        keep_alive.append(fm)

        def renorm(v):
            if isinstance(v, FakeTensor):
                return fm.from_tensor(torch.empty_strided(v.size(), v.stride(), dtype=v.dtype))
            if isinstance(v, (tuple, list)):
                return type(v)(renorm(e) for e in v)
            return v

        for n in program.graph_module.graph.nodes:
            if "val" in n.meta:
                n.meta["val"] = renorm(n.meta["val"])
        return program

    orig_transform = vp._transform

    def patched_transform(program, *a, **kw):
        return orig_transform(normalize_fake_modes(program), *a, **kw)

    vp._transform = patched_transform

    orig_unsafe = vp.unsafe_remove_auto_functionalized_pass

    def patched_unsafe(program):
        return orig_unsafe(copy.deepcopy(program))

    vp.unsafe_remove_auto_functionalized_pass = patched_unsafe

    # The attention mask constant serializes as -Infinity, which json.dumps
    # emits but flatc cannot parse (it wants -inf).
    def fixed_convert(vk_graph):
        s = json.dumps(vk_graph, cls=_DataclassEncoder)
        s = s.replace("-Infinity", "-inf").replace("Infinity", "inf").replace("NaN", "nan")
        with tempfile.TemporaryDirectory() as d:
            schema_path = os.path.join(d, "schema.fbs")
            with open(schema_path, "wb") as f:
                f.write(vgs._resources.read_binary(vgs.serialization_package, "schema.fbs"))
            json_path = os.path.join(d, "schema.json")
            with open(json_path, "wb") as f:
                f.write(s.encode("ascii"))
            _flatc_compile(d, schema_path, json_path)
            with open(os.path.join(d, "schema.bin"), "rb") as f:
                return f.read()

    vgs.convert_to_flatbuffer = fixed_convert


# Ops the 1.4.1 export-side partitioner delegates but the Vulkan runtime
# inside react-native-executorch's libexecutorch.so does not register
# (VK_HAS_OP aborts at load). Diffed against the .so's op-name strings.
_VULKAN_RUNTIME_MISSING = [
    ("aten", "_assert_scalar", "default"),
    ("aten", "alias_copy", "default"),
    ("aten", "bitwise_or", "Tensor"),
    ("aten", "copy", "default"),
    ("aten", "eq", "Scalar"),
    ("aten", "ge", "Scalar"),
    ("aten", "grid_sampler_2d", "default"),
    ("aten", "gt", "Scalar"),
    ("aten", "le", "Scalar"),
    ("aten", "logical_not", "default"),
    ("aten", "logical_or", "default"),
    ("aten", "lt", "Scalar"),
    ("aten", "ne", "Scalar"),
    ("aten", "sym_constrain_range_for_size", "default"),
    ("aten", "sym_size", "int"),
    ("aten", "t_copy", "default"),
    ("dim_order_ops", "_clone_dim_order", "default"),
]


def _vulkan_blocklist():
    # The partitioner compares blocklist entries against node targets, which
    # are OpOverloads pre-edge and EdgeOpOverloads post-edge; register both.
    from executorch.exir.dialects._ops import ops as exir_ops

    entries = []
    for ns, op, overload in _VULKAN_RUNTIME_MISSING:
        for root in (torch.ops, exir_ops.edge):
            try:
                entries.append(getattr(getattr(getattr(root, ns), op), overload))
            except (AttributeError, RuntimeError):
                pass
    return entries


def stage_export_vulkan(args):
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.exir import to_edge_transform_and_lower

    net = load_net(args.model)
    wrapper = DA3Export(net, args.views, args.height, args.width).eval()
    _rebuffer_rope(wrapper, args.height, args.width)
    _patch_vulkan_preprocess()
    x = make_input(args.views, args.height, args.width)

    t0 = time.time()
    with torch.no_grad():
        ep = torch.export.export(wrapper, (x,))
    print(f"torch.export ok in {time.time() - t0:.0f}s")

    t0 = time.time()
    lowered = to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner(operator_blocklist=_vulkan_blocklist())]
    )
    et = lowered.to_executorch()
    OUT.mkdir(exist_ok=True)
    name = f"da3_{args.model}_{args.views}v_{args.height}x{args.width}_vulkan.pte"
    path = OUT / name
    path.write_bytes(et.buffer)
    print(f"lower+serialize ok in {time.time() - t0:.0f}s -> {path} ({path.stat().st_size / 1e6:.0f}MB)")


def stage_verify(args):
    from executorch.runtime import Runtime

    name = f"da3_{args.model}_{args.views}v_{args.height}x{args.width}_xnnpack.pte"
    path = OUT / name
    x = make_input(args.views, args.height, args.width)

    rt = Runtime.get()
    program = rt.load_program(str(path))
    method = program.load_method("forward")
    t0 = time.time()
    pte_outs = method.execute([x.contiguous()])
    dt_pte = time.time() - t0

    net = load_net(args.model)
    wrapper = DA3Export(net, args.views, args.height, args.width).eval()
    t0 = time.time()
    with torch.no_grad():
        ref_outs = wrapper(x)
    dt_ref = time.time() - t0

    for name_, p, r in zip(["depth", "conf", "pose_enc"], pte_outs, ref_outs):
        p = torch.as_tensor(p)
        err = (p - r).abs().max().item()
        rel = err / (r.abs().max().item() + 1e-9)
        print(f"{name_}: max abs err {err:.5f} (rel {rel:.5f})")
    print(f"pte {dt_pte:.1f}s vs eager {dt_ref:.1f}s")


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_views(paths, height, width):
    """Load images, center-crop to target aspect, resize, normalize. Returns (norm, rgb01)."""
    from PIL import Image

    norm, rgb = [], []
    for p in paths:
        im = Image.open(p).convert("RGB")
        w0, h0 = im.size
        target_ar = width / height
        if w0 / h0 > target_ar:
            new_w = int(h0 * target_ar)
            im = im.crop(((w0 - new_w) // 2, 0, (w0 - new_w) // 2 + new_w, h0))
        else:
            new_h = int(w0 / target_ar)
            im = im.crop((0, (h0 - new_h) // 2, w0, (h0 - new_h) // 2 + new_h))
        im = im.resize((width, height), Image.BICUBIC)
        a = np.asarray(im, dtype=np.float32) / 255.0
        rgb.append(a)
        norm.append((a - IMAGENET_MEAN) / IMAGENET_STD)
    norm = torch.from_numpy(np.stack(norm)).permute(0, 3, 1, 2)[None]  # 1,N,3,H,W
    return norm.contiguous(), np.stack(rgb)


def pose_enc_to_cameras(pose_enc, height, width):
    """pose_enc (N,9) [t(3), quat xyzw(4), fov_h, fov_w] -> c2w (N,3,4), K (N,3,3). Numpy."""
    t = pose_enc[:, :3]
    i, j, k, r = pose_enc[:, 3], pose_enc[:, 4], pose_enc[:, 5], pose_enc[:, 6]
    two_s = 2.0 / (pose_enc[:, 3:7] ** 2).sum(-1)
    R = np.stack(
        [
            1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
            two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
            two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
        ],
        -1,
    ).reshape(-1, 3, 3)
    fy = (height / 2.0) / np.clip(np.tan(pose_enc[:, 7] / 2.0), 1e-6, None)
    fx = (width / 2.0) / np.clip(np.tan(pose_enc[:, 8] / 2.0), 1e-6, None)
    K = np.zeros((pose_enc.shape[0], 3, 3), dtype=np.float32)
    K[:, 0, 0], K[:, 1, 1] = fx, fy
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = width / 2, height / 2, 1.0
    c2w = np.concatenate([R, t[:, :, None]], axis=-1)
    return c2w.astype(np.float32), K


def unproject(depth, conf, rgb, c2w, K, conf_percentile=40.0, stride=2):
    """Fuse per-view depth into one world-space colored point cloud."""
    n, h, w = depth.shape
    pts, cols = [], []
    thresh = np.percentile(conf, conf_percentile)
    for v in range(n):
        ys, xs = np.mgrid[0:h:stride, 0:w:stride]
        d = depth[v, ys, xs]
        m = conf[v, ys, xs] >= thresh
        fx, fy, cx, cy = K[v, 0, 0], K[v, 1, 1], K[v, 0, 2], K[v, 1, 2]
        x_cam = (xs - cx) / fx * d
        y_cam = (ys - cy) / fy * d
        p_cam = np.stack([x_cam[m], y_cam[m], d[m]], -1)
        p_world = p_cam @ c2w[v, :, :3].T + c2w[v, :, 3]
        pts.append(p_world)
        cols.append(rgb[v, ys, xs][m])
    return np.concatenate(pts), np.concatenate(cols)


def save_ply(path, pts, cols):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    body = np.zeros(
        len(pts),
        dtype=[("xyz", np.float32, 3), ("rgb", np.uint8, 3)],
    )
    body["xyz"] = pts.astype(np.float32)
    body["rgb"] = (cols * 255).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(header.encode())
        body.tofile(f)


def stage_quality(args):
    import cv2
    from executorch.runtime import Runtime

    if args.images:
        paths = args.images
        frames = None
    else:
        video = HERE / "da3-repo" / "assets" / "examples" / "robot_unitree.mp4"
        cap = cv2.VideoCapture(str(video))
        all_frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            all_frames.append(fr)
        cap.release()
        # Narrow window: the robot walks, so frames far apart violate the
        # static-scene assumption. ~0.8s of video keeps it roughly still.
        idxs = np.linspace(0, min(40, len(all_frames) - 1), args.views).astype(int)
        frames = [cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2RGB) for i in idxs]
        paths = []
        (HERE / "quality").mkdir(exist_ok=True)
        for n, fr in enumerate(frames):
            p = HERE / "quality" / f"frame_{n}.png"
            cv2.imwrite(str(p), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            paths.append(str(p))

    x, rgb01 = load_views(paths, args.height, args.width)
    assert x.shape[1] == args.views, f"need exactly {args.views} images, got {x.shape[1]}"

    path = OUT / f"da3_{args.model}_{args.views}v_{args.height}x{args.width}_xnnpack.pte"
    rt = Runtime.get()
    program = rt.load_program(str(path))
    method = program.load_method("forward")
    t0 = time.time()
    depth, conf, pose_enc = [torch.as_tensor(o).numpy() for o in method.execute([x])]
    print(f"pte inference: {time.time() - t0:.1f}s")

    depth, conf, pose_enc = depth[0], conf[0], pose_enc[0]
    print(f"depth range: {depth.min():.3f}..{depth.max():.3f}")
    c2w, K = pose_enc_to_cameras(pose_enc, args.height, args.width)
    print("camera centers:")
    print(np.round(c2w[:, :, 3], 3))
    pts, cols = unproject(depth, conf, rgb01, c2w, K)
    print(f"{len(pts)} points after confidence filter")

    qdir = HERE / "quality"
    qdir.mkdir(exist_ok=True)
    save_ply(qdir / "cloud.ply", pts, cols)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    center = pts.mean(0)
    p = pts - center
    lim = np.percentile(np.abs(p), 95)
    keep = (np.abs(p) < lim * 1.5).all(1)
    p, c = p[keep], cols[keep]
    sub = np.random.default_rng(0).choice(len(p), min(len(p), 120_000), replace=False)
    fig = plt.figure(figsize=(15, 5), facecolor="black")
    for n, (elev, azim) in enumerate([(-90, -90), (-60, -90), (-75, -45)]):
        ax = fig.add_subplot(1, 3, n + 1, projection="3d", facecolor="black")
        ax.scatter(p[sub, 0], p[sub, 2], p[sub, 1], c=c[sub], s=1.0, linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    fig.tight_layout()
    fig.savefig(qdir / "orbits.png", dpi=110)
    print(f"wrote {qdir / 'cloud.ply'} and {qdir / 'orbits.png'}")


def stage_export_coreml(args):
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend

    net = load_net(args.model)
    wrapper = DA3Export(net, args.views, args.height, args.width).eval()
    x = make_input(args.views, args.height, args.width)

    t0 = time.time()
    with torch.no_grad():
        ep = torch.export.export(wrapper, (x,))
    print(f"torch.export ok in {time.time() - t0:.0f}s")

    # fp32 default: fp16 can overflow DINOv2 activation outliers
    precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    compute_unit = (
        ct.ComputeUnit.CPU_AND_GPU if args.compute_units == "cpu_and_gpu" else ct.ComputeUnit.ALL
    )
    compile_specs = CoreMLBackend.generate_compile_specs(
        compute_precision=precision,
        compute_unit=compute_unit,
        minimum_deployment_target=ct.target.iOS18,
    )
    t0 = time.time()
    from executorch.exir import EdgeCompileConfig

    lowered = to_edge_transform_and_lower(
        ep,
        # CoreML cannot ingest non-contiguous dim orders; keep everything
        # in standard contiguous layout.
        compile_config=EdgeCompileConfig(_skip_dim_order=True),
        partitioner=[
            CoreMLPartitioner(
                compile_specs=compile_specs,
                # coremltools mistypes this op (int32 x vs fp32 update);
                # run it on the portable runtime instead.
                skip_ops_for_coreml_delegation=["aten.select_scatter.default"],
            )
        ],
    )
    et = lowered.to_executorch()
    OUT.mkdir(exist_ok=True)
    suffix = "_gpu" if args.compute_units == "cpu_and_gpu" else ""
    path = OUT / f"da3_{args.model}_{args.views}v_{args.height}x{args.width}_coreml_{args.precision}{suffix}.pte"
    path.write_bytes(et.buffer)
    print(f"coreml lower ok in {time.time() - t0:.0f}s -> {path} ({path.stat().st_size / 1e6:.0f}MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        required=True,
        choices=["smoke", "export", "verify", "quality", "export-coreml", "export-vulkan"],
    )
    ap.add_argument("--model", choices=["small", "base"], default="small")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--compute-units", choices=["all", "cpu_and_gpu"], default="all")
    ap.add_argument("--images", nargs="*", default=None, help="quality stage: explicit image paths")
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--width", type=int, default=448)
    args = ap.parse_args()
    torch.manual_seed(0)
    globals()[f"stage_{args.stage.replace('-', '_')}"](args)
