import React, { useEffect, useRef } from 'react';
import { StyleSheet, View } from 'react-native';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import { Canvas, useCanvasRef } from 'react-native-webgpu';
import tgpu, { d, std } from 'typegpu';
import { mat4 } from 'wgpu-matrix';
import type { PointCloud } from '../pipeline/pointcloud';

// Splat diameter as a fraction of the scene radius.
const SPLAT_SCALE = 0.006;
const AUTO_ROTATE_SPEED = 0.35; // rad/s until the first touch
const INERTIA_DECAY = 4; // 1/s exponential decay of fling velocity

type Orbit = { yaw: number; pitch: number; dist: number };

export function PointCloudView({ cloud }: { cloud: PointCloud }) {
  const canvasRef = useCanvasRef();
  const orbit = useRef<Orbit>({ yaw: 0, pitch: -0.4, dist: 2.5 });
  const gestureStart = useRef<Orbit>({ ...orbit.current });
  const velocity = useRef({ yaw: 0, pitch: 0 });
  const dragging = useRef(false);
  const interacted = useRef(false);

  useEffect(() => {
    let stop = false;
    let cleanup: (() => void) | undefined;

    (async () => {
      // The canvas needs a layout pass before a context exists.
      let context = null;
      for (let tries = 0; tries < 60 && !context; tries++) {
        try {
          context = canvasRef.current?.getContext('webgpu') ?? null;
        } catch {
          context = null;
        }
        if (!context) await new Promise((r) => setTimeout(r, 50));
      }
      if (!context || stop) return;

      const root = await tgpu.init();
      const device = root.device;
      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: 'opaque' });

      const { width, height } = context.canvas;

      const positions = root
        .createReadonly(d.arrayOf(d.vec4f, cloud.count))
        .$name('positions');
      const colors = root
        .createReadonly(d.arrayOf(d.vec4f, cloud.count))
        .$name('colors');
      positions.buffer.write(cloud.positions.buffer as ArrayBuffer);
      colors.buffer.write(cloud.colors.buffer as ArrayBuffer);

      const Camera = d.struct({
        view: d.mat4x4f,
        proj: d.mat4x4f,
        splatSize: d.f32,
      });
      const camera = root.createUniform(Camera);

      const corners = tgpu.const(d.arrayOf(d.vec2f, 6), [
        d.vec2f(-1, -1),
        d.vec2f(1, -1),
        d.vec2f(-1, 1),
        d.vec2f(-1, 1),
        d.vec2f(1, -1),
        d.vec2f(1, 1),
      ]);

      const vertexMain = tgpu.vertexFn({
        in: {
          vertexIndex: d.builtin.vertexIndex,
          instanceIndex: d.builtin.instanceIndex,
        },
        out: { position: d.builtin.position, color: d.vec4f, uv: d.vec2f },
      })((input) => {
        'use gpu';
        const p = positions.$[input.instanceIndex];
        const corner = corners.$[input.vertexIndex];
        // Billboard in view space so splats have a real world size: they grow
        // as the camera approaches and close up into a surface.
        const viewPos = std.mul(camera.$.view, d.vec4f(p.xyz, 1));
        const offset = std.mul(corner, camera.$.splatSize);
        const offsetPos = d.vec4f(
          std.add(viewPos.xy, offset),
          viewPos.z,
          viewPos.w
        );
        return {
          position: std.mul(camera.$.proj, offsetPos),
          color: colors.$[input.instanceIndex],
          uv: corner,
        };
      });

      const fragmentMain = tgpu.fragmentFn({
        in: { color: d.vec4f, uv: d.vec2f },
        out: d.vec4f,
      })((input) => {
        'use gpu';
        if (std.dot(input.uv, input.uv) > 1) {
          std.discard();
        }
        return input.color;
      });

      const depthTexture = device.createTexture({
        size: [width, height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      const depthView = depthTexture.createView();

      const pipeline = root.createRenderPipeline({
        vertex: vertexMain,
        fragment: fragmentMain,
        targets: { format },
        depthStencil: {
          format: 'depth24plus',
          depthWriteEnabled: true,
          depthCompare: 'less',
        },
      });

      const proj = new Float32Array(16);
      const view = new Float32Array(16);
      mat4.perspective(
        Math.PI / 3,
        width / height,
        cloud.radius * 0.02,
        cloud.radius * 40,
        proj
      );
      const [cx, cy, cz] = cloud.center;
      const splatSize = cloud.radius * SPLAT_SCALE;

      let raf = 0;
      let last = performance.now();
      const frame = () => {
        if (stop) return;
        const now = performance.now();
        const dt = Math.min((now - last) / 1000, 0.1);
        last = now;

        if (!interacted.current) {
          orbit.current.yaw += AUTO_ROTATE_SPEED * dt;
        } else if (!dragging.current) {
          orbit.current.yaw += velocity.current.yaw * dt;
          orbit.current.pitch = Math.min(
            1.5,
            Math.max(-1.5, orbit.current.pitch + velocity.current.pitch * dt)
          );
          const decay = Math.exp(-INERTIA_DECAY * dt);
          velocity.current.yaw *= decay;
          velocity.current.pitch *= decay;
        }

        const { yaw, pitch, dist } = orbit.current;
        const r = dist * cloud.radius;
        const eye = [
          cx + r * Math.cos(pitch) * Math.sin(yaw),
          cy + r * Math.sin(pitch),
          cz - r * Math.cos(pitch) * Math.cos(yaw),
        ];
        // The model's camera space is +y down; use a flipped up vector so the
        // scene appears upright.
        mat4.lookAt(eye, [cx, cy, cz], [0, -1, 0], view);
        camera.write({ view, proj, splatSize });

        pipeline
          .withColorAttachment({
            view: context,
            clearValue: [0.02, 0.02, 0.04, 1],
          })
          .withDepthStencilAttachment({
            view: depthView,
            depthClearValue: 1,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
          })
          .draw(6, cloud.count);
        context.present();
        raf = requestAnimationFrame(frame);
      };
      raf = requestAnimationFrame(frame);

      cleanup = () => {
        cancelAnimationFrame(raf);
        depthTexture.destroy();
        root.destroy();
      };
    })();

    return () => {
      stop = true;
      cleanup?.();
    };
  }, [cloud, canvasRef]);

  const pan = Gesture.Pan()
    .runOnJS(true)
    .onStart(() => {
      interacted.current = true;
      dragging.current = true;
      velocity.current = { yaw: 0, pitch: 0 };
      gestureStart.current = { ...orbit.current };
    })
    .onUpdate((e) => {
      orbit.current.yaw = gestureStart.current.yaw - e.translationX * 0.01;
      orbit.current.pitch = Math.min(
        1.5,
        Math.max(-1.5, gestureStart.current.pitch - e.translationY * 0.01)
      );
    })
    .onEnd((e) => {
      dragging.current = false;
      velocity.current = {
        yaw: -e.velocityX * 0.01,
        pitch: -e.velocityY * 0.01,
      };
    });

  const pinch = Gesture.Pinch()
    .runOnJS(true)
    .onStart(() => {
      interacted.current = true;
      gestureStart.current = { ...orbit.current };
    })
    .onUpdate((e) => {
      orbit.current.dist = Math.min(
        10,
        Math.max(0.3, gestureStart.current.dist / e.scale)
      );
    });

  return (
    <GestureDetector gesture={Gesture.Simultaneous(pan, pinch)}>
      <View style={styles.container}>
        <Canvas ref={canvasRef} style={styles.canvas} />
      </View>
    </GestureDetector>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  canvas: { flex: 1 },
});
