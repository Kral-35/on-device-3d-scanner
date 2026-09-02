import React, { useCallback, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Image,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import { ScalarType, useExecutorchModule } from 'react-native-executorch';
import { StatusBar } from 'expo-status-bar';
import { HEIGHT, MODEL_SOURCE, VIEWS, WIDTH } from './src/config';
import { buildInput } from './src/pipeline/preprocess';
import { buildPointCloud, PointCloud } from './src/pipeline/pointcloud';
import { PointCloudView } from './src/renderer/PointCloudView';

type Phase = 'capture' | 'processing' | 'view';

export default function App() {
  const [phase, setPhase] = useState<Phase>('capture');
  const [photos, setPhotos] = useState<string[]>([]);
  const [status, setStatus] = useState('');
  const [cloud, setCloud] = useState<PointCloud | null>(null);
  const [inferenceSecs, setInferenceSecs] = useState<string | null>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);
  const model = useExecutorchModule({ modelSource: MODEL_SOURCE });

  const runPipeline = useCallback(
    async (uris: string[]) => {
      setPhase('processing');
      try {
        setStatus('Preparing photos');
        const { input, rgbs } = await buildInput(uris);
        setStatus('Running Depth Anything 3');
        const t0 = Date.now();
        const outputs = await model.forward([
          {
            dataPtr: input,
            sizes: [1, VIEWS, 3, HEIGHT, WIDTH],
            scalarType: ScalarType.FLOAT,
          },
        ]);
        const dt = ((Date.now() - t0) / 1000).toFixed(1);
        setInferenceSecs(dt);
        setStatus(`Inference ${dt}s, fusing points`);
        const [depth, conf, poseEnc] = outputs.map(
          (o) => new Float32Array(o.dataPtr as ArrayBuffer)
        );
        const pc = buildPointCloud(depth, conf, rgbs, poseEnc);
        console.log(
          `inference ${dt}s, ${pc.count} points, radius ${pc.radius.toFixed(3)}`
        );
        setCloud(pc);
        setPhase('view');
      } catch (e) {
        console.error(e);
        setStatus(`Failed: ${e}`);
        setTimeout(() => setPhase('capture'), 3000);
      }
    },
    [model]
  );

  const takePhoto = useCallback(async () => {
    const photo = await cameraRef.current?.takePictureAsync({ quality: 0.9 });
    if (!photo) return;
    const next = [...photos, photo.uri];
    setPhotos(next);
    if (next.length === VIEWS) runPipeline(next);
  }, [photos, runPipeline]);

  const pickFromLibrary = useCallback(async () => {
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: 'images',
      allowsMultipleSelection: true,
      selectionLimit: VIEWS,
      quality: 1,
    });
    if (res.canceled || res.assets.length !== VIEWS) return;
    runPipeline(res.assets.map((a) => a.uri));
  }, [runPipeline]);

  if (!permission?.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.text}>Camera access is needed to scan.</Text>
        <Pressable style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant camera access</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <GestureHandlerRootView style={styles.root}>
      <StatusBar style="light" />
      {phase === 'capture' && (
        <View style={styles.root}>
          <CameraView ref={cameraRef} style={styles.camera} />
          <View style={styles.overlay}>
            <Text style={styles.counter}>
              {photos.length} / {VIEWS}
            </Text>
            <Text style={styles.hint}>
              Circle the object, one photo per step
            </Text>
            <View style={styles.thumbRow}>
              {photos.map((uri) => (
                <Image key={uri} source={{ uri }} style={styles.thumb} />
              ))}
            </View>
            <View style={styles.controls}>
              <Pressable
                style={styles.smallButton}
                onPress={() => setPhotos([])}
              >
                <Text style={styles.buttonText}>Reset</Text>
              </Pressable>
              <Pressable
                style={[styles.shutter, !model.isReady && styles.disabled]}
                disabled={!model.isReady}
                onPress={takePhoto}
              />
              <Pressable style={styles.smallButton} onPress={pickFromLibrary}>
                <Text style={styles.buttonText}>Library</Text>
              </Pressable>
            </View>
            {!model.isReady && (
              <Text style={styles.hint}>
                Downloading model {Math.round(model.downloadProgress * 100)}%
              </Text>
            )}
          </View>
        </View>
      )}
      {phase === 'processing' && (
        <View style={styles.center}>
          <ActivityIndicator size="large" color="#fff" />
          <Text style={styles.text}>{status}</Text>
        </View>
      )}
      {phase === 'view' && cloud && (
        <View style={styles.root}>
          <PointCloudView cloud={cloud} />
          {inferenceSecs && (
            <Text style={styles.statsOverlay}>
              {inferenceSecs}s on-device · {(cloud.count / 1000).toFixed(0)}k
              points
            </Text>
          )}
          <Pressable
            style={[styles.button, styles.newScan]}
            onPress={() => {
              setPhotos([]);
              setCloud(null);
              setPhase('capture');
            }}
          >
            <Text style={styles.buttonText}>New scan</Text>
          </Pressable>
        </View>
      )}
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  center: {
    flex: 1,
    backgroundColor: '#000',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
  },
  camera: { flex: 1 },
  overlay: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    padding: 24,
    alignItems: 'center',
    gap: 12,
  },
  counter: { color: '#fff', fontSize: 32, fontWeight: '700' },
  hint: { color: '#ccc', fontSize: 14 },
  thumbRow: { flexDirection: 'row', gap: 4, flexWrap: 'wrap' },
  thumb: { width: 36, height: 48, borderRadius: 4 },
  controls: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 32,
    marginTop: 8,
  },
  shutter: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: '#fff',
    borderWidth: 4,
    borderColor: '#888',
  },
  disabled: { opacity: 0.3 },
  smallButton: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 8,
    backgroundColor: '#333',
  },
  button: {
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderRadius: 10,
    backgroundColor: '#2563eb',
  },
  newScan: {
    position: 'absolute',
    bottom: 48,
    alignSelf: 'center',
  },
  statsOverlay: {
    position: 'absolute',
    top: 64,
    alignSelf: 'center',
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
    backgroundColor: '#0008',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    overflow: 'hidden',
  },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  text: { color: '#fff', fontSize: 16, textAlign: 'center', padding: 16 },
});
