import { Platform } from 'react-native';

// Must match the exported graph: da3_{MODEL}_{VIEWS}v_{HEIGHT}x{WIDTH}_*.pte
export const MODEL: 'small' | 'base' = 'base';
// iOS normally runs CoreML; xnnpack forces CPU inference (slower, less memory).
const IOS_BACKEND: 'coreml' | 'xnnpack' = 'coreml';
// base only fits on the phone as fp16; small ships as fp32.
const COREML_PRECISION = MODEL === 'base' ? 'fp16_gpu' : 'fp32';
export const VIEWS = 8;
export const HEIGHT = 560;
export const WIDTH = 420;

// Where the .pte files live. Defaults to the published Hugging Face repo;
// set EXPO_PUBLIC_MODEL_BASE in .env to serve your own exports (see .env.example).
const MODEL_BASE =
  process.env.EXPO_PUBLIC_MODEL_BASE ??
  'https://huggingface.co/nklockiewicz/react-native-executorch-demo-models/resolve/main/da3-scanner';

export const MODEL_SOURCE = Platform.select({
  ios:
    IOS_BACKEND === 'coreml'
      ? `${MODEL_BASE}/coreml/da3_${MODEL}_${VIEWS}v_${HEIGHT}x${WIDTH}_coreml_${COREML_PRECISION}.pte`
      : `${MODEL_BASE}/xnnpack/da3_${MODEL}_${VIEWS}v_${HEIGHT}x${WIDTH}_xnnpack.pte`,
  // Android always uses the small model: base needs more memory than most
  // devices allow. See the Android section of the README before judging speed.
  default: `${MODEL_BASE}/xnnpack/da3_small_${VIEWS}v_${HEIGHT}x${WIDTH}_xnnpack.pte`,
});
