import { Platform } from 'react-native';

// Must match the exported graph: da3_small_{VIEWS}v_{HEIGHT}x{WIDTH}_*.pte
export const VIEWS = 8;
export const HEIGHT = 560;
export const WIDTH = 420;

// Where the .pte files live. Defaults to the published Hugging Face repo;
// set EXPO_PUBLIC_MODEL_BASE in .env to serve your own exports (see .env.example).
const MODEL_BASE =
  process.env.EXPO_PUBLIC_MODEL_BASE ??
  'https://huggingface.co/nklockiewicz/react-native-executorch-demo-models/resolve/main/da3-scanner';

export const MODEL_SOURCE = Platform.select({
  ios: `${MODEL_BASE}/coreml/da3_small_${VIEWS}v_${HEIGHT}x${WIDTH}_coreml_fp32.pte`,
  default: `${MODEL_BASE}/xnnpack/da3_small_${VIEWS}v_${HEIGHT}x${WIDTH}_xnnpack.pte`,
});
