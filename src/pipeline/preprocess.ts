import {
  AlphaType,
  ColorType,
  Skia,
} from '@shopify/react-native-skia';
import { HEIGHT, VIEWS, WIDTH } from '../config';

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// Decode a photo, center-crop to the model aspect, rasterize to WIDTH x HEIGHT,
// and return both the ImageNet-normalized CHW floats and raw RGB for coloring.
async function loadView(
  uri: string
): Promise<{ chw: Float32Array; rgb: Uint8Array }> {
  const data = await Skia.Data.fromURI(uri);
  const decoded = Skia.Image.MakeImageFromEncoded(data);
  if (!decoded) throw new Error(`Could not decode image: ${uri}`);

  const w = decoded.width();
  const h = decoded.height();
  const targetAr = WIDTH / HEIGHT;
  let sx = 0;
  let sy = 0;
  let sw = w;
  let sh = h;
  if (w / h > targetAr) {
    sw = h * targetAr;
    sx = (w - sw) / 2;
  } else {
    sh = w / targetAr;
    sy = (h - sh) / 2;
  }

  const surface = Skia.Surface.MakeOffscreen(WIDTH, HEIGHT);
  if (!surface) throw new Error('Could not create surface');
  const canvas = surface.getCanvas();
  canvas.drawImageRect(
    decoded,
    { x: sx, y: sy, width: sw, height: sh },
    { x: 0, y: 0, width: WIDTH, height: HEIGHT },
    Skia.Paint()
  );
  surface.flush();
  const image = surface.makeImageSnapshot().makeNonTextureImage();
  if (!image) throw new Error('Could not snapshot surface');
  const pixels = image.readPixels(0, 0, {
    width: WIDTH,
    height: HEIGHT,
    colorType: ColorType.RGBA_8888,
    alphaType: AlphaType.Unpremul,
  }) as Uint8Array | null;
  if (!pixels) throw new Error('Could not read pixels');

  const plane = WIDTH * HEIGHT;
  const chw = new Float32Array(3 * plane);
  const rgb = new Uint8Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      chw[c * plane + i] = (pixels[i * 4 + c] / 255 - MEAN[c]) / STD[c];
      rgb[i * 3 + c] = pixels[i * 4 + c];
    }
  }
  decoded.dispose();
  image.dispose();
  surface.dispose();
  return { chw, rgb };
}

// Pack VIEWS photos into the model input (1, VIEWS, 3, HEIGHT, WIDTH).
export async function buildInput(
  uris: string[]
): Promise<{ input: Float32Array; rgbs: Uint8Array[] }> {
  if (uris.length !== VIEWS) {
    throw new Error(`Need exactly ${VIEWS} photos, got ${uris.length}`);
  }
  const plane = 3 * WIDTH * HEIGHT;
  const input = new Float32Array(VIEWS * plane);
  const rgbs: Uint8Array[] = [];
  for (let v = 0; v < VIEWS; v++) {
    const { chw, rgb } = await loadView(uris[v]);
    input.set(chw, v * plane);
    rgbs.push(rgb);
  }
  return { input, rgbs };
}
