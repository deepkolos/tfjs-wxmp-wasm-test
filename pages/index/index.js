import * as tf from '@tensorflow/tfjs-core'
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import { FrameAdapter } from './FrameAdapter'
import { fetchFunc } from 'fetch-wechat'
import * as posenet from '@tensorflow-models/posenet'
import { drawPoses } from './utils'

const { configPlugin } = requirePlugin('tfjsPlugin')

configPlugin({
  tf,
  fetchFunc: fetchFunc(),
})

setWasmPaths(
  {
    'tfjs-backend-wasm.wasm': '/tfjs-backend-wasm.wasm',
    'tfjs-backend-wasm-simd.wasm': '/tfjs-backend-wasm.wasm',
    'tfjs-backend-wasm-threaded-simd.wasm': '/tfjs-backend-wasm.wasm',
  },
  true,
)

function getNode(id, ctx) {
  return new Promise(resolve => {
    wx.createSelectorQuery().in(ctx).select(id).fields({ node: true, rect: true }).exec(resolve);
  });
}

const POSENET_URL =
  'https://www.gstaticcnapps.cn/tfjs-models/savedmodel/posenet/mobilenet/float/050/model-stride16.json';

Page({
  data: { FPS: '', backend: '' },

  async onReady() {
    await tf.setBackend('wasm');
    console.log('setBackend wasm');
    this.setData({ backend: tf.getBackend() })
    const model = await posenet.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      inputResolution: 193,
      multiplier: 0.5,
      modelUrl: POSENET_URL
    });
    console.log('posenet load done');

    const [{ node: canvas }] = await getNode('#canvas', this);
    const cameraCtx = wx.createCameraContext();
    const frameAdapter = new FrameAdapter();
    const cameraListener = cameraCtx.onCameraFrame(frameAdapter.triggerFrame.bind(frameAdapter));
    const ctx = canvas.getContext('2d');

    function drawFrame(frame) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      canvas.width = frame.width;
      canvas.height = frame.height;
      // @ts-ignore
      const imageData = canvas.createImageData(
        new Uint8Array(frame.data),
        frame.width,
        frame.height,
      );
      ctx.putImageData(imageData, 0, 0);
    }

    frameAdapter.onProcessFrame(async frame => {
      const t = Date.now()
      const video = {
        width: frame.width,
        height: frame.height,
        data: new Uint8Array(frame.data),
      }
      const prediction = await model.estimateSinglePose(video, { flipHorizontal: false })

      drawFrame(frame);
      drawPoses([prediction], ctx);

      // if (isAndroid) await new Promise(resolve => canvas.requestAnimationFrame(resolve));
      this.setData({ FPS: (1000 / (Date.now() - t)).toFixed(2) });
    });

    cameraListener.start()
    console.log('init done');
  }
})
