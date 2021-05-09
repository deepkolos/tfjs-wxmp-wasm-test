# tfjs-wxmp-wasm-test

a test demo for tensorflow wechat wasm support PR. tested on android wechat 8.0.3 MI 8 FPS around 5.

related issue: https://github.com/tensorflow/tfjs/issues/5021

### run

use `微信开发工具` open this project, change `appid` to your own test `appid`, disable `urlCheck`, then preview

`@tensorflow/tfjs-backend-wasm` is linked to local package which add fixWechatWebAssembly patch