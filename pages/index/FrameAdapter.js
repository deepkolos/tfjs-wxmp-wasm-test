export class FrameAdapter {
  constructor(maxProcessFrame = Number.MAX_SAFE_INTEGER, frameGap = 30) {
    this.frameGap = frameGap;
    this.maxProcessFrame = maxProcessFrame;
    this.reset()
  }

  onProcessFrame(cb) {
    this.frameProcesser = cb;
  }

  onMaxFrame(cb) {
    this.maxFrameCB = cb;
  }

  reset() {
    this.currGap = 0
    this.frameNum = 0
    this.processFrameNum = 0
    this.lastFrameDone = true
  }

  async triggerFrame(frame) {
    if (this.frameProcesser && this.processFrameNum < this.maxProcessFrame && this.lastFrameDone) {
      // console.log('triggerFrame', this.frameNum, Date.now())

      if (this.frameNum === 0 || this.lastProcessTime === undefined) {
        await this.processFrame(frame);
      } else {
        const gap = Math.max(Math.round(this.lastProcessTime / this.frameGap), 1);
        this.currGap = gap
        if (this.frameNum >= gap) {
          await this.processFrame(frame);
          this.frameNum = 0;
        }
      }

      this.frameNum++;
    }

    if (this.processFrameNum === this.maxProcessFrame) {
      this.processFrameNum++
      this.maxFrameCB && this.maxFrameCB()
    }
  }

  async processFrame(frame) {
    if (this.frameProcesser) {
      this.lastFrameDone = false
      const t = Date.now();
      // console.log('processFrame', this.frameNum, t)
      await this.frameProcesser(frame);
      this.lastFrameDone = true
      this.lastProcessTime = Date.now() - t;
    }
    this.processFrameNum++;
  }
}
