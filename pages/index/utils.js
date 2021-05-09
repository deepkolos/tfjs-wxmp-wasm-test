"use strict";
exports.__esModule = true;
exports.drawPoses = exports.drawBoundingBox = exports.drawKeypoints = exports.drawSkeleton = exports.drawSegment = exports.drawPoint = void 0;
var posenet = require("@tensorflow-models/posenet");
var color = 'aqua';
var boundingBoxColor = 'red';
var lineWidth = 2;
function toTuple(_a) {
    var y = _a.y, x = _a.x;
    return [y, x];
}
function drawPoint(ctx, y, x, r, color) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
}
exports.drawPoint = drawPoint;
/**
 * Draws a line on a canvas, i.e. a joint
 */
function drawSegment(_a, _b, color, scale, ctx) {
    var ay = _a[0], ax = _a[1];
    var by = _b[0], bx = _b[1];
    ctx.beginPath();
    ctx.moveTo(ax * scale, ay * scale);
    ctx.lineTo(bx * scale, by * scale);
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = color;
    ctx.stroke();
}
exports.drawSegment = drawSegment;
/**
 * Draws a pose skeleton by looking up all adjacent keypoints/joints
 */
// tslint:disable-next-line:no-any
function drawSkeleton(keypoints, minConfidence, ctx, scale) {
    if (scale === void 0) { scale = 1; }
    var adjacentKeyPoints = posenet.getAdjacentKeyPoints(keypoints, minConfidence);
    // tslint:disable-next-line:no-any
    adjacentKeyPoints.forEach(function (keypoints) {
        drawSegment(toTuple(keypoints[0].position), toTuple(keypoints[1].position), color, scale, ctx);
    });
}
exports.drawSkeleton = drawSkeleton;
/**
 * Draw pose keypoints onto a canvas
 */
// tslint:disable-next-line:no-any
function drawKeypoints(keypoints, minConfidence, ctx, scale) {
    if (scale === void 0) { scale = 1; }
    for (var i = 0; i < keypoints.length; i++) {
        var keypoint = keypoints[i];
        if (keypoint.score < minConfidence) {
            continue;
        }
        var _a = keypoint.position, y = _a.y, x = _a.x;
        drawPoint(ctx, y * scale, x * scale, 3, color);
    }
}
exports.drawKeypoints = drawKeypoints;
/**
 * Draw the bounding box of a pose. For example, for a whole person standing
 * in an image, the bounding box will begin at the nose and extend to one of
 * ankles
 */
// tslint:disable-next-line:no-any
function drawBoundingBox(keypoints, ctx) {
    var boundingBox = posenet.getBoundingBox(keypoints);
    ctx.rect(boundingBox.minX, boundingBox.minY, boundingBox.maxX - boundingBox.minX, boundingBox.maxY - boundingBox.minY);
    ctx.strokeStyle = boundingBoxColor;
    ctx.stroke();
}
exports.drawBoundingBox = drawBoundingBox;
function drawPoses(poses, ctx) {
    var minPoseConfidence = 0.3;
    var minPartConfidence = 0.3;
    // For each pose (i.e. person) detected in an image, loop through the poses
    // and draw the resulting skeleton and keypoints if over certain confidence
    // scores
    poses.forEach(function (_a) {
        var score = _a.score, keypoints = _a.keypoints;
        if (score >= minPoseConfidence) {
            drawKeypoints(keypoints, minPartConfidence, ctx);
            drawSkeleton(keypoints, minPartConfidence, ctx);
        }
    });
    return poses;
}
exports.drawPoses = drawPoses;
