#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import argparse
from datetime import timedelta

# Weights to use when blending depth/rgb image (should equal 1.0)
rgbWeight = 0.4
depthWeight = 0.6

parser = argparse.ArgumentParser()
parser.add_argument('-alpha', type=float, default=None, help="Alpha scaling parameter to increase float. [0,1] valid interval.")
args = parser.parse_args()
alpha = args.alpha

def updateBlendWeights(percent_rgb):
    """
    Update the rgb and depth weights used to blend depth/rgb image

    @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
    """
    global depthWeight
    global rgbWeight
    rgbWeight = float(percent_rgb)/100.0
    depthWeight = 1.0 - rgbWeight


fps = 30
# resolution = 960, 600

# Create pipeline
pipeline = dai.Pipeline()
device = dai.Device()
queueNames = []

# Define sources and outputs
center = pipeline.create(dai.node.ColorCamera)
left = pipeline.create(dai.node.ColorCamera)
right = pipeline.create(dai.node.ColorCamera)
stereo = pipeline.create(dai.node.StereoDepth)
sync = pipeline.create(dai.node.Sync)

centerOut = pipeline.create(dai.node.XLinkOut)
leftOut = pipeline.create(dai.node.XLinkOut)
rightOut = pipeline.create(dai.node.XLinkOut)
disparityOut = pipeline.create(dai.node.XLinkOut)
rectOut = pipeline.create(dai.node.XLinkOut)
xoutGrp = pipeline.create(dai.node.XLinkOut)
xoutGrp.setStreamName("xout")

sync.setSyncThreshold(timedelta(milliseconds=50))

centerOut.setStreamName("center")
queueNames.append("center")
leftOut.setStreamName("left")
queueNames.append("left")
rightOut.setStreamName("right")
queueNames.append("right")
disparityOut.setStreamName("disp")
queueNames.append("disp")
rectOut.setStreamName("rect")
queueNames.append("rect")

# stereo.setOutputSize(resolution[0], resolution[1])

center.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
center.setIspScale(1, 3)
center.setBoardSocket(dai.CameraBoardSocket.CAM_A)
center.setFps(fps)
# center.setSize(1920, 1080)

# For now, RGB needs fixed focus to properly align with depth.
# This value was used during calibration
try:
    calibData = device.readCalibration2()
    lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
    if lensPosition:
        center.initialControl.setManualFocus(lensPosition)
except:
    raise

left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
left.setIspScale(1, 2)
left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
left.setFps(fps)
left.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
# left.setSize(1920, 1200)
# left.setVideoSize(960, 600)

right.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
right.setIspScale(1, 2)
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
right.setFps(fps)
right.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
# right.setSize(1920, 1200)
# right.setVideoSize(960, 600)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# LR-check is required for depth alignment
stereo.setLeftRightCheck(True)
# stereo.setDepthAlign(dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_RIGHT)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_C)
# stereo.setDepthAlign()

# Linking
center.isp.link(centerOut.input)
left.isp.link(stereo.left)
right.isp.link(stereo.right)
left.isp.link(leftOut.input)
right.isp.link(rightOut.input)
stereo.disparity.link(sync.inputs["disparity"])
# center.isp.link(sync.inputs["center"])
stereo.rectifiedRight.link(rectOut.input)

sync.out.link(xoutGrp.input)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

stereo.initialConfig.setBilateralFilterSigma(16)
config = stereo.initialConfig.get()
config.postProcessing.speckleFilter.enable = True
config.postProcessing.speckleFilter.speckleRange = 60
config.postProcessing.temporalFilter.enable = False
config.postProcessing.temporalFilter.persistencyMode = dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_3

config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 150  # mm
config.postProcessing.thresholdFilter.maxRange = 1500  # mm
config.postProcessing.decimationFilter.decimationFactor = 1
config.censusTransform.enableMeanMode = True
config.costMatching.linearEquationParameters.alpha = 0
config.costMatching.linearEquationParameters.beta = 2

stereo.initialConfig.setDepthUnit(stereo.initialConfig.AlgorithmControl.DepthUnit.MILLIMETER)

stereo.initialConfig.set(config)


# center.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)
# right.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)
if alpha is not None:
    center.setCalibrationAlpha(alpha)
    stereo.setAlphaScaling(alpha)

# Connect to device and start pipeline
with device:
    device.startPipeline(pipeline)

    frameRgb = None
    frameLeft = None
    frameRight = None
    frameDisp = None
    frameRect = None

    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    centerWindowName = "center"
    leftWindowName = "right"
    rightWindowName = "right"
    depthWindowName = "depth"
    rectWindowName = "rect"
    blendedWindowName = "rgb-depth"
    cv2.namedWindow(centerWindowName)
    cv2.namedWindow(leftWindowName)
    cv2.namedWindow(rightWindowName)
    cv2.namedWindow(depthWindowName)
    cv2.namedWindow(blendedWindowName)
    cv2.namedWindow(rectWindowName)
    cv2.createTrackbar('RGB Weight %', blendedWindowName, int(rgbWeight*100), 100, updateBlendWeights)

    while True:
        latestPacket = {}
        latestPacket["center"] = None
        latestPacket["disp"] = None
        latestPacket["left"] = None
        latestPacket["right"] = None
        latestPacket["rect"] = None

        queueEvents = device.getQueueEvents(("center", "disp", "left", "right", "rect"))
        for queueName in queueEvents:
            packets = device.getOutputQueue("xout").tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]

        if latestPacket["center"] is not None:
            frameCenter = latestPacket["center"]
            cv2.imshow(centerWindowName, frameCenter.getCvFrame())

        if latestPacket["left"] is not None:
            frameLeft = latestPacket["left"]
            cv2.imshow(leftWindowName, frameLeft.getCvFrame()) 

        if latestPacket["right"] is not None:
            frameRight = latestPacket["right"]
            cv2.imshow(rightWindowName, frameRight.getCvFrame())

        if latestPacket["rect"] is not None:
            frameRect = latestPacket["rect"].getCvFrame()
            cv2.imshow(rectWindowName, frameRect)

        if latestPacket["disp"] is not None:
            frameDisp = latestPacket["disp"].getFrame()
            maxDisparity = stereo.initialConfig.getMaxDisparity()
            # Optional, extend range 0..95 -> 0..255, for a better visualisation
            if 1: frameDisp = (frameDisp * 255. / maxDisparity).astype(np.uint8)
            # Optional, apply false colorization
            if 1: frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_HOT)
            frameDisp = np.ascontiguousarray(frameDisp)
            cv2.imshow(depthWindowName, frameDisp)

        if((frameCenter is not None) and (frameLeft is not None) and (frameRight is not None)):
            print("=====================")
            # print(f"Center timestamp: {frameCenter.getTimestamp()}")
            # print(f"Left timestamp: {frameLeft.getTimestamp()}")
            # print(f"Right timestamp: {frameRight.getTimestamp()}")

            print("Right - Left:\t{:.6f}".format(frameRight.getTimestamp().total_seconds() - frameLeft.getTimestamp().total_seconds()))
            print("Right - Center\t{:.6f}".format(frameRight.getTimestamp().total_seconds() - frameCenter.getTimestamp().total_seconds()))


        # # Blend when both received
        # if frameCenter is not None and frameDisp is not None:
        #     # Need to have both frames in BGR format before blending
        #     if len(frameDisp.shape) < 3:
        #         frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)
        #     blended = cv2.addWeighted(frameCenter, rgbWeight, frameDisp, depthWeight, 0)
        #     cv2.imshow(blendedWindowName, blended)
        #     frameRgb = None
        #     frameDisp = None

        if cv2.waitKey(1) == ord('q'):
            break
