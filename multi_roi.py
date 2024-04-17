#!/usr/bin/env python3

import cv2
import depthai as dai
import math
import numpy as np

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
colorLeft = pipeline.create(dai.node.ColorCamera)
colorRight = pipeline.create(dai.node.ColorCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

xoutRgb.setStreamName("rgb")
xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# Properties
stereoCamsRes = dai.ColorCameraProperties.SensorResolution.THE_1200_P
colorLeft.setResolution(stereoCamsRes)
colorLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
colorLeft.setIspScale(1, 3)

colorRight.setResolution(stereoCamsRes)
colorRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
colorRight.setIspScale(1, 3)

# stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# stereo.setLeftRightCheck(True)
# stereo.setSubpixel(True)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
stereo.initialConfig.setDepthUnit(stereo.initialConfig.AlgorithmControl.DepthUnit.MILLIMETER)

stereo.initialConfig.setExtendedDisparity(True)
stereo.initialConfig.setSubpixel(True)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
# stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
# stereo.initialConfig.setConfidenceThreshold(168)
stereo.initialConfig.setLeftRightCheckThreshold(7)
stereo.initialConfig.setSubpixelFractionalBits(3)
stereo.initialConfig.setNumInvalidateEdgePixels(4)
# stereo.initialConfig.setDisparityShift(5)

config = stereo.initialConfig.get()
config.postProcessing.temporalFilter.enable = True
config.postProcessing.temporalFilter.persistencyMode = dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_3
config.censusTransform.enableMeanMode = True
config.costMatching.linearEquationParameters.alpha = 5
config.costMatching.linearEquationParameters.beta = 1
config.costMatching.linearEquationParameters.threshold = 79
config.costAggregation.horizontalPenaltyCostP1 = 2
config.costAggregation.horizontalPenaltyCostP2 = 235
config.postProcessing.temporalFilter.alpha = 0.1
config.postProcessing.temporalFilter.delta = 3
config.postProcessing.thresholdFilter.minRange = 0
config.postProcessing.thresholdFilter.maxRange = 1000
config.postProcessing.speckleFilter.enable = True
config.postProcessing.speckleFilter.speckleRange = 8
config.postProcessing.decimationFilter.decimationFactor = 1

stereo.initialConfig.set(config)


# stereo.setDepthAlign(align=dai.StereoDepthConfig.AlgorithmControl.DepthAlign.CENTER)
stereo.setDepthAlign(camera=dai.CameraBoardSocket.CAM_C)
# stereo.setRectification(True)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)

# Create 10 ROIs
for i in range(10):
    config = dai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = 100
    config.depthThresholds.upperThreshold = 1000
    config.roi = dai.Rect(dai.Point2f(i*0.1, 0.45), dai.Point2f((i+1)*0.1, 0.55))
    spatialLocationCalculator.initialConfig.addROI(config)

# Linking
colorLeft.isp.link(stereo.left)
colorRight.isp.link(stereo.right)
colorRight.isp.link(xoutRgb.input)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # device.setIrLaserDotProjectorBrightness(1000)

    # Output queue will be used to get the depth frames from the outputs defined above
    rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    color = (0,200,40)
    fontType = cv2.FONT_HERSHEY_TRIPLEX

    while True:
        inRgb = rgbQueue.get()
        rgbFrame = inRgb.getCvFrame()
        inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived
        depthFrame = inDepth.getFrame() # depthFrame values are in millimeters

        depth_downscaled = depthFrame[::4]
        if np.all(depth_downscaled == 0):
            min_depth = 0  # Set a default minimum depth value when all elements are zero
        else:
            min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
        max_depth = np.percentile(depth_downscaled, 99)
        depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

        spatialData = spatialCalcQueue.get().getSpatialLocations()
        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])

            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            coords = depthData.spatialCoordinates
            distance = math.sqrt(coords.x ** 2 + coords.y ** 2 + coords.z ** 2)

            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, thickness=2)
            cv2.putText(depthFrameColor, "{:.2f}m".format(distance/1000), (xmin + 10, ymin + 20), fontType, 0.5, color)

            cv2.rectangle(rgbFrame, (xmin, ymin), (xmax, ymax), color, thickness=2)
            cv2.putText(rgbFrame, "{:.2f}m".format(distance/1000), (xmin + 10, ymin + 20), fontType, 0.5, color)

        # Show the frame
        cv2.imshow("depth", depthFrameColor)
        cv2.imshow("rgb", rgbFrame)

        if cv2.waitKey(1) == ord('q'):
            break