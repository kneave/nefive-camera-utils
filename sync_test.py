import depthai as dai
import numpy as np
import cv2
from datetime import timedelta

pipeline = dai.Pipeline()

colorLeft = pipeline.create(dai.node.ColorCamera)
colorRight = pipeline.create(dai.node.ColorCamera)
color = pipeline.create(dai.node.ColorCamera)
stereo = pipeline.create(dai.node.StereoDepth)
sync = pipeline.create(dai.node.Sync)

xoutGrp = pipeline.create(dai.node.XLinkOut)

xoutGrp.setStreamName("xout")

colorLeft.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
colorLeft.setCamera("left")
colorLeft.setIspScale(1, 3)
colorRight.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
colorRight.setCamera("right")
colorRight.setIspScale(1, 3)

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
stereo.initialConfig.setDisparityShift(5)

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
config.costMatching.disparityWidth = dai.StereoDepthConfig.CostMatching.DisparityWidth.DISPARITY_64

stereo.initialConfig.set(config)


stereo.setDepthAlign(align=dai.StereoDepthConfig.AlgorithmControl.DepthAlign.CENTER)
stereo.setRectification(True)

color.setCamera("color")
color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
color.setIspScale(1, 3)

sync.setSyncThreshold(timedelta(milliseconds=50))

colorLeft.isp.link(stereo.left)
colorRight.isp.link(stereo.right)

stereo.disparity.link(sync.inputs["disparity"])
color.video.link(sync.inputs["video"])
colorRight.isp.link(sync.inputs["right"])

sync.out.link(xoutGrp.input)

disparityMultiplier = 255.0 / (stereo.initialConfig.getMaxDisparity() + 5)
with dai.Device(pipeline) as device:
    queue = device.getOutputQueue("xout", 10, False)
    while True:
        msgGrp = queue.get()
        for name, msg in msgGrp:
            frame = msg.getCvFrame()
            if name == "disparity":
                frame = (frame * disparityMultiplier).astype(np.uint8)
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
            cv2.imshow(name, frame)
        if cv2.waitKey(1) == ord("q"):
            break