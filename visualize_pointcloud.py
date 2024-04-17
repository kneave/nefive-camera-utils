import depthai as dai
from time import sleep
import numpy as np
import cv2
import time
import sys
try:
    import open3d as o3d
except ImportError:
    sys.exit("Critical dependency missing: Open3D. Please install it using the command: '{} -m pip install open3d' and then rerun the script.".format(sys.executable))

FPS = 30

class FPSCounter:
    def __init__(self):
        self.frameCount = 0
        self.fps = 0
        self.startTime = time.time()

    def tick(self):
        self.frameCount += 1
        if self.frameCount % 10 == 0:
            elapsedTime = time.time() - self.startTime
            self.fps = self.frameCount / elapsedTime
            self.frameCount = 0
            self.startTime = time.time()
        return self.fps

pipeline = dai.Pipeline()
camRgb = pipeline.create(dai.node.ColorCamera)
colorLeft = pipeline.create(dai.node.ColorCamera)
colorRight = pipeline.create(dai.node.ColorCamera)
stereo = pipeline.create(dai.node.StereoDepth)
pointcloud = pipeline.create(dai.node.PointCloud)
sync = pipeline.create(dai.node.Sync)
xOut = pipeline.create(dai.node.XLinkOut)
xOut.input.setBlocking(False)

camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setIspScale(1,3)
camRgb.setFps(FPS)

colorLeft.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
colorLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
colorLeft.setIspScale(1,4)
colorLeft.setFps(FPS)
colorLeft.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)

colorRight.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
colorRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
colorRight.setIspScale(1,4)
colorRight.setFps(FPS)
colorRight.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)


# manipLeft = pipeline.create(dai.node.ImageManip)
# manipLeft.initialConfig.setFrameType(dai.RawImgFrame.Type.GRAY8)
# colorLeft.isp.link(manipLeft.inputImage)

# manipRight = pipeline.create(dai.node.ImageManip)
# manipRight.initialConfig.setFrameType(dai.RawImgFrame.Type.GRAY8)
# colorRight.isp.link(manipRight.inputImage)

"""
In-place post-processing configuration for a stereo depth node
The best combo of filters is application specific. Hard to say there is a one size fits all.
They also are not free. Even though they happen on device, you pay a penalty in fps.
"""

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
stereo.initialConfig.setDepthUnit(stereo.initialConfig.AlgorithmControl.DepthUnit.MILLIMETER)

stereo.initialConfig.setExtendedDisparity(True)
stereo.initialConfig.setSubpixel(True)
# stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.initialConfig.setConfidenceThreshold(168)
stereo.initialConfig.setLeftRightCheckThreshold(7)
stereo.initialConfig.setSubpixelFractionalBits(3)
stereo.initialConfig.setNumInvalidateEdgePixels(4)

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
config.postProcessing.thresholdFilter.maxRange = 650
config.postProcessing.speckleFilter.enable = True
config.postProcessing.speckleFilter.speckleRange = 8
config.postProcessing.decimationFilter.decimationFactor = 1
config.costMatching.disparityWidth = dai.StereoDepthConfig.CostMatching.DisparityWidth.DISPARITY_64

stereo.initialConfig.set(config)


# stereo.setDepthAlign(align=dai.StereoDepthConfig.AlgorithmControl.DepthAlign.CENTER)
stereo.setDepthAlign(camera=dai.CameraBoardSocket.CAM_C)
# stereo.setRectification(True)
# pointcloud.initialConfig.setSparse(True)

# manipLeft.out.link(depth.left)
# manipRight.out.link(depth.right)
colorLeft.isp.link(stereo.left)
colorRight.isp.link(stereo.right)
stereo.depth.link(pointcloud.inputDepth)
# camRgb.isp.link(sync.inputs["rgb"])
colorRight.isp.link(sync.inputs["rgb"])
pointcloud.outputPointCloud.link(sync.inputs["pcl"])
# camRgb.isp.link(xOut.inputs["rgb"])
# pointcloud.outputPointCloud.link(xOut.inputs["pcl"])
sync.out.link(xOut.input)
xOut.setStreamName("out")



with dai.Device(pipeline) as device:
    isRunning = True
    def key_callback(vis, action, mods):
        global isRunning
        if action == 0:
            isRunning = False

    q = device.getOutputQueue(name="out", maxSize=4, blocking=False)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.register_key_action_callback(81, key_callback)
    pcd = o3d.geometry.PointCloud()
    coordinateFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0,0,0])
    vis.add_geometry(coordinateFrame)

    first = True
    fpsCounter = FPSCounter()
    while isRunning:
        inMessage = q.get()
        inColor = inMessage["rgb"]
        inPointCloud = inMessage["pcl"]
        cvColorFrame = inColor.getCvFrame()
        # Convert the frame to RGB
        cvRGBFrame = cv2.cvtColor(cvColorFrame, cv2.COLOR_BGR2RGB)
        fps = fpsCounter.tick()
        # Display the FPS on the frame
        cv2.putText(cvColorFrame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("color", cvColorFrame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if inPointCloud:
            t_before = time.time()
            # inPointCloud.setMaxX(1.75)
            # inPointCloud.setMaxY(1.75)
            # inPointCloud.setMaxZ(1.75)

            # print(inPointCloud.getMaxX(), inPointCloud.getMaxY(), inPointCloud.getMaxZ())

            points = inPointCloud.getPoints().astype(np.float64)
            pcd.points = o3d.utility.Vector3dVector(points)
            # pcd = pcd.voxel_down_sample(voxel_size=0.01)
            colors = (cvRGBFrame.reshape(-1, 3) / 255.0).astype(np.float64)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            # pcd = pcd.remove_statistical_outlier(30, 0.1)[0]
            
            if first:
                vis.add_geometry(pcd)
                first = False
            else:
                vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
    vis.destroy_window()
