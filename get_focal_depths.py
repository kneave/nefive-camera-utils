import depthai as dai
import math

def calculate_focal_depths(focal_length_in_pixels, baseline, shift=0):
    # For OAK-D @ 400P mono cameras and disparity of eg. 50 pixels
    minDisparityNormal = 95
    minDisparityExtended = 190
    maxDisparity = 1 if shift == 0 else shift

    min_distance_normal = focal_length_in_pixels * baseline / (minDisparityNormal + shift) # = 882.5 * 7.5cm / 190 = 34.84cm
    min_distance_extended = focal_length_in_pixels * baseline / (minDisparityExtended + shift) # = 882.5 * 7.5cm / 190 = 34.84cm

    max_distance_normal = focal_length_in_pixels * baseline / maxDisparity
    max_distance_extended = focal_length_in_pixels * baseline

    return min_distance_normal, min_distance_extended, max_distance_normal, max_distance_extended


with dai.Device() as device:
    calibData = device.readCalibration()

    intrinsicsLeft = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B)
    intrinsicsRight = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C)

    averageFocalLength = (intrinsicsLeft[0][0] + intrinsicsRight[0][0]) / 2
    print('Left camera focal length in pixels:', intrinsicsLeft[0][0])
    print('Right camera focal length in pixels:', intrinsicsRight[0][0])
    print('Average focal length in pixels:', averageFocalLength)
    normal_min, extended_min, normal_max, extended_max = calculate_focal_depths(averageFocalLength, 5.83)
    shifted_min, _, shifted_max, _ = calculate_focal_depths(averageFocalLength, 5.83, 30)

    print('Normal mode minimum distance:', normal_min)
    print('Normal mode maximum distance:', normal_max)
    print('Normal (shifted) mode minimum distance:', shifted_min)
    print('Normal (shifted) mode maximum distance:', shifted_max)
    print('Extended mode minimum distance:', extended_min)
    print('Extended mode maximum distance:', extended_max)

