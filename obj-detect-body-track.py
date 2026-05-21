########################################################################
#
# Copyright (c) 2025, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    Detects CNC machines (Haas ST-20Y, VF-2YT, VF-3) using a custom YOLOv11
    ONNX model, feeds bounding boxes into the ZED SDK for 3D localization,
    and tracks people with body tracking.
"""

import argparse
import sys
import numpy as np
import cv2
import pyzed.sl as sl
from ultralytics import YOLO

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

ONNX_MODEL_PATH = "/home/txst-robotics/MARS-code/makerspace-detector/images/best.onnx"
CNC_CLASSES = ['haas-st-20y', 'haas-vf-2yt', 'haas-vf-3']

USE_BATCHING = False

def main(opt):
    print("Running CNC machine detection + body tracking ... Press 'Esc' to quit")
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.depth_maximum_distance = 20
    is_playback = False

    if opt.input_svo_file is not None:
        print(f"Using SVO file: {opt.input_svo_file}")
        init_params.set_from_svo_file(opt.input_svo_file)
        is_playback = True

    status = zed.open(init_params)
    if status > sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    print(f"Loading CNC model: {ONNX_MODEL_PATH}")
    cnc_model = YOLO(ONNX_MODEL_PATH, task='detect')

    # Enable positional tracking
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)

    # Enable custom box object detection (ZED adds 3D positions to our detections)
    obj_param = sl.ObjectDetectionParameters()
    obj_param.instance_module_id = 0
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)

    # Enable body tracking for people
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True
    body_param.enable_body_fitting = False
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    body_param.body_format = sl.BODY_FORMAT.BODY_18
    body_param.instance_module_id = 1
    zed.enable_body_tracking(body_param)

    # Initialize GPU support after ZED models are loaded
    gl._initialize_gpu()
    use_gpu = gl.GPU_ACCELERATION_AVAILABLE and not opt.disable_gpu_data_transfer
    mem_type = sl.MEM.GPU if use_gpu else sl.MEM.CPU
    if use_gpu:
        print("Using GPU data transfer with CuPy")

    camera_infos = zed.get_camera_information()
    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(
        min(camera_infos.camera_configuration.resolution.width, 720),
        min(camera_infos.camera_configuration.resolution.height, 404)
    )
    point_cloud_render = sl.Mat()
    viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)

    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40

    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = 50

    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, mem_type)
    objects = sl.Objects()

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40
    bodies = sl.Bodies()

    image_left = sl.Mat()

    display_resolution = sl.Resolution(
        min(camera_infos.camera_configuration.resolution.width, 1280),
        min(camera_infos.camera_configuration.resolution.height, 720)
    )
    image_scale = [
        display_resolution.width / camera_infos.camera_configuration.resolution.width,
        display_resolution.height / camera_infos.camera_configuration.resolution.height
    ]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    cam_w_pose = sl.Pose()
    quit_app = False

    while viewer.is_available() and not quit_app:
        if zed.grab(runtime_params) <= sl.ERROR_CODE.SUCCESS:
            # Retrieve image first — needed for YOLO inference
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            image_render_left = image_left.get_data()
            np.copyto(image_left_ocv, image_render_left)

            # Run CNC detection and feed boxes into ZED for 3D localization
            frame_bgr = image_render_left[:, :, :3]
            results = cnc_model(frame_bgr, imgsz=640, verbose=False)
            objects_in = []
            for result in results:
                for box in result.boxes:
                    tmp = sl.CustomBoxObjectData()
                    tmp.unique_object_id = sl.generate_unique_id()
                    tmp.probability = float(box.conf)
                    tmp.label = int(box.cls)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    tmp.bounding_box_2d = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                    tmp.is_grounded = True
                    objects_in.append(tmp)
            zed.ingest_custom_box_objects(objects_in)

            # Retrieve objects (ZED has now added 3D positions)
            returned_state = zed.retrieve_objects(objects, obj_runtime_param, obj_param.instance_module_id)
            returned_state2 = zed.retrieve_bodies(bodies, body_runtime_param, body_param.instance_module_id)

            if returned_state <= sl.ERROR_CODE.SUCCESS and objects.is_new:
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, mem_type, point_cloud_res)
                zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)
                viewer.updateData(point_cloud, objects)
                cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)

            if returned_state2 <= sl.ERROR_CODE.SUCCESS and bodies.is_new:
                cv_viewer.render_2D_SK(image_left_ocv, image_scale, bodies.body_list, obj_param.enable_tracking, sl.BODY_FORMAT.BODY_18)

            cv2.imshow("ZED | CNC Detection + Body Tracking", image_left_ocv)
            cv2.waitKey(1)

        if is_playback and zed.get_svo_position() == zed.get_svo_number_of_frames() - 1:
            print("End of SVO")
            quit_app = True

    cv2.destroyAllWindows()
    viewer.exit()
    image_left.free(sl.MEM.CPU)
    point_cloud.free(mem_type)
    point_cloud_render.free(sl.MEM.CPU)

    zed.disable_object_detection()
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it', default=None)
    parser.add_argument('--disable-gpu-data-transfer', action='store_true', help='Disable GPU data transfer acceleration with CuPy even if CuPy is available')
    opt = parser.parse_args()
    main(opt)
