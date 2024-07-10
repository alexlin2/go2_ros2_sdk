import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from vision_msgs.msg import Detection2D, Detection2DArray, Detection3D, Detection3DArray

import numpy as np
import cv2
from cv_bridge import CvBridge
import PyKDL
import yaml
import os
from ament_index_python.packages import get_package_share_directory
from sensor_msgs_py.point_cloud2 import read_points, create_cloud_xyz32

import tf2_ros

from ultralytics import YOLO

class DetectImages(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        self.image_subscription = self.create_subscription(
            Image,
            '/go2_camera/color/image',
            self.image_callback,
            10)
        self.lidar_subscription = self.create_subscription(
            PointCloud2,
            '/robot0/point_cloud2',
            self.lidar_callback,
            10)
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        package_dir = get_package_share_directory('go2_detection')
        yaml_file_path = os.path.join(package_dir, 'config', 'go2_camerainfo.yaml')
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)
            camera_info = CameraInfo()
            camera_info.width = data['image_width']
            camera_info.height = data['image_height']
            camera_info.distortion_model = data['distortion_model']
            camera_info.d = data['distortion_coefficients']['data']
            camera_info.k = data['camera_matrix']['data']
            camera_info.r = data['rectification_matrix']['data']
            camera_info.p = data['projection_matrix']['data']

        self.camera_info = camera_info
        self.image_topic_frame = "robot0/front_camera"


        self.camera_info_publisher = self.create_publisher(CameraInfo, 'camera_info', 10)
        self.overlay_image_publisher = self.create_publisher(Image, 'overlay_image', 10)
        #self.detection3d_publisher = self.create_publisher(Detection2DArray, 'detection2d', 10)
        self.transformed_pc_publisher = self.create_publisher(PointCloud2, 'transformed_pc', 10)

    def transform_to_kdl(self, t):
        return PyKDL.Frame(PyKDL.Rotation.Quaternion(
                           t.transform.rotation.x, t.transform.rotation.y,
                           t.transform.rotation.z, t.transform.rotation.w),
                           PyKDL.Vector(t.transform.translation.x,
                           t.transform.translation.y,
                           t.transform.translation.z))

    def do_transform_cloud(self, cloud, transform):
        t_kdl = self.transform_to_kdl(transform)
        points_out = []
        for p_in in read_points(cloud, field_names=("x", "y", "z")):
            p_out = t_kdl * PyKDL.Vector(p_in[0], p_in[1], p_in[2])
            points_out.append([p_out[0], p_out[1], p_out[2]])
        res = create_cloud_xyz32(transform.header, points_out)
        return res

    def lidar_callback(self, msg):
        try:
            transform = self.tf_buffer.lookup_transform(self.image_topic_frame,
                                                        msg.header.frame_id,
                                                        rclpy.time.Time(),
                                                        rclpy.duration.Duration(seconds=0.5))
            transformed_pc = self.do_transform_cloud(msg, transform)
            self.transformed_pc_publisher.publish(transformed_pc)

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error('Failed to transform PointCloud2: %s' % str(e))

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model.track(cv_image, persist=True, verbose=True)
        annotated_frame = results[0].plot()
        img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, 'bgr8')
        self.overlay_image_publisher.publish(img_msg)
        self.camera_info_publisher.publish(self.camera_info)


def main(args=None):
    rclpy.init(args=args)

    image_subscriber = DetectImages()

    rclpy.spin(image_subscriber)

    image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()