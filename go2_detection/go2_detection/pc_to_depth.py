#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import PyKDL
from scipy.interpolate import NearestNDInterpolator
from sensor_msgs_py.point_cloud2 import read_points, create_cloud_xyz32
import tf2_ros
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf2_ros.buffer
import tf2_ros.transform_listener

class PointCloudToDepth(Node):
    def __init__(self):
        super().__init__('pointcloud_to_depth')
        
        # Declare parameters
        self.declare_parameter('interpolate', False)
        self.declare_parameter('interpolation_scale', 0.1)
        self.declare_parameter('min_depth', 0.1)
        self.declare_parameter('max_depth', 10.0)
        self.declare_parameter('queue_size', 5)
        self.declare_parameter('slop', 0.1)
        
        # Get parameters
        self.do_interpolate = self.get_parameter('interpolate').value
        self.interpolation_scale = self.get_parameter('interpolation_scale').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        self.queue_size = self.get_parameter('queue_size').value
        self.slop = self.get_parameter('slop').value

        # Set up QoS profile for better synchronization
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=self.queue_size
        )

        # Initialize tf buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Initialize bridge
        self.bridge = CvBridge()
        
        # Create subscribers with message filters for synchronization
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/point_cloud2',
            self.pointcloud_callback,
            qos)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            qos)
            
        # Publishers
        self.depth_pub = self.create_publisher(Image, '/camera/depth_raw', 1)
        self.depth_info_pub = self.create_publisher(CameraInfo, '/camera/depth/camera_info', 1)
        self.transformed_pc_pub = self.create_publisher(PointCloud2, '/camera/depth/points', 1)
        
        # Store latest camera info
        self.camera_info = None
        self.camera_matrix = None

    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.camera_matrix = np.array([
            [msg.k[0], 0, msg.k[2]],
            [0, msg.k[4], msg.k[5]],
            [0, 0, 1]
        ], dtype=np.float32)  # Ensure float32

    def transform_to_kdl(self, t):
        """Convert a geometry msgs Transform to PyKDL Frame."""
        return PyKDL.Frame(
            PyKDL.Rotation.Quaternion(
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w),
            PyKDL.Vector(
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z))

    def transform_cloud(self, cloud, transform):
        """Transform point cloud using PyKDL."""
        t_kdl = self.transform_to_kdl(transform)
        points_out = []
        for p_in in read_points(cloud, field_names=("x", "y", "z")):
            p_out = t_kdl * PyKDL.Vector(p_in[0], p_in[1], p_in[2])
            points_out.append([p_out[0], p_out[1], p_out[2]])
        return np.array(points_out, dtype=np.float32)  # Ensure float32

    def points_to_cloud_msg(self, points, frame_id, stamp):
        """Convert points array to PointCloud2 message."""
        return create_cloud_xyz32(
            header=cloud_msg.header,
            points=points.tolist()
        )

    def project_points(self, points, image_width, image_height):
        """Project 3D points to 2D image plane."""
        # Project points onto the camera plane
        projected_points = np.dot(self.camera_matrix, points.T).T

        # Filter points behind camera
        valid_points = projected_points[:, 2] > self.min_depth
        projected_points = projected_points[valid_points]
        depths = projected_points[:, 2].astype(np.float32)  # Ensure float32

        # Normalize coordinates
        pixel_coords = projected_points[:, :2] / projected_points[:, 2:]

        # Round to nearest pixel
        pixel_coords = np.round(pixel_coords).astype(np.int32)

        # Filter points outside image bounds
        valid_points = (
            (pixel_coords[:, 0] >= 0) &
            (pixel_coords[:, 0] < image_width) &
            (pixel_coords[:, 1] >= 0) &
            (pixel_coords[:, 1] < image_height) &
            (depths < self.max_depth)
        )

        return pixel_coords[valid_points], depths[valid_points]

    def create_depth_map(self, pixel_coords, depths, image_width, image_height):
        """Create depth map from projected points."""
        depth_map = np.zeros((image_height, image_width), dtype=np.float32)
        depth_map[pixel_coords[:, 1], pixel_coords[:, 0]] = depths
        return depth_map

    def interpolate_depth_map(self, depth_map, pixel_coords, depths):
        """Interpolate sparse depth map to dense depth map."""
        # Downscale for faster interpolation
        small_shape = (
            int(depth_map.shape[0] * self.interpolation_scale),
            int(depth_map.shape[1] * self.interpolation_scale)
        )
        
        # Scale coordinates
        coords_scaled = pixel_coords * self.interpolation_scale
        
        # Create interpolator
        interpolator = NearestNDInterpolator(coords_scaled, depths)
        
        # Create coordinate grid
        grid_y, grid_x = np.mgrid[0:small_shape[0], 0:small_shape[1]]
        
        # Interpolate
        depth_small = interpolator((grid_x, grid_y))
        
        # Upscale back to original size
        depth_interpolated = cv2.resize(
            depth_small.astype(np.float32),  # Ensure float32 before resize
            (depth_map.shape[1], depth_map.shape[0]),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)  # Ensure float32 after resize
        
        return depth_interpolated

    def pointcloud_callback(self, cloud_msg):
        if self.camera_info is None:
            self.get_logger().warn('No camera info received yet')
            return

        target_frame = self.camera_info.header.frame_id

        try:
            # Look up transform
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                cloud_msg.header.frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.4))

            # Transform point cloud
            points = self.transform_cloud(cloud_msg, transform)

            # Publish transformed point cloud
            transformed_cloud_msg = create_cloud_xyz32(
                header=cloud_msg.header,
                points=points.tolist()
            )
            transformed_cloud_msg.header.frame_id = target_frame
            self.transformed_pc_pub.publish(transformed_cloud_msg)

            # Project points
            pixel_coords, depths = self.project_points(
                points,
                self.camera_info.width,
                self.camera_info.height
            )

            if len(depths) == 0:
                self.get_logger().warn('No valid points in view')
                return

            # Create depth map
            depth_map = self.create_depth_map(
                pixel_coords,
                depths,
                self.camera_info.width,
                self.camera_info.height
            )

            # Interpolate if requested
            if self.do_interpolate:
                depth_map = self.interpolate_depth_map(depth_map, pixel_coords, depths)

            # Ensure final depth map is float32
            depth_map = depth_map.astype(np.float32)

            # Publish depth image
            depth_msg = self.bridge.cv2_to_imgmsg(depth_map, encoding="32FC1")
            depth_msg.header = cloud_msg.header
            depth_msg.header.frame_id = target_frame
            self.depth_pub.publish(depth_msg)

            # Publish depth camera info (same as regular camera info for now)
            depth_camera_info = self.camera_info
            depth_camera_info.header.stamp = cloud_msg.header.stamp
            self.depth_info_pub.publish(depth_camera_info)

        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f'Could not transform point cloud: {str(ex)}')

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudToDepth()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()