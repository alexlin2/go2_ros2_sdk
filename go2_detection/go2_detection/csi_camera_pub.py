#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import zenoh
import numpy as np
import cv2
import struct
import time
from builtin_interfaces.msg import Time

class CSICameraPublisher(Node):
    def __init__(self):
        super().__init__('csi_camera_publisher')
        
        # Create publishers with optimized QoS for real-time streaming
        # Use BEST_EFFORT reliability for faster throughput (doesn't resend lost messages)
        # Use KEEP_LAST with small history to minimize latency
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        self.image_publisher = self.create_publisher(Image, 'csi_camera/image_raw', qos)
        self.camera_info_publisher = self.create_publisher(CameraInfo, 'csi_camera/camera_info', qos)
        
        # Reuse CvBridge instance
        self.bridge = CvBridge()
        
        # Setup Zenoh session with optimized configuration
        self.zenoh_config = zenoh.Config()
        # Add Zenoh optimization settings if needed
        self.session = zenoh.open(self.zenoh_config)
        
        # Track frame timing for diagnostics
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.last_frame_time = 0
        
        # Declare ROS parameters
        self.declare_parameter('skip_frames', 0)
        self.skip_frames = self.get_parameter('skip_frames').value
        self.frame_counter = 0
        
        # Pre-allocate reusable camera info message
        self.camera_frame_id = "csi_camera_frame"
        self.last_width = 0
        self.last_height = 0
        self.last_K = None
        self.cam_info_msg = CameraInfo()
        self.cam_info_msg.header.frame_id = self.camera_frame_id
        
        # Subscribe to camera frames
        self.subscriber = self.session.declare_subscriber(
            "camera/frames",
            self.on_frame
        )
        
        # Add timer to publish diagnostics
        self.create_timer(5.0, self.publish_diagnostics)
        
        self.get_logger().info('CSI Camera Publisher started with optimized settings')

    def on_frame(self, sample):
        try:
            # Skip frames if configured (can help reduce CPU load)
            self.frame_counter += 1
            if self.skip_frames > 0 and (self.frame_counter % (self.skip_frames + 1) != 0):
                return
                
            # Track timing for FPS calculation
            current_time = time.time()
            self.frame_count += 1
            frame_interval = 0
            if self.last_frame_time > 0:
                frame_interval = current_time - self.last_frame_time
            self.last_frame_time = current_time
            
            # Get raw bytes from payload
            data = bytes(sample.payload)
            
            # Check if data is long enough to contain metadata with camera matrix K
            metadata_size = struct.calcsize('dii9d')  # timestamp, width, height, K matrix (9 values)
            
            # Process metadata
            K = None
            if len(data) >= metadata_size:
                # New format with camera matrix K
                metadata_values = struct.unpack('dii9d', data[:metadata_size])
                timestamp, width, height = metadata_values[0], metadata_values[1], metadata_values[2]
                # Extract K matrix values (last 9 values in metadata)
                K = np.array(metadata_values[3:]).reshape((3, 3))
                jpg_data = data[metadata_size:]
            elif len(data) >= 16:
                # Old format without camera matrix
                timestamp, width, height = struct.unpack('dii', data[:16])
                jpg_data = data[16:]
                
                # Log error about missing camera intrinsics
                self.get_logger().error('Camera intrinsics not available in the received data')
                # Only publish the image in this case, not the camera info
                K = None
            else:
                self.get_logger().warn('Received data too short')
                return
            
            # Use direct memory access for more efficient image decoding
            np_arr = np.frombuffer(jpg_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().warn('Failed to decode frame')
                return
            
            # Create ROS timestamp from Zenoh timestamp
            ros_time = self.get_clock().now().to_msg()
            
            # Convert to ROS Image message - avoid unnecessary copies
            img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            img_msg.header.stamp = ros_time
            img_msg.header.frame_id = self.camera_frame_id
            
            # Publish image message
            self.image_publisher.publish(img_msg)
            
            # Only create camera info if necessary (dimensions or K changed)
            if K is not None and (width != self.last_width or height != self.last_height or 
                                (self.last_K is None or not np.array_equal(K, self.last_K))):
                self._update_camera_info_msg(K, width, height)
            
            # Only publish camera info if we have valid intrinsics
            if K is not None:
                # Update timestamp
                self.cam_info_msg.header.stamp = ros_time
                self.camera_info_publisher.publish(self.cam_info_msg)
                
        except Exception as e:
            self.get_logger().warn(f'Error processing frame: {str(e)}')

    def _update_camera_info_msg(self, K, width, height):
        """Update the reusable CameraInfo message with new parameters."""
        self.cam_info_msg.height = height
        self.cam_info_msg.width = width
        
        # Set camera matrix K (row-major 3x3 matrix)
        self.cam_info_msg.k = K.flatten().tolist()
        
        # Set projection matrix P (row-major 3x4 matrix)
        # For undistorted cameras, P is just [K|0]
        P = np.zeros((3, 4))
        P[:3, :3] = K
        self.cam_info_msg.p = P.flatten().tolist()
        
        # Set rotation matrix R to identity (row-major 3x3 matrix)
        self.cam_info_msg.r = np.eye(3).flatten().tolist()
        
        # Set distortion model and coefficients
        # Since we're working with already rectified images, we can set this to empty
        self.cam_info_msg.distortion_model = "plumb_bob"
        self.cam_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # 5 zeros for plumb_bob model
        
        # Store the values we used to construct this message
        self.last_width = width
        self.last_height = height
        self.last_K = K.copy()
    
    def publish_diagnostics(self):
        """Publish diagnostic information like FPS."""
        current_time = time.time()
        interval = current_time - self.last_fps_time
        if interval > 0:
            fps = self.frame_count / interval
            self.get_logger().info(f'Publishing at {fps:.1f} FPS')
            self.frame_count = 0
            self.last_fps_time = current_time

    def __del__(self):
        if hasattr(self, 'session'):
            self.session.close()

def main(args=None):
    rclpy.init(args=args)
    publisher = CSICameraPublisher()
    
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()