# #!/usr/bin/env python3
# # camera_subscriber/camera_listener.py

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image, CameraInfo
# from cv_bridge import CvBridge
# import cv2
# import tf2_ros
# import numpy as np
# import ros2_numpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile
# from threading import Lock
# import tf_transformations


# # lock = Lock()

# def ros_qt_to_rt(quat, trans):
#     # Convert quaternion and translation to a 4x4 transformation matrix
#     # This should implement the conversion, typically using numpy or other libraries
#     # Example:
#     RT = np.eye(4)
#     RT[:3, :3] = tf_transformations.quaternion_matrix(quat)[:3, :3]
#     RT[:3, 3] = trans
#     return RT

# class CameraSubscriber(Node):
#     def __init__(self):
#         super().__init__('camera_subscriber')
        
#         # Declare the 'bot' parameter with a default value
#         self.declare_parameter('bot', 'default_robot')
        
#         # Get the parameter value
#         bot = self.get_parameter('bot').get_parameter_value().string_value
#         self.get_logger().info(f"Bot parameter value: {bot}")




#         # Set up the transform listener and buffer
#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

#         # Camera and base frame names (these should match your setup)
#         self.base_frame = 'base_link'
#         self.camera_frame = 'camera_depth_frame'
#         self.bot=bot

#         # Initialize the CvBridge
#         self.bridge = CvBridge()

#         # RGB Camera Subscription
#         self.rgb_subscription = self.create_subscription(
#             Image,
#             f'/{bot}/camera/image_raw',  # Replace with your actual RGB camera topic
#             self.rgb_callback,
#             10
#         )

#         # Depth Camera Subscription
#         self.depth_subscription = self.create_subscription(
#             Image,
#             f'/{bot}/camera/depth/image_raw',  # Replace with your actual depth camera topic
#             self.depth_callback,
#             10
#         )

        
#         #for rgbd_depth
#         self.depth_subscriber = self.create_subscription(
#             Image,
#             f'/{bot}/camera/depth/image_raw', # Change to the correct topic name
#             self.callback_rgbd_depth,
#             QoSProfile(depth=10)
#         )

#         # Camera Info Subscription
#         self.camera_info_subscription = self.create_subscription(
#             CameraInfo,
#             f'/{bot}/camera/camera_info',  # Replace with your actual camera info topic
#             self.camera_info_callback,
#             10
#         )

#         # Initialize variables
#         self.depth = None
#         # self.RT_camera = None
#         self.lock = Lock()

#         self.intrinsics = None
#         self.fx = self.fy = self.px = self.py = None
#         self.rgb_frame_id = None
#         self.rgb_frame_stamp = None
#         self.im = None
#         self.RT_camera = None
#         self.RT_base = None
#         self.RT_laser = None
#         self.latest_image=None
#         self.latest_depth_img=None

#         self.pub_images = self.create_publisher(Image, f'/{bot}/image_data', 10)
#         # Timer to publish markers periodically
#         self.pub_images_publish_rate = 2.0  # seconds
#         self.pub_images_publish_timer = self.create_timer(self.pub_images_publish_rate, self.publish_image)

#         self.pub_depth_images = self.create_publisher(Image, f'/{bot}/depth_image_data', 10)
#         # Timer to publish markers periodically
#         self.pub_depth_images_publish_rate = 2.0  # seconds
#         self.pub_depth_images_publish_timer = self.create_timer(self.pub_depth_images_publish_rate, self.publish_depth_image)

#         self.get_logger().info("Camera Subscriber Node has been started.")

#     def rgb_callback(self, msg):
#         """Callback for RGB camera images."""
#         self.get_logger().info("RGB image received.")

#         try:
#             # Convert ROS Image message to OpenCV format
#             cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#             self.latest_image = cv_image.copy()
#             # Display the image using OpenCV
#             # cv2.imshow(f"RGB Image {self.bot}", cv_image)
#             # cv2.waitKey(1)
#         except Exception as e:
#             self.get_logger().error(f"Failed to convert RGB image: {e}")

#         self.rgb_frame_id = msg.header.frame_id
#         self.rgb_frame_stamp = msg.header.stamp
#         self.im = ros2_numpy.numpify(msg)
#         if self.im is None:                                                         #for debugging remove it after
#             self.get_logger().warn("self.im is None.")
#         else:
#             self.get_logger().info(f"self.im is not None, shape: {self.im.shape}")

#         self.get_logger().info(f"Received RGB Image: {msg.header.stamp}")
#         # return cv_image


#     def publish_image(self):
#         image = self.bridge.cv2_to_imgmsg(self.latest_image,"bgr8")
#         self.pub_images.publish(image)

#     def publish_depth_image(self):
#         depth_bgr = cv2.cvtColor(self.latest_depth_img, cv2.COLOR_GRAY2BGR)
#         self.get_logger().info(f"{type(self.latest_depth_img)}    typetype")
#         image = self.bridge.cv2_to_imgmsg(depth_bgr,"bgr8")
#         self.pub_depth_images.publish(image)
    
#     def depth_callback(self, msg):
#         self.get_logger().info("Depth image received.")

#         try:
#             # Handle different encodings (convert depth to meters if necessary)
#             if msg.encoding == "16UC1":
#                 depth_cv = self.bridge.imgmsg_to_cv2(msg, "16UC1").astype(np.float32) / 1000.0
#             elif msg.encoding == "32FC1":
#                 depth_cv = self.bridge.imgmsg_to_cv2(msg, "32FC1")
#             else:
#                 self.get_logger().error(f"Unsupported depth encoding: {msg.encoding}")
#                 return


#             self.latest_depth_img = depth_cv
#             # Replace NaN and invalid values
#             depth_cv[np.isnan(depth_cv)] = 0.0
#             depth_cv[depth_cv < 0] = 0.0  # Clamp negative depths

            
#             # Debug: Log min and max depth values
#             min_depth, max_depth = .1,10
#             self.get_logger().info(f"Depth range: {min_depth:.2f} to {max_depth:.2f} meters")
#             # specific_depth = depth_cv[20,170]
#             # self.get_logger().info(f"Depth: {specific_depth}")
#             # output_file = "/home/b4by_y0d4/Desktop/m_explore_ws/src/script_package/scripts/depth_matrix.csv"
#             # np.savetxt(output_file, depth_cv, fmt="%.2f", delimiter=",")
#             # self.get_logger().info(f"Depth matrix saved to {output_file}")


#             # Convert the depth to a grayscale image without normalization
#             depth_grayscale = (depth_cv * 255.0 / max_depth).astype(np.uint8)

#             # Display the depth image in grayscale
#             # cv2.imshow(f'Grayscale Depth Image {self.bot}', depth_grayscale)
#             # cv2.waitKey(1)
#         except Exception as e:
#             self.get_logger().error(f"Failed to process depth image: {e}")


    
#     def callback_rgbd_depth(self, msg):
#         # Get the transform from the base frame to the camera frame
#         try:
#             transform = self.tf_buffer.lookup_transform(
#                 self.base_frame, self.camera_frame, self.get_clock().now().to_msg()
#             )

#             # Extract translation and rotation
#             translation = transform.transform.translation
#             rotation = transform.transform.rotation

#             # Convert to numpy format or 4x4 matrix if needed
#             RT_camera = ros_qt_to_rt(
#                 [rotation.x, rotation.y, rotation.z, rotation.w],
#                 [translation.x, translation.y, translation.z]
#             )

#             # Additional transforms (e.g., for lidar or robot base)
#             transform_laser = self.tf_buffer.lookup_transform(
#                 self.base_frame, "two_d_lidar", self.get_clock().now().to_msg()
#             )
#             translation_laser = transform_laser.transform.translation
#             rotation_laser = transform_laser.transform.rotation
#             RT_laser = ros_qt_to_rt(
#                 [rotation_laser.x, rotation_laser.y, rotation_laser.z, rotation_laser.w],
#                 [translation_laser.x, translation_laser.y, translation_laser.z]
#             )

#             # Transform from map to base (if needed)
#             transform_base = self.tf_buffer.lookup_transform(
#                 "map", self.base_frame, self.get_clock().now().to_msg()
#             )
#             translation_base = transform_base.transform.translation
#             rotation_base = transform_base.transform.rotation
#             RT_base = ros_qt_to_rt(
#                 [rotation_base.x, rotation_base.y, rotation_base.z, rotation_base.w],
#                 [translation_base.x, translation_base.y, translation_base.z]
#             )

#         except tf2_ros.LookupException as e:
#             self.get_logger().warn(f"Transform lookup failed: {str(e)}")
#             RT_camera = None
#             RT_laser = None
#             RT_base = None
#         except tf2_ros.TransformException as e:
#             self.get_logger().warn(f"Transform lookup failed: {str(e)}")
#             return

#         # Process the depth message
#         if msg.encoding == "32FC1":
#             depth_cv = ros2_numpy.numpify(msg)
#             depth_cv[np.isnan(depth_cv)] = 0
#         elif msg.encoding == "16UC1":
#             depth_cv = ros2_numpy.numpify(msg).astype(np.float64) / 1000.0
#         else:
#             self.get_logger().error(f"Unsupported depth type: {msg.encoding}")
#             return

#         # Update depth and transform matrices with threading lock
#         with self.lock:
#             self.depth = depth_cv.copy()
#             self.RT_camera = RT_camera
#             self.RT_base = RT_base
#             self.RT_laser = RT_laser
#             if self.depth is not None:
#                 self.get_logger().info(f"self.depth shape: {self.depth.shape}, dtype: {self.depth.dtype}")
#             else:
#                 self.get_logger().warn("self.depth is None.")



#     def camera_info_callback(self, msg):
#         """Callback for CameraInfo."""
#         self.get_logger().info("Camera Info received.")

#         # Print some of the camera info, such as the camera matrix and distortion coefficients
#         try:
#             # Camera Matrix
#             camera_matrix = msg.k
#             self.get_logger().info(f"Camera Matrix (K): {camera_matrix}")

#             # Distortion coefficients
#             distortion_coeffs = msg.d
#             self.get_logger().info(f"Distortion Coefficients (D): {distortion_coeffs}")

#             # Print the resolution (width, height)
#             resolution = (msg.width, msg.height)
#             self.get_logger().info(f"Camera Resolution: {resolution}")
#         except Exception as e:
#             self.get_logger().error(f"Failed to retrieve Camera Info: {e}")

#         self.intrinsics = np.array(camera_matrix).reshape(3, 3)
#         self.fx = self.intrinsics[0, 0]
#         self.fy = self.intrinsics[1, 1]
#         self.px = self.intrinsics[0, 2]
#         self.py = self.intrinsics[1, 2]
#         self.get_logger().info(f"Intrinsics: \n{self.intrinsics}")

#     def get_data_to_save(self):
#         with self.lock:
#         #     if self.im is None:
#         #         return None, None
#         #     RT_camera = self.RT_camera.copy()
#         #     RT_base = self.RT_base.copy()
#         # return RT_camera, RT_base
#             if self.im is None:
#                 return None, None
#             if self.RT_camera is None or self.RT_base is None:
#                 self.get_logger().warn("RT_camera or RT_base is None!")
#                 return None, None
#             RT_camera = self.RT_camera.copy()
#             RT_base = self.RT_base.copy()
#         return RT_camera, RT_base

#     def save_transformation(self, RT_camera, RT_base):
#         # Example: Save matrices to a file or process
#         self.get_logger().info(f"Saving transformation data: Camera: {RT_camera}, Base: {RT_base}")
#         # For example, save to a file
#         np.save('RT_camera.npy', RT_camera)
#         np.save('RT_base.npy', RT_base)

#     def save_data(self):
#         # RT_camera, RT_base = self.get_data_to_save()
#         # if RT_camera is not None and RT_base is not None:
#         #     self.save_transformation(RT_camera, RT_base)
#         RT_camera, RT_base = self.get_data_to_save()
#         if RT_camera is not None and RT_base is not None:
#             self.save_transformation(RT_camera, RT_base)
#         else:
#             self.get_logger().warn("Cannot save transformation: data is missing!")



# def main(args=None):
#     rclpy.init(args=args)

#     # Create the camera subscriber node
#     node = CameraSubscriber()

#     node.timer = node.create_timer(1.0, node.save_data)  # Call save_data every second
#     # Spin the node to keep it alive
#     rclpy.spin(node)

#     # Clean up and shutdown
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()




#!/usr/bin/env python3
# camera_subscriber/camera_listener.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg._float64_multi_array import Float64MultiArray
from std_msgs.msg import MultiArrayDimension
from cv_bridge import CvBridge
import cv2
import tf2_ros
import numpy as np
import ros2_numpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from threading import Lock
import tf_transformations
from tf_transformations import euler_from_quaternion
from datetime import datetime
# lock = Lock()

def ros_qt_to_rt(quat, trans):
    # Convert quaternion and translation to a 4x4 transformation matrix
    # This should implement the conversion, typically using numpy or other libraries
    # Example:
    RT = np.eye(4)
    RT[:3, :3] = tf_transformations.quaternion_matrix(quat)[:3, :3]
    RT[:3, 3] = trans
    return RT

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        
        # Declare the 'bot' parameter with a default value
        self.declare_parameter('bot', 'default_robot')
        
        # Get the parameter value
        bot = self.get_parameter('bot').get_parameter_value().string_value
        self.get_logger().info(f"Bot parameter value: {bot}")




        # Set up the transform listener and buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Camera and base frame names (these should match your setup)
        self.base_frame = 'base_link'
        self.camera_frame = 'camera_depth_frame'
        self.bot=bot

        # Initialize the CvBridge
        self.bridge = CvBridge()

        # RGB Camera Subscription
        self.rgb_subscription = self.create_subscription(
            Image,
            f'/{bot}/camera/image_raw',  # Replace with your actual RGB camera topic
            self.rgb_callback,
            10
        )

        # Depth Camera Subscription
        self.depth_subscription = self.create_subscription(
            Image,
            f'/{bot}/camera/depth/image_raw',  # Replace with your actual depth camera topic
            self.depth_callback,
            10
        )

        
        #for rgbd_depth
        self.depth_subscriber = self.create_subscription(
            Image,
            f'/{bot}/camera/depth/image_raw', # Change to the correct topic name
            self.callback_rgbd_depth,
            QoSProfile(depth=10)
        )

        # Camera Info Subscription
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            f'/{bot}/camera/camera_info',  # Replace with your actual camera info topic
            self.camera_info_callback,
            10
        )

        # Initialize variables
        self.depth = None
        # self.RT_camera = None
        self.lock = Lock()

        self.intrinsics = None
        self.fx = self.fy = self.px = self.py = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
        self.im = None
        self.RT_camera = None
        self.RT_base = None
        self.RT_laser = None
        self.latest_image=None
        self.latest_depth_img=None

        self.pub_images = self.create_publisher(Image, f'/{bot}/image_data', 10)
        # Timer to publish markers periodically
        self.pub_images_publish_rate = 1.0  # seconds
        self.pub_images_publish_timer = self.create_timer(self.pub_images_publish_rate, self.publish_image)

        self.pub_depth_images = self.create_publisher(Float64MultiArray, f'/{bot}/depth_image_data', 10)
        # Timer to publish markers periodically
        self.pub_depth_images_publish_rate = 1.0  # seconds
        self.pub_depth_images_publish_timer = self.create_timer(self.pub_depth_images_publish_rate, self.publish_depth_image)

        self.get_logger().info("Camera Subscriber Node has been started.")

    def rgb_callback(self, msg):
        """Callback for RGB camera images."""
        self.get_logger().info(f"RGB image received. for {self.bot}")

        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image.copy()
            # Display the image using OpenCV
            # cv2.imshow(f"RGB Image {self.bot}", cv_image)
            # cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")

        self.rgb_frame_id = msg.header.frame_id
        self.rgb_frame_stamp = msg.header.stamp
        self.im = ros2_numpy.numpify(msg)
        if self.im is None:                                                         #for debugging remove it after
            self.get_logger().warn("self.im is None.")
        else:
            self.get_logger().info(f"self.im is not None, shape: {self.im.shape}")

        # self.get_logger().info(f"Received RGB Image: {msg.header.stamp}")
        # return cv_image


    def publish_image(self):
        if self.latest_depth_img is None:
            self.get_logger().warn(f"No image available for {self.bot} to publish")
            return
        elif not isinstance(self.latest_image, np.ndarray):
            self.get_logger().warn(f"Latest image is not a numpy array for {self.bot}, skipping publish.")
            return
        
        image = self.bridge.cv2_to_imgmsg(self.latest_image,"bgr8")
        self.pub_images.publish(image)
        self.get_logger().warn(f"image published for {self.bot} to publish")


    def publish_depth_image(self):
        if self.latest_depth_img is None:
            self.get_logger().warn(f"No depth image available for {self.bot} to publish")
            return

        try:
            # Create Float64MultiArray message
            self.latest_depth_img = np.asarray(self.latest_depth_img, dtype=np.float32)
            self.latest_depth_img = np.round(self.latest_depth_img, decimals=3)
            depth_msg = Float64MultiArray()
            depth_data = self.latest_depth_img.flatten().tolist()


            depth_data = [float(-999.999) if np.isnan(x) else float(x) for x in depth_data]
            # Set up the layout
            depth_msg.layout.dim = [
                MultiArrayDimension(
                    label="height",
                    size=self.latest_depth_img.shape[0],
                    stride=self.latest_depth_img.shape[0] * self.latest_depth_img.shape[1]
                ),
                MultiArrayDimension(
                    label="width",
                    size=self.latest_depth_img.shape[1],
                    stride=self.latest_depth_img.shape[1]
                )
            ]

            # Flatten the numpy array and convert to list

            # Handle NaN values

            depth_msg.data = depth_data

            # Publish the message
            self.pub_depth_images.publish(depth_msg)
            self.get_logger().info(f"Published depth data with shape: {self.latest_depth_img.shape}")

        except Exception as e:
            self.get_logger().error(f"Error publishing depth data: {str(e)}")

    def depth_callback(self, msg):
        self.get_logger().info("Depth image received. for {self.bot}")

        try:
            # Handle different encodings (convert depth to meters if necessary)
            if msg.encoding == "16UC1":
                depth_cv = self.bridge.imgmsg_to_cv2(msg, "16UC1").astype(np.float32) / 1000.0
            elif msg.encoding == "32FC1":
                depth_cv = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            else:
                self.get_logger().error(f"Unsupported depth encoding: {msg.encoding}")
                return


            # Replace NaN and invalid values
            depth_cv[np.isnan(depth_cv)] = -1
            self.latest_depth_img = depth_cv.copy()
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # output_file = f"/home/vunknow/patroling_data/depth_matrix_{timestamp}.npy"
            # np.save(output_file, depth_cv)
            # depth_cv[depth_cv < 0] = 0.0  # Clamp negative depths #TODO

            
            # # Debug: Log min and max depth values
            # min_depth, max_depth = .1,10
            # self.get_logger().info(f"Depth range: {min_depth:.2f} to {max_depth:.2f} meters")
            
            # # specific_depth = depth_cv[20,170]
            # # self.get_logger().info(f"Depth: {specific_depth}")
            # # self.get_logger().info(f"Depth matrix saved to {output_file}")


            # # Convert the depth to a grayscale image without normalization
            # depth_grayscale = (depth_cv * 255.0 / max_depth).astype(np.uint8)

            # # Display the depth image in grayscale
            # # cv2.imshow(f'Grayscale Depth Image {self.bot}', depth_grayscale)
            # # cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Failed to process depth image: {e}")
    
    def callback_rgbd_depth(self, msg):
        # Get the transform from the base frame to the camera frame
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, self.get_clock().now().to_msg()
            )

            # Extract translation and rotation
            translation = transform.transform.translation
            rotation = transform.transform.rotation

            # Convert to numpy format or 4x4 matrix if needed
            RT_camera = ros_qt_to_rt(
                [rotation.x, rotation.y, rotation.z, rotation.w],
                [translation.x, translation.y, translation.z]
            )

            # Additional transforms (e.g., for lidar or robot base)
            transform_laser = self.tf_buffer.lookup_transform(
                self.base_frame, "two_d_lidar", self.get_clock().now().to_msg()
            )
            translation_laser = transform_laser.transform.translation
            rotation_laser = transform_laser.transform.rotation
            RT_laser = ros_qt_to_rt(
                [rotation_laser.x, rotation_laser.y, rotation_laser.z, rotation_laser.w],
                [translation_laser.x, translation_laser.y, translation_laser.z]
            )

            # Transform from map to base (if needed)
            transform_base = self.tf_buffer.lookup_transform(
                "map", self.base_frame, self.get_clock().now().to_msg()
            )
            translation_base = transform_base.transform.translation
            rotation_base = transform_base.transform.rotation
            RT_base = ros_qt_to_rt(
                [rotation_base.x, rotation_base.y, rotation_base.z, rotation_base.w],
                [translation_base.x, translation_base.y, translation_base.z]
            )

        except tf2_ros.LookupException as e:
            self.get_logger().warn(f"Transform lookup failed: {str(e)}")
            RT_camera = None
            RT_laser = None
            RT_base = None
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"Transform lookup failed: {str(e)}")
            return

        # Process the depth message
        if msg.encoding == "32FC1":
            depth_cv = ros2_numpy.numpify(msg)
            depth_cv[np.isnan(depth_cv)] = 0
        elif msg.encoding == "16UC1":
            depth_cv = ros2_numpy.numpify(msg).astype(np.float64) / 1000.0
        else:
            self.get_logger().error(f"Unsupported depth type: {msg.encoding}")
            return

        # Update depth and transform matrices with threading lock
        with self.lock:
            self.depth = depth_cv.copy()
            self.RT_camera = RT_camera
            self.RT_base = RT_base
            self.RT_laser = RT_laser
            if self.depth is not None:
                self.get_logger().info(f"self.depth shape: {self.depth.shape}, dtype: {self.depth.dtype}")
            else:
                self.get_logger().warn("self.depth is None.")



    def camera_info_callback(self, msg):
        """Callback for CameraInfo."""
        self.get_logger().info("Camera Info received.")

        # Print some of the camera info, such as the camera matrix and distortion coefficients
        try:
            # Camera Matrix
            camera_matrix = msg.k
            self.get_logger().info(f"Camera Matrix (K): {camera_matrix}")

            # Distortion coefficients
            distortion_coeffs = msg.d
            self.get_logger().info(f"Distortion Coefficients (D): {distortion_coeffs}")

            # Print the resolution (width, height)
            resolution = (msg.width, msg.height)
            self.get_logger().info(f"Camera Resolution: {resolution}")
        except Exception as e:
            self.get_logger().error(f"Failed to retrieve Camera Info: {e}")

        self.intrinsics = np.array(camera_matrix).reshape(3, 3)
        self.fx = self.intrinsics[0, 0]
        self.fy = self.intrinsics[1, 1]
        self.px = self.intrinsics[0, 2]
        self.py = self.intrinsics[1, 2]
        self.get_logger().info(f"Intrinsics: \n{self.intrinsics}")

    def get_data_to_save(self):
        with self.lock:
        #     if self.im is None:
        #         return None, None
        #     RT_camera = self.RT_camera.copy()
        #     RT_base = self.RT_base.copy()
        # return RT_camera, RT_base
            if self.im is None:
                return None, None
            if self.RT_camera is None or self.RT_base is None:
                self.get_logger().warn("RT_camera or RT_base is None!")
                return None, None
            RT_camera = self.RT_camera.copy()
            RT_base = self.RT_base.copy()
        return RT_camera, RT_base

    def save_transformation(self, RT_camera, RT_base):
        # Example: Save matrices to a file or process
        self.get_logger().info(f"Saving transformation data: Camera: {RT_camera}, Base: {RT_base}")
        # For example, save to a file
        np.save('RT_camera.npy', RT_camera)
        np.save('RT_base.npy', RT_base)

    def save_data(self):
        # RT_camera, RT_base = self.get_data_to_save()
        # if RT_camera is not None and RT_base is not None:
        #     self.save_transformation(RT_camera, RT_base)
        RT_camera, RT_base = self.get_data_to_save()
        if RT_camera is not None and RT_base is not None:
            self.save_transformation(RT_camera, RT_base)
        else:
            self.get_logger().warn("Cannot save transformation: data is missing!")



def main(args=None):
    rclpy.init(args=args)

    # Create the camera subscriber node
    node = CameraSubscriber()

    node.timer = node.create_timer(1.0, node.save_data)  # Call save_data every second
    # Spin the node to keep it alive
    rclpy.spin(node)

    # Clean up and shutdown
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()