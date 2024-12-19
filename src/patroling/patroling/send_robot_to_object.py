import rclpy
from rclpy.node import Node
from patrolling_interfaces.srv import SendRobotToObject

class SendRobotToObjectNode(Node):
    def __init__(self):
        super().__init__('send_robot_to_object_node')
        self.client = self.create_client(SendRobotToObject, 'send_robot_to_object')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def send_request(self, robot_name, object_name):
        req = SendRobotToObject.Request()
        req.robot_name = robot_name
        req.object_name = object_name
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

def main():
    rclpy.init()
    node = SendRobotToObjectNode()

    robot_name = input("Enter the robot name: ")
    object_name = input("Enter the object name: ")

    response = node.send_request(robot_name, object_name)
    if response.success:
        node.get_logger().info(response.message)
    else:
        node.get_logger().error(response.message)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()