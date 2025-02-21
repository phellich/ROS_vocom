import rclpy
from rclpy.node import Node
# from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from sensor_msgs.msg import Joy
from rclpy.executors import MultiThreadedExecutor
import threading

class FakeCSnode(Node):

    def __init__(self):
        super().__init__('fake_cs_node')

        # Création d'un client pour envoyer des requêtes au service
        self.client = self.create_client(SetBool, '/CS/vocom_activation_service')
        
        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('En attente que le service "vocom_activation_service" soit disponible...')

        # Subscriber for listening to responses
        self.subscription = self.create_subscription(
            Joy, 
            '/CS/NAV_gamepad',
            self.listener_callback,
            10)

        self.get_logger().info("FakeCSnode started. Use 'y', 'n', or 'q' to interact.")

    def listener_callback(self, msg):
        """Logs the received response from the subscribed topic."""
        self.get_logger().info(f"Received Joy.axes: {msg.axes}")

    def send_request(self, activate: bool):
        """Sends a request to the service to activate or deactivate."""
        request = SetBool.Request()
        request.data = activate

        # Call the service asynchronously
        future = self.client.call_async(request)

        # Attach a callback for when the future completes
        future.add_done_callback(self.handle_service_response)

    def handle_service_response(self, future):
        """Handles the response from the service."""
        try:
            response = future.result()
            self.get_logger().info(f"Response: success={response.success}, message='{response.message}'")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {str(e)}")

    def handle_user_input(self):
        """Handles user input in a seperate Ros2 callback thread"""
        while rclpy.ok():
            command = input("Enter command (y/n/q): \n").strip().lower()
            if command == 'q':
                self.get_logger().info("Exiting FakeCSnode.")
                rclpy.shutdown()
            elif command == 'y':
                self.get_logger().info("Sending activation request to the service...")
                self.send_request(True)
            elif command == 'n':
                self.get_logger().info("Sending deactivation request to the service...")
                self.send_request(False)
            else:
                self.get_logger().warning("Invalid input. Please enter 'y', 'n', or 'q'.")
            

def main(args=None):
    rclpy.init(args=args)
    node = FakeCSnode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    # Create a separate thread for user input
    user_input_thread = threading.Thread(target=node.handle_user_input, daemon=True)
    user_input_thread.start()
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
