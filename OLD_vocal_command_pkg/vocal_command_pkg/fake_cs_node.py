import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from rclpy.executors import MultiThreadedExecutor
import threading

class FakeCSnode(Node):

    def __init__(self):
        super().__init__('fake_cs_node')
        # Publisher for sending commands
        self.publisher_ = self.create_publisher(Bool, "fake_cs_cmd", 10)
        # Subscriber for listening to responses
        self.subscription = self.create_subscription(
            String, 
            'fake_cs_response',
            self.listener_callback,
            10)
        
        # Timer for publishing user input
        self.timer = self.create_timer(0.1, self.cmd_acquisition)

        self.user_input = None  # Store the most recent user input
        self.get_logger().info("FakeCSnode started. Use 'y', 'n', or 'q' to interact.")

    def listener_callback(self, msg):
        """Logs the received response from the subscribed topic."""
        self.get_logger().info(f"Received response: {msg.data}")
    
    def cmd_acquisition(self):
        """Publishes the user input as a Bool message."""
        if self.user_input is None:
            return  # No new input to process

        if self.user_input == 'q':
            self.get_logger().info("Exiting FakeCSnode.")
            # rclpy.shutdown()
            return

        bool_msg = Bool()
        if self.user_input == 'y':
            bool_msg.data = True
        elif self.user_input == 'n':
            bool_msg.data = False
        else:
            self.get_logger().warning("Invalid input. Please enter 'y', 'n', or 'q'.")
            self.user_input = None
            return

        self.publisher_.publish(bool_msg)
        self.get_logger().info(f"Published state: {bool_msg.data}")
        self.user_input = None  # Reset after processing

    def handle_user_input(self):
        """Handles user input in a seperate Ros2 callback thread"""
        while rclpy.ok():
            command = input("Enter command (y/n/q): ").strip()
            self.user_input = command
            

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
