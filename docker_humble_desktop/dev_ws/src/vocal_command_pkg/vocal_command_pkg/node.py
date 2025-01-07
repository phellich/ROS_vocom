import json
import os 
import numpy as np
import rclpy
from rclpy.node import Node
# from custom_msg.msg import ScDrillCmds
from std_srvs.srv import SetBool
from sensor_msgs.msg import Joy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import time
from llama_cpp import Llama
import threading
from dotenv import load_dotenv, dotenv_values

from vocal_command_pkg.utils_comb import * 

class VoCom_PubSub(Node):

    def __init__(self):

        # VARIABLES INITIALIZATION
        super().__init__('vocom_node')
        
        self.vocom_model_state = False
        self.model_thread = None
        self.stop_event = threading.Event() 

        self.pub_freq_4_joy = 0.5                                                                           # sec
        self.commands = []                                                                                
        self.last_modified_time = None                                                                      # Track the last modified time commands.json
        self.results_folder = "/home/xplore/dev_ws/src/vocal_command_pkg/models_and_results/results/"
        self.models_folder = "/home/xplore/dev_ws/src/vocal_command_pkg/models_and_results/models/"
        self.json_file_path = self.results_folder + "commands.json"
        self.delay_2_give_commands = 30
        load_dotenv("/home/xplore/dev_ws/src/.env")                                                         # load porcupine key from .env file
        self.initialize_models()

        self.get_logger().info('Human-Rover communication node has started. Please activate model via CS')

        # PUBLISHER
        # qos_profile = QoSProfile(
        #     reliability=QoSReliabilityPolicy.BEST_EFFORT, # BEST_EFFORT: message will attempt to send message but if it fails it will not try again
        #     durability=QoSDurabilityPolicy.VOLATILE, # VOLATILE: if no subscribers are listening, the message sent is not saved
        #     history=QoSHistoryPolicy.KEEP_LAST, # KEEP_LAST: only the last n = depth messages are stored in the queue
        #     depth=1,
        # )
        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, depth=10)
        self.publisher_nav = self.create_publisher(Joy, "fake_cs_response", qos_profile)                       # /CS/NAV_gamepad
        # self.publisher_drill = self.create_publisher(ScDrillCmds, "fake_CS_drill_publi", qos_profile)                   # /SC/drill_cmd

        
        # SERVICE                                                                                           # https://docs.ros.org/en/noetic/api/std_srvs/html/srv/SetBool.html 
        self.service_ = self.create_service(SetBool, 'vocom_activation_service', self.handle_activation_request)  
        self.cam_client = self.create_client(SetBool, 'topic_service')                                   # See details in camera command pub
        # while not self.client.wait_for_service(timeout_sec=2.0):
        #     self.get_logger().info('En attente que le service "topic_service" soit disponible...')

        # FOR TEST with one node (without CS activation)
        self.vocom_model_state = True
        self.model_thread = threading.Thread(target=self.running_vocom_model, daemon=True)
        self.model_thread.start()

    def initialize_models(self):
        """Initialize Whisper models."""                                          # https://github.com/openai/whisper/discussions/1463
        self.model_whisper_base_en = whisper_config("base.en")
        self.get_logger().info("Whisper Base.en ready")

    def handle_activation_request(self, request, response):
        """
        Service callback to handle activation or deactivation requests.
        """
        if request.data: 
            if not self.vocom_model_state:
                self.vocom_model_state = True
                self.stop_event.clear()
                self.get_logger().info("Activating Vocal Command system.")
                self.model_thread = threading.Thread(target=self.running_vocom_model, daemon=True)
                self.model_thread.start()
                response.success = True
                response.message = "Vocal Command system activated."
            else:
                response.success = False
                response.message = "Vocal Command system is already active."
        else:  
            if self.vocom_model_state:
                self.vocom_model_state = False
                self.stop_event.set()
                self.get_logger().info("Deactivating Vocal Command system.")
                response.success = True
                response.message = "Vocal Command system deactivated."
            else:
                response.success = False
                response.message = "Vocal Command system is already inactive."

        return response

    def running_vocom_model(self):
        """Run the Vocal Command system if the model is active."""
        while not self.stop_event.is_set():                                                                 # Continue tant que l'événement n'est pas déclenché
            S2T_output = self.run_S2T()
            if self.stop_event.is_set():                                                                    # Vérifie si l'arrêt a été demandé pendant S2T
                break
            # S2T_output = "Drill" # TESTING
            llm_output = self.run_T2C(S2T_output)
            if self.stop_event.is_set():  
                break
            self.check_4_json()                                                                             # (from command to Joy publication)

        self.get_logger().info("Vocal Command system has stopped.")
        if self.p:
            close_microphone_stream(self.p, self.stream)

    def run_S2T(self):
        '''
        Run the Speech to Text system.
        Input: None
        Output: text (str) - The recognized text from the Speech to Text system.
        '''
            
        # self.p, self.stream, metadata = open_microphone_stream()
        # if wait_4_wake_word(self.stream, self.wake_word, metadata, self.model_whisper_tiny_en):
        wakeword_model_path = self.models_folder+"Hey-Explore_en_linux_v3_0_0.ppn" 
        sleepword_model_path = self.models_folder+"Bye-Explore_en_linux_v3_0_0.ppn" 
        if wait_4_wake_word_TEST(os.getenv("PORCUPINE_KEY"), wakeword_model_path):
        
            if self.stop_event.is_set():
                return None

            self.get_logger().info(f"Start listening for commands for {self.delay_2_give_commands}s max. Terminate with: 'Bye Explore'")
                    
            audio, sampwidth, framerate = wait_4_sleep_word_TEST(os.getenv("PORCUPINE_KEY"), sleepword_model_path, timeout=self.delay_2_give_commands)

            if self.stop_event.is_set():
                return None
            
            # close_microphone_stream(self.p, self.stream) 
            save_audio(audio, self.results_folder+"audio.wav", sampwidth, framerate)
            # save_audio(audio, self.results_folder+"audio.wav", get_sample_size(pyaudio.paInt16), 16000)
    #             metadata = {
    #     "chunk": 1024, # 16384,                                                             # 4096, # buffer size pour flux audio (+ grand = + de latence, + petit = + de CPU)
    #     "sample_format": pyaudio.paInt16,
    #     "nchannel": 1,                                                              # 1 pour mono, 2 pour stéréo (inutile car pas besoin de spacialisation)
    #     "framerate": 16000                                                          # fréquence d'échantillonnage (Hz) (16k pour speech recognition en général, 44.1k pour musique)
    # }
            # save_audio(audio, self.results_folder+"audio.wav", self.p.get_sample_size(metadata["sample_format"]), metadata["framerate"])
            self.get_logger().info(f"Succesfully recorded command.") 

            # text, _ = recognize_speech_WHISPER(self.model_whisper_tiny_en, audio)
            # with open(self.results_folder + "text_whisper_tiny.txt", "w") as output_file:
            #     output_file.write(text)

            whisper_prompt = whisper_prompt = (
                "Listen to the audio and transcribe any commands given to a rover. "
                "Commands may involve actions like moving, turning, drilling, or activating cameras, "
                "and can include details such as distance, angle, or speed. "
                "Ignore any unrelated conversation or noise. Focus only on commands addressed to the rover."
            )

            text, _ = recognize_speech_WHISPER(self.model_whisper_base_en, audio, whisper_prompt=whisper_prompt)
            with open(self.results_folder + "text_whisper.txt", "w") as output_file:
                output_file.write(text)

            self.get_logger().info(f"Whisper Base.en: \n{text} \n")
            return text

    def run_T2C(self, S2T_output): 
        '''
        Run the Text to Command system.
        Input: S2T_output (str) - The output of the Speech to Text system.
        Output: llm_output (dict) - LLM generation (usually a JSON) saved in a file.
        '''

        llm = Llama(
            model_path= self.models_folder + "Llama-3.2-3B-Instruct-Q6_K_L.gguf",
            n_ctx = 700,
            verbose= False
        )
        system_prompt, json_schema = prepare_llama_cpp_generation()
        # self.get_logger().info(f"Prompt: \n{system_prompt}")
        # self.get_logger().info(f"JSON: \n{json_schema}")
        if self.stop_event.is_set():
            return None
        llm_output = llama_cpp_generation(llm, system_prompt, S2T_output, json_schema)

        save_json(llm_output, self.results_folder, f"commands")                                         # _1B # _HF
        self.get_logger().info(f"JSON generated (Llama 3.2 3B): \n{llm_output}\n")

        return llm_output

  # CHECK FOR JSON and PUBLISH
    def check_4_json(self):
        """Check if the commands.json file exists and process it if found."""

        self.get_logger().info(f"Checking for updates to: \n{self.json_file_path}")

        if os.path.isfile(self.json_file_path):
            current_modified_time = os.path.getmtime(self.json_file_path)

            if self.last_modified_time is None or current_modified_time > self.last_modified_time:
                self.get_logger().info("commands.json file has been updated. Reloading commands...")
                self.last_modified_time = current_modified_time
                self.load_commands_from_json()
                for cmd in self.commands:
                    self.publish_cmd_general(cmd) if cmd['command'] in ['drill', 'cameras', 'camera'] else self.publish_cmd_as_joy(cmd)
                self.commands = []                                                                      # Clear commands after publishing
        else:
            self.get_logger().warning("New commands.json not found")
            self.last_modified_time = None

    def load_commands_from_json(self):
        """Load the series of commands from commands.json."""
        try:
            with open(self.json_file_path, 'r') as f:
                data = json.load(f)
                self.commands = data.get("commands", [])
                self.nav_execution_speed = self.get_nav_speed(data)
                self.get_logger().info(f'Loaded sequence of {len(self.commands)} commands with nav command execution speed of {self.nav_execution_speed}.')
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.get_logger().error(f"Error reading commands.json: {e}")

    def get_nav_speed(self, data):
        """Get the navigation speed from the JSON file."""
        # sentiment analysis text
        # sentiment analysis audio
        nav_speed = data.get("nav_execution_speed", "default")
        nav_speed = (
            0.5 if nav_speed == "slow" 
            else 1.5 if nav_speed == "fast" 
            else 1.0 # default
        )                                              
        return nav_speed
    
    def publish_cmd_general(self, cmd): 
        if cmd['command'] == 'drill':                                                                   # publie un message comme une fake CS qui demande de lancer le drill

            # drill_msg = ScDrillCmds()
            # drill_msg.mode = 'auto'

            rotation_speed = float(cmd.get('drill_rotation_speed', 78000000))
            distance_ratio = cmd.get('drill_distance_ratio', 1.0)

            self.get_logger().info(f"Drill command 'auto': rotation speed: {rotation_speed}, distance ratio: {distance_ratio}")

            # drill_msg.send_parameter = True
            # drill_msg.rotation_speed = rotation_speed
            # drill_msg.distance_ratio = distance_ratio

            # self.publisher_drill.publish(drill_msg) 
            # publie rpour une certaine durée? 
            # check good deroulé? 
            
        elif cmd['command'] in ['cameras', 'camera']:
            new_camera_state = cmd.get('cameras_state', 'turn_on')
            self.get_logger().info(f"New camera state: {new_camera_state}")

            cam_request = SetBool.Request()
            cam_request.data = True if new_camera_state == 'turn_on' else False
            future = self.cam_client.call_async(cam_request)    # Call the service asynchronously
            future.add_done_callback(self.handle_service_response)   # Attach a callback for when the future completes

            # In https://github.com/EPFLXplore/ERC_CAMERAS/blob/245691682c4bc95f63ebdcfe069caf849a5b4708/camera/camera/camera_node.py :
            # # Service to activate the camera. For now we hardcode the parameters so we use just a SetBool
            # self.declare_parameter("topic_service", self.default)
            # self.service_topic = self.get_parameter("topic_service").get_parameter_value().string_value
            # self.service_activation = self.create_service(SetBool, self.service_topic, self.start_cameras_callback)
            # Initialize publishers to publish results: 
            # self.cam_pubs = self.create_publisher(CompressedImage, self.publisher_topic, 1, callback_group=self.callback_group)
            # self.cam_bw = self.create_publisher(Float32, self.publisher_topic_bw, 1)

    def publish_cmd_as_joy(self, cmd): 
        """Translate command into Joy message and publish the command to the topic."""
        self.joy_msg = Joy()
        self.joy_msg.axes = [0.0] * 6
        self.joy_msg.buttons = [0] * 9  
        # self.joy_msg.buttons[0] = 0                                                                   # Cross on PS4: switch between normal and lateral displacement mode
        # self.joy_msg.buttons[1] = 0                                                                   # Round on PS4: switch between manual and auto nav mode

        if cmd['command'] == 'move':
            
            if cmd['direction'] not in ['forward', 'backward']:
                return                                                                                  # Skip publishing for invalid directions
            self.distance = cmd.get("distance", 0)
            self.default_speed_moving = 0.5 * self.nav_execution_speed                                  # = 0.5 m/s ?
            self.publish_duration = self.distance / self.default_speed_moving if self.distance else 0.0

            concerned_axe = 5 if cmd['direction'] == 'forward' else 2                                   # GP_axis_R2 = 5 (forward) and GP_axis_L2 = 2 (backward)
            self.joy_msg.axes[concerned_axe] = self.default_speed_moving  
            self.get_logger().info(f"Publishing for {self.publish_duration:.2f} seconds at frequency {self.pub_freq_4_joy:.2f} and safety speed {self.default_speed_moving} m/s.")

        elif cmd['command'] == 'turn':
            if cmd['direction'] not in ['left', 'right', '180_turn', '360_turn']:
                return                                                                                  # Skip publishing for invalid directions
            if cmd['direction'] == '180_turn':
                cmd['angle'] = 180
            elif cmd['direction'] == '360_turn':
                cmd['angle'] = 360
            elif (cmd['direction'] in ['right', 'left']) and (cmd.get("angle", 0) == 0):
                cmd['angle'] = 360
                
            self.angle = np.radians(cmd.get("angle", 0)) if cmd.get("angle", 0) else 0
            self.default_speed_angle = 0.5 * self.nav_execution_speed                                   # = 0.8 rad/s?
            self.publish_duration = self.angle / self.default_speed_angle                               # angle not distance!

            self.joy_msg.buttons[7] = 1                                                                 # Rotation on itself (crab)
            concerned_axe = 0
            self.joy_msg.axes[concerned_axe] = self.default_speed_angle if cmd['direction'] == 'right' else -self.default_speed_angle # par défaut tourner a gauche si direction is 180 or 360_turn
            self.get_logger().info(f"Publishing for {self.publish_duration:.2f} seconds at frequency {self.pub_freq_4_joy:.2f} and safety speed {self.default_speed_angle} rad/s.")

        elif cmd['command'] in ['not_a_command', 'not a command']:
            return                                                                                      # Skip publishing for invalid commands
        else:
            self.get_logger().warning(f"Unknown command: {cmd['command']}")
            return                                                                                      # Skip publishing for unknown commands

        # PUBLICATION
        self.start_time = time.time()
        elapsed_time = time.time() - self.start_time
        
        while elapsed_time < self.publish_duration:
            if self.stop_event.is_set():  
                break            
            # self.get_logger().info(f"{elapsed_time:.2f} seconds elapsed.")
            self.joy_msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher_nav.publish(self.joy_msg)
            time.sleep(self.pub_freq_4_joy)
            elapsed_time = time.time() - self.start_time
        
        self.joy_msg.axes = [0.0] * 6
        self.joy_msg.buttons = [0] * 9
        self.joy_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_nav.publish(self.joy_msg)
        self.get_logger().info("Completed publishing duration. \n\n")

    def handle_service_response(self, future):
        """Handles the response from the service."""
        try:
            response = future.result()
            self.get_logger().info(f"Response: success={response.success}, message='{response.message}'")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {str(e)}")

def main(args=None):

    rclpy.init(args=args)                                                                               # Init ROS python
    node = VoCom_PubSub()                                                                               # Create a Node instance
    rclpy.spin(node)                                                                                    # Run the node in a Thread
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()