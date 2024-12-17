import json
import os 
import numpy as np
import rclpy
import vosk
from rclpy.node import Node
# from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from sensor_msgs.msg import Joy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import time
from llama_cpp import Llama
import threading

from vocal_command_pkg.utils_comb import * 

class VoCom_PubSub(Node):

    def __init__(self):

        # VARIABLES INITIALIZATION
        super().__init__('vocom_node')
        
        self.vocom_model_state = False
        self.model_thread = None
        self.stop_event = threading.Event() 

        self.pub_freq_4_joy = 0.1                                                                          # sec
        self.commands = []                                                                                
        self.last_modified_time = None                                                                      # Track the last modified time commands.json
        self.results_folder = "/home/xplore/dev_ws/src/vocal_command_pkg/models_and_results/results/"
        self.models_folder = "/home/xplore/dev_ws/src/vocal_command_pkg/models_and_results/models/"
        self.json_file_path = self.results_folder + "commands.json"
        self.p = None
        self.wake_word = ["hello", "halo", "hullo", "allo", "jello", "fellow", "yellow", "he low", "hell oh"]
        self.sleep_word = ["goodbye", "good way", "good buy", "good eye", "could buy", "good guy", "good try", "good by", "guide by", "good pie", "good bi", "good bai", "good bie", "good bae", "good bay", "good pay"]
        self.delay_2_give_commands = 30
        self.get_logger().info('\n\nHuman-Rover communication node has started. Please activate model via CS')

        # PUBLISHER
        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, depth=10)
        self.publisher_nav = self.create_publisher(Joy, "fake_cs_response", qos_profile)                       # /CS/NAV_gamepad
        # self.publisher_drill = self.create_publisher(Joy, "fake_drill_response", qos_profile)                   # /CS/DRILL_gamepad
        # self.publisher_cam = self.create_publisher(Joy, "fake_cam_response", qos_profile)                       # /CS/CAM_gamepad


        # SERVICE                                                                                           # https://docs.ros.org/en/noetic/api/std_srvs/html/srv/SetBool.html 
        self.service_ = self.create_service(SetBool, 'activation_service', self.handle_activation_request)  

        # Initialisation du modèle
        self.initialize_models()

    def initialize_models(self):
        """Initialize Vosk and Whisper models."""
        # 1. Vosk
        model_path_vosk = self.models_folder + "Vosk_Small/"                                                # "vosk-model-en-us-0.22" (40M/1.8G) (https://alphacephei.com/vosk/models)
        self.model_vosk = vosk_config(model_path_vosk)
        self.get_logger().info("Vosk Small-en ready")

        # 2. Whisper
        model_name_whisper = "base.en"                                                                      # https://github.com/openai/whisper/discussions/1463
        self.model_whisper = whisper_config(model_name_whisper)
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

        self.p, self.stream, metadata = open_microphone_stream()
        self.get_logger().info(f"\n\n\nOpened microphone stream")

        rate = metadata["framerate"]
        recognizer = vosk.KaldiRecognizer(self.model_vosk, rate)                                            # rate = metadata.fs

        self.get_logger().info(f"Speech 2 Text runnning. Waiting Wake-word: {self.wake_word[0]} \n")

        if wait_4_wake_word(self.stream, self.wake_word, metadata, recognizer):
            if self.stop_event.is_set():
                return None

            self.get_logger().info(f"Start listening for commands for {self.delay_2_give_commands}s max. Terminate with: {self.sleep_word[0]}")
        
            audio, text = wait_4_sleep_word(self.stream, 
                                                self.sleep_word,
                                                metadata, 
                                                recognizer,
                                                timeout=self.delay_2_give_commands)
            
            if self.stop_event.is_set():
                return None
            
            self.get_logger().info(f"Succesfully recorded command.") 

            save_audio(audio, self.results_folder+"audio.wav", self.p, metadata)
            with open(self.results_folder + "text_vosk.txt", "w") as output_file:
                output_file.write(text)

            text = recognize_speech_WHISPER(self.model_whisper, audio)
            with open(self.results_folder + "text_whisper.txt", "w") as output_file:
                output_file.write(text)

            close_microphone_stream(self.p, self.stream) 

            self.get_logger().info(f"Whisper: \n{text} \n")
            return text

    def run_T2C(self, S2T_output): 
        '''
        Run the Text to Command system.
        Input: S2T_output (str) - The output of the Speech to Text system.
        Output: llm_output (dict) - LLM generation (usually a JSON) saved in a file.
        '''

        llm = Llama(
            model_path= self.models_folder + "Llama-3.2-3B-Instruct-Q6_K_L.gguf",
            n_ctx = 512,
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
                    self.publish_cmd_as_joy(cmd)
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
                self.get_logger().info(f'Loaded sequence of {len(self.commands)} commands and {self.nav_execution_speed} nav command execution.')
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
    
    def publish_cmd_as_joy(self, cmd): 
        """Translate command into Joy message and publish the command to the topic."""
        self.joy_msg = Joy()
        self.joy_msg.axes = [0.0] * 6
        self.joy_msg.buttons = [0] * 9
        self.joy_msg.buttons[1] = 0                                                                     # auto state deactivate
        self.joy_msg.buttons[2] = 1                                                                     # manual state activate
        self.joy_msg.buttons[8] = 0                                                                     # change_kinematic_state

        if cmd['command'] == 'move':
            
            if cmd['direction'] not in ['forward', 'backward']:
                return                                                                                  # Skip publishing for invalid directions
            self.distance = cmd.get("distance", 0)
            self.default_speed_moving = 0.5 * self.nav_execution_speed                                                           # = 0.5 m/s ?
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
            self.default_speed_angle = 0.5 * self.nav_execution_speed                                                              # = 0.8 rad/s?
            self.publish_duration = self.angle / self.default_speed_angle                                # angle not distance!

            # rajouter le bouttons crab?  rajouter dans la doc
            concerned_axe = 0
            self.joy_msg.axes[concerned_axe] = self.default_speed_angle if cmd['direction'] == 'right' else -self.default_speed_angle # par défaut tourner a gauche si direction is 180 or 360_turn
            self.get_logger().info(f"Publishing for {self.publish_duration:.2f} seconds at frequency {self.pub_freq_4_joy:.2f} and safety speed {self.safety_speed_angle} rad/s.")

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

def main(args=None):

    rclpy.init(args=args)                                                                               # Init ROS python
    node = VoCom_PubSub()                                                                               # Create a Node instance
    rclpy.spin(node)                                                                                    # Run the node in a Thread
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()