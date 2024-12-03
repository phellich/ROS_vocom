import json
import os
import numpy as np
import rclpy
import vosk
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Joy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import time
# from llama_cpp import Llama
import torch

from vocal_command_pkg.utils_comb import * 

class VoCom_PubSub(Node):

    def __init__(self):

        # VARIABLES INITIALIZATION
        super().__init__('vocom_node')
        self.json_file_path = os.path.join(os.path.dirname(__file__), "commands.json")
        self.vocom_model_state = False
        self.safety_speed = 0.6
        self.commands = []                      # Initialize as an empty list
        self.last_modified_time = None          # Track the last modified time
        self.output_folder = "outputs/"
        self.p = None
        self.get_logger().info('Node has been started. Vocom_model_state is False')

        # PUBLISHER
        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, depth=10)
        self.publisher_ = self.create_publisher(Joy, "fake_cs_response", qos_profile) # /CS/NAV_gamepad
        self.publisher_.publish(Joy()) # Initialize the publisher with an empty message	

        # SUBSCRIBER 
        self.subscription_ = self.create_subscription(Bool, 'fake_cs_cmd', self.listener_callback_CS, 10) # /CS/vocom_state

        # download all models in a cache before running

        # Initialisation du modèle 

        # 1. Vosk
        # local 
        model_path_vosk = "/home/xplore/dev_ws/src/vocal_command_pkg/vocal_command_pkg/Vosk_Small/" # "vosk-model-en-us-0.22" (40M/1.8G) (https://alphacephei.com/vosk/models)
        self.model_vosk = vosk_config(model_path_vosk)
        # via vosk server websocket
        # self.ws_url = "ws://localhost:2700"  # Vosk server WebSocket URL
        self.get_logger().info("Vosk Small-en ready")

        # 2. Whisper
        model_name_whisper = "base.en" 
        self.model_whisper = whisper_config(model_name_whisper)
        self.get_logger().info("Whisper Base.en ready \n")

    # SUBSCRIBER CALLBACK OF CS
    def listener_callback_CS(self, msg):
        '''Callback function for the sub to the Control Station deiciding the Vocal Command system state.'''
        if msg.data: # True because msg is of type Bool
            self.vocom_model_state = True
            self.get_logger().info("Activating Vocal Command system.")
        else:
            self.vocom_model_state = False
            self.get_logger().info("Deactivating Vocal Command system.")
        self.running_vocom_model()

    def running_vocom_model(self):
        '''Run the Vocal Command system if the model is active.'''
        while self.vocom_model_state:
            S2T_output = self.run_S2T() 
            llm_output = self.run_T2C(S2T_output, interfere_with_model_with="ollama") # HF_transfo ollama llamacpp
            self.check_4_json() # (from command to Joy publication)

        self.get_logger().info("Vocal Command system is inactive.")
        if self.p:
            close_microphone_stream(self.p, self.stream)

    def run_S2T(self):
        '''
        Run the Speech to Text system.
        Input: None
        Output: text (str) - The recognized text from the Speech to Text system.
        '''

        wake_word = ["hello", "halo", "hullo", "allo", "jello", "fellow", "yellow", "he low", "hell oh"]
        sleep_word = ["goodbye", "dubai", "good buy", "good eye", "could buy", "good guy", "good try", "good by", "guide by"]
        delay_for_commands = 30

        self.p, self.stream, metadata = open_microphone_stream()
        self.get_logger().info(f"Opened microphone stream")

        # Si Vosk en local et non pas en websocket
        rate = metadata["framerate"]
        recognizer = vosk.KaldiRecognizer(self.model_vosk, rate) # metadata.fs

        self.get_logger().info(f"Speech 2 Text runnning. Waiting for wake word {wake_word[0]} \n")

        if listen_wake_word_local(self.stream, wake_word, metadata, recognizer):
            self.get_logger().info(f"Start listening for commands for {delay_for_commands}s max. Terminate with: {sleep_word}")
        
            audio, text = process_audio_stream_local(self.stream, 
                                                    sleep_word,
                                                    metadata, 
                                                    recognizer,
                                                    timeout=delay_for_commands)
            
            self.get_logger().info(f"Succesfully recorded command.") 

            save_audio(audio, self.output_folder+"audio.wav", self.p, metadata)
            with open(self.output_folder+"text_vosk.txt", "w") as output_file:
                output_file.write(text)

            text, _ = recognize_speech_WHISPER(self.model_whisper, audio)
            with open(self.output_folder+"text_whisper.txt", "w") as output_file:
                output_file.write(text)

            close_microphone_stream(self.p, self.stream) 

            self.get_logger().info(f"Returning text: {text}")
            return text

    def run_T2C(self, S2T_output, interfere_with_model_with="ollama"): 
        '''
        Run the Text to Command system.
        Input: S2T_output (str) - The output of the Speech to Text system.
        Output: llm_output (dict) - LLM generation (usually a JSON) saved in a file.
        '''

        if interfere_with_model_with == "ollama":
            prompt_name = "final"
            prompt = get_prompt(prompt_name, S2T_output)
            # self.get_logger().info(f"Prompt used: {prompt_name}")
            llm_output = call_llama(prompt, model_name="llama3.2") # "llama3.2:1b"
        
        elif interfere_with_model_with == "HF_transfo":
            prompt_name = "final"
            prompt = get_prompt(prompt_name, S2T_output)
            # self.get_logger().info(f"Prompt used: {prompt_name}")
            tokenizer, model = load_tokenizer_model("T2C/Tests_Notebooks/model_cache/", "meta-llama/Llama-3.2-3B")

            self.get_logger().info(f"torch.cuda.is_available is {torch.cuda.is_available()}")  # Doit renvoyer True si CUDA est activé
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.get_logger().info(torch.xpu.is_available())  # Doit renvoyer True si CUDA est activé
            # device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
            model.to(device)   
            
            llm_output_long = generate_text(model, tokenizer, prompt, device, max_length=260) # around 50 tokens per commands
            save_json(llm_output_long, self.output_folder, f"Commands_interm_HF") # _1B # _HF
            llm_output = extract_first_json(llm_output_long)

        else: # llamacpp => on .venv environment
            llm = Llama(
                model_path="model_cache/Llama-3.2-3B-Instruct-Q6_K_L.gguf",
                verbose= False
            )
            system_prompt, json_schema = prepare_llama_cpp_generation()
            llm_output = llama_cpp_generation(llm, system_prompt, S2T_output, json_schema)

        save_json(llm_output, self.output_folder, f"Commands") # _1B # _HF
        validate_json_struct(llm_output, "flat") 
        self.get_logger().info(f"JSON generated (Llama 3.2 3B): \n{llm_output}\n\n")

        return llm_output

  # CHECK FOR JSON and PUBLISH
    def check_4_json(self):
        """Check if the commands.json file exists and process it if found."""

        self.get_logger().info(f"Checking for updates to: {self.json_file_path}")

        if os.path.isfile(self.json_file_path):
            current_modified_time = os.path.getmtime(self.json_file_path)

            # If the file is new or has been modified, reload the commands
            if self.last_modified_time is None or current_modified_time > self.last_modified_time:
                self.get_logger().info("commands.json file has been updated. Reloading commands...")
                self.last_modified_time = current_modified_time
                self.load_commands_from_json()
                for cmd in self.commands:
                    self.publish_cmd(cmd)
                self.commands = []  # Clear commands after publishing
        else:
            self.get_logger().warning("commands.json file not found!")
            self.last_modified_time = None

    def load_commands_from_json(self):
        """Load the series of commands from commands.json."""
        try:
            with open(self.json_file_path, 'r') as f:
                data = json.load(f)
                self.commands = data.get("commands", [])
                self.get_logger().info(f'Loaded {len(self.commands)} commands.')
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.get_logger().error(f"Error reading commands.json: {e}")

    def publish_cmd_as_joy(self, cmd): 
        """Publish the command to the topic."""
        self.joy_msg = Joy()
        self.joy_msg.axes = [0.0] * 6
        self.joy_msg.buttons = [0] * 9
        self.joy_msg.buttons[1] = 0 # auto state deactivate
        self.joy_msg.buttons[2] = 1 # manual state activate
        self.joy_msg.buttons[8] = 0 # change_kinematic_state

        if cmd['command'] == 'move':
            
            if cmd['direction'] not in ['forward', 'backward']:
                return # Skip publishing for invalid directions
            self.distance = cmd.get("distance", 0)
            self.safety_speed_distance = 0.5 # = 0.5 m/s ?
            self.publish_duration = self.distance / self.safety_speed_distance

            concerned_axe = 5 if cmd['direction'] == 'forward' else 2
            self.joy_msg.axes[concerned_axe] = self.safety_speed # pas le meme axe si pos ou neg 

        elif cmd['command'] == 'turn':
            self.safety_speed = 0.6
            if cmd['direction'] not in ['left', 'right', '180_turn', '360_turn']:
                return # Skip publishing for invalid directions
            if cmd['direction'] == '180_turn':
                cmd['angle'] = 180
            elif cmd['direction'] == '360_turn':
                cmd['angle'] = 360
            elif (cmd['direction'] in ['right', 'left']) and (cmd.get("angle", 0) == 0):
                cmd['angle'] = 360
                
            self.angle = np.radians(np.ndarray(cmd.get("angle", 0)))[0]
            self.safety_speed_angle = 0.3 # = 0.3 rad/s?
            self.publish_duration = self.angle / self.safety_speed_angle # angle not distance!

            concerned_axe = 0
            self.joy_msg.axes[concerned_axe] = self.safety_speed if cmd['direction'] == 'right' else -self.safety_speed # par défaut tourner a gauche si direction is 180 or 360_turn

        elif cmd['command'] == 'not_a_command':
            return # Skip publishing for invalid commands
        else:
            self.get_logger().warning(f"Unknown command: {cmd['command']}")
            return  # Skip publishing for unknown commands

        self.get_logger().info(f"Publishing for {self.duration:.2f} seconds at safety speed {self.safety_speed} m/s.")
        self.start_time = time.time()
        self.pub_timer = self.create_timer(0.1, self.continuous_publish_joy)

    def continuous_publish_joy(self):
        '''Publish the joystick message during a fixed time and at a fixed rate.'''
        elapsed_time = time.time() - self.start_time

        if elapsed_time > self.publish_duration:
            self.get_logger().info("Completed publishing duration.")
            self.destroy_timer(self.pub_timer)
            return
        
        self.joy_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(self.joy_msg)
        self.get_logger().info(f"Published joystick message at elapsed time: {elapsed_time:.2f} seconds.")


def main(args=None):
    rclpy.init(args=args)   # Init ROS python
    node = VoCom_PubSub()  # Create a Node instance
    rclpy.spin(node)  # Run the node in a Thread
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()