import numpy as np
import wave
import pyaudio
import json
import vosk
import whisper
import warnings
# from jiwer import wer
import json
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Union, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import re 
import torch
import ollama

import websocket
import json

## SPEECH 2 TEXT ##################################################

# PyAudio stream
def open_microphone_stream():
    """Ouvre et retourne un flux audio depuis le microphone."""
    
    # Advanced Linux Sound Architecture (ALSA) : API for sound card device drivers.
    # arecord -l in terminal to list audio devices
    p = pyaudio.PyAudio()                       # https://people.csail.mit.edu/hubert/pyaudio/docs/#pyaudio.PyAudio.Stream.__init__ 
    
    metadata = { 
        "chunk": 16384,  # 4096, # buffer size pour flux audio (+ grand = + de latence, + petit = + de CPU)
        "sample_format": pyaudio.paInt16,
        "nchannel": 1, # 1 pour mono, 2 pour stéréo (inutile car pas besoin de spacialisation)
        "framerate": 16000 # fréquence d'échantillonnage (Hz) (16k pour speech recognition en général, 44.1k pour musique)
    }
    for i in range(p.get_device_count()):
        print(p.get_device_info_by_index(i).get('name'))
        print(p.get_device_info_by_index(i))
        print()
    try: 
        stream = p.open(format=metadata["sample_format"],
                    channels=metadata["nchannel"],
                    rate=metadata["framerate"],
                    input=True,
                    frames_per_buffer=metadata["chunk"]) 
    except Exception as e:
        print(f"Error lors de l'ouverture du flux audio: {e}")
        p.terminate()
        return None, None, None
        
    return p, stream, metadata

def close_microphone_stream(p, stream):
    """Ferme le flux audio et termine PyAudio."""
    stream.stop_stream()
    stream.close()
    p.terminate()
            
# Vosk and Whisper configuration and speech recognition functions

def whisper_config(model_name):
    """Initialise le modèle et le recognizer Whisper."""
    warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
    whisper_model = whisper.load_model(model_name)
    return whisper_model

def vosk_config(model_path):
    """Initialise le modèle et le recognizer Vosk."""
    vosk.SetLogLevel(-1) # -1 pour désactiver les logs
    vosk_model = vosk.Model(model_path)
    return vosk_model

# Fonctions Speech recognition
def recognize_speech_WHISPER(model, audio, word_2_rec=None, audio_path=None): # Use prompt to recognize some specific words or enhance the recognition? https://platform.openai.com/docs/guides/speech-to-text 
    """Processes audio data and checks for the wake word."""
    if audio_path:
        # Charger et préparer l'audio depuis un fichier si un chemin est fourni
        audio = whisper.load_audio(audio_path)
    else:
        # Convertir les données `bytes` en un tableau numpy de type float32 normalisé
        audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

    text = model.transcribe(audio)['text']
    word_2_rec_bool = any(w in text.lower() for w in word_2_rec)
    if text:
        print(f"Whisper: {text}")
    return text, word_2_rec_bool

def listen_wake_word_ws(stream, word_2_rec, metadata, ws_url):
    """Listens for the wake word via Vosk WebSocket."""
    ws = websocket.create_connection(ws_url)
    try:
        while True:
            # Read audio chunk
            audio = stream.read(metadata["chunk"] * 4, exception_on_overflow=True)

            # Send audio to Vosk WebSocket server
            ws.send_binary(audio)
            result = ws.recv()
            result = json.loads(result)

            # Check if the wake word is detected
            if "text" in result:
                text = result["text"]
                print(f"Vosk: {text}")
                if any(w in text.lower() for w in word_2_rec):
                    return True
    finally: # gets executed either exception is generated or not
        ws.close()

def listen_wake_word_local(stream, word_2_rec, metadata, recognizer):
    """Lit l'audio du flux et applique une fonction de reconnaissance."""
    while True:
        # durée d'un chunk = (chunk_size/framerate) = 16384/16000 = 1.024s
        audio = stream.read(metadata["chunk"]*4, exception_on_overflow=True) 

        _, wake_word_detected = recognize_speech_VOSK_local(recognizer, audio, word_2_rec)

        if wake_word_detected:
            return True

def recognize_speech_VOSK_local(recognizer, audio, word_2_rec):
    """Processes audio data and checks for the wake word."""
    recognizer.AcceptWaveform(audio)
    result = json.loads(recognizer.Result())
    text = result.get("text", "")
    word_2_rec_bool = any(w in text.lower() for w in word_2_rec)
    if text:
        print(f"Vosk: {text}")
    return text, word_2_rec_bool

def recognize_speech_VOSK_ws(ws_url, audio, word_2_rec):
    """Processes audio data via Vosk WebSocket and checks for the specified word."""
    ws = websocket.create_connection(ws_url)
    try:
        # Send audio to Vosk WebSocket server
        ws.send_binary(audio)
        result = ws.recv()
        result = json.loads(result)

        text = result.get("text", "")
        word_2_rec_bool = any(w in text.lower() for w in word_2_rec)

        if text:
            print(f"Vosk: {text}")

        return text, word_2_rec_bool
    finally:
        ws.close()

def process_audio_stream_local(stream, word_2_rec, metadata, recognizer, ws_url=None, timeout=30):
    """Processes the audio stream and applies speech recognition."""

    audio_chunks, text_chunks = [], []
    while True:
        audio = stream.read(metadata["chunk"]*4, exception_on_overflow=True) 

        text, wake_word_detected = recognize_speech_VOSK_local(recognizer, audio, word_2_rec)

        audio_chunks.append(audio)
        text_chunks.append(text)

        # If audio chunks accumulated are more than timeout seconds, return the results
        if len(b''.join(audio_chunks)) / metadata["framerate"] > timeout*2:
            print(f"Max audio duration of {timeout}s reached. Processing audio...")
            return b''.join(audio_chunks), ' '.join(text_chunks)

        if wake_word_detected:
            return b''.join(audio_chunks), ' '.join(text_chunks)

def process_audio_stream_ws(stream, word_2_rec, metadata, ws_url, timeout=30):
    """Processes the audio stream and applies speech recognition."""

    audio_chunks, text_chunks = [], []
    while True:
        audio = stream.read(metadata["chunk"]*4, exception_on_overflow=True) 

        text, wake_word_detected = recognize_speech_VOSK_ws(ws_url, audio, word_2_rec)

        audio_chunks.append(audio)
        text_chunks.append(text)

        # If audio chunks accumulated are more than timeout seconds, return the results
        if len(b''.join(audio_chunks)) / metadata["framerate"] > timeout*2:
            print(f"Max audio duration of {timeout}s reached. Processing audio...")
            return b''.join(audio_chunks), ' '.join(text_chunks)

        if wake_word_detected:
            return b''.join(audio_chunks), ' '.join(text_chunks)
        
# Fonctions d'audio
def save_audio(audio, filepath, p, metadata):
    """Sauvegarde les données audio dans un fichier WAV."""
    wf = wave.open(filepath, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(metadata["sample_format"]))
    wf.setframerate(metadata["framerate"])
    wf.writeframes(audio)
    wf.close()

def open_wav_file(file_path):
    '''Ouvre un fichier WAV et retourne les données audio en mono.'''
    with wave.open(file_path) as wav_file:
        metadata = wav_file.getparams()
        frames = wav_file.readframes(metadata.nframes)
        mono_frames = stereo_to_mono(frames, metadata)
    return mono_frames, metadata

def stereo_to_mono(frames, metadata):
    '''Convertit des frames stéréo en mono.'''
    # Convert the raw byte data to a NumPy array
    audio_data = np.frombuffer(frames, dtype=np.int16)
    
    if metadata.nchannels == 2:
        # Stereo: reshape into 2D array (n_samples, 2 channels)
        audio_data = np.reshape(audio_data, (-1, 2))
        # Convert to mono by averaging the two channels
        mono_data = audio_data.mean(axis=1).astype(np.int16)
        return mono_data.tobytes()
    else:
        # Already mono, no conversion needed
        return frames

def read_text_file(file_path):
    '''Lit un fichier texte et retourne une liste de lignes.'''
    with open(file_path, "r") as file:
        text = file.read()
    return text

# def calculate_model_wer(Text_GT_path, Text_Rec_path, intro_text=""):
#     '''Calculates the WER for a given model and audio file.'''
#     Text_GT = read_text_file(Text_GT_path)
#     Text_Rec = read_text_file(Text_Rec_path)
#     print(f"{intro_text}{wer(Text_GT, Text_Rec):.2f}")

## TEXT 2 COMMAND ##################################################

# Loading model and generating text
def call_llama(prompt, model_name="llama3.2"):
    '''Call the Ollama command line tool to generate JSON from text prompt'''
    response = ollama.generate(model=model_name, prompt=prompt, stream=False)
    return response["response"]

def load_tokenizer_model(path_to_cache, model_name): # "meta-llama/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=path_to_cache)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=path_to_cache)
    return tokenizer, model

def generate_text(model, tokenizer, prompt, device, max_length=260):
    # Encode input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    inputs_length = inputs["input_ids"].size(1)
    attention_mask = inputs['attention_mask'].to(device)

    # Generate output (using greedy search or other decoding strategies like beam search)
    outputs = model.generate(
        inputs["input_ids"], 
        attention_mask=attention_mask,
        max_length=inputs_length + max_length,  # Max length of generated text
        num_return_sequences=1,  # Return 1 sequence
        # no_repeat_ngram_size=2,  # Avoid repeating bigrams
        temperature=0.6,  # Make the output a little more deterministic
        top_p=0.9,  # Use nucleus sampling
        # repetition_penalty=1.2,  # Apply penalty for repetition
        do_sample=True,  # Use sampling instead of greedy
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode the generated token IDs back to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text_without_prompt = generated_text[len(prompt):].strip()
    
    return generated_text_without_prompt

def get_prompt(prompt_name, input):
    if input is None:
        input = ""  # Or provide some default value if necessary

    if prompt_name == "final":
        prompt = """You are assisting a rover's navigation and control system to interpret user instructions accurately. Your task is to identify specific commands (such as "move," "turn", or "not a command") and their details (such as distance or angle) from the provided instructions. 

The output should be a valid JSON instance that matches the schema below. Do not include any explanations, preambles, code snippets, or additional input-output examples. Only provide one JSON output in response to the last input. If no valid command is found, generate a single "commands" instance where all fields are null.

Here is the JSON schema:

{
  "commands": [
    {
      "command": "string (either 'displacement', 'rotation' or 'not_a_command')",
      "direction": "string (either 'forward', 'backward', 'right', 'left', '180_turn', '360_turn' or null, depending on the command)",
      "distance": "positive integer (Distance in meters, or null)",
      "angle": "positive integer (Rotation angle in degrees, or null)"
    }
  ]
}

For each command, only fill in the relevant fields based on "command":
- For "not_a_command": include nothing.
- For "displacement": include "direction" (either 'forward' or 'backward') and "distance" in meters.
- For "rotation": include "direction" (either 'right', 'left', '180_turn' or '360_turn') and "angle" in degrees.

Examples:

Input: "Go ahead by 6 meters, make a U-turn, drive backwards a little, maybe one meter. Then rotate 60 degrees to the left."
Output:
{
  "commands": [
    {
      "command": "move",
      "direction": "forward",
      "distance":6
    },
    {
      "command": "turn",
      "direction": "180_turn",
      "angle": 180
    },
    {
      "command": "move",
      "direction": "backward",
      "distance":1
    },
    {
      "command": "turn",
      "direction": "left",
      "angle": 60
    }
  ]
}

Input: "Please spin on yourself 1/3 to the left and travel back for 8 meters."
Output:
{
  "commands": [
    {
      "command": "turn",
      "direction": "left",
      "angle":120
    },
    {
      "command": "move",
      "direction": "backward",
      "distance": 8
    }
  ]
}

Input: "Please ensure all systems are operating normally. Here is a phrase that has nothing to do with rover commands."
Output:
{
  "commands": [
    {
      "command": "not a command"
    }
  ]
}

Input: " """ + input + """ "
Output:
"""
    return prompt

def prepare_llama_cpp_generation():
    class RoverCommands(BaseModel):
        command: Literal["move", "turn", "not a command"] = Field(description="Type of command to execute")
        direction: Optional[Literal["forward", "backward", "right", "left", "180_turn", "360_turn"]] = Field(None, description="Direction of movement or rotation, depending on the command")
        angle: Optional[int] = Field(None, description="Rotation angle in degrees")
        distance: Optional[int] = Field(None, description="Distance to cover in meters")
        
    class MissionPlan(BaseModel):
        commands: List[RoverCommands] = Field(description="List of commands to be executed by the rover")

    json_schema = MissionPlan.model_json_schema()

    system_prompt = """You are assisting a rover's navigation and control system to interpret user instructions accurately. Your task is to identify specific commands (such as "move," "turn", or "not a command") and their details (such as distance or angle) from the provided instructions. 

Examples:

Input: "Spin on yourself 1/3 to the right"
Output:
{
  "commands": [
    {
      "command": "turn",
      "direction": "right",
      "angle": 120
    }
  ]
}

Input: "Please turn left 90 degrees and then drive backwards a little, maybe one meter."
Output:
{
  "commands": [
    {
      "command": "turn",
      "direction": "left",
      "angle": 90
    },
    {
      "command": "move",
      "direction": "backward",
      "distance": 1
    }
  ]
}

Input: "Here is a phrase that has nothing to do with rover commands."
Output:
{
  "commands": [
    {
      "command": "not a command"
    }
  ]
}"""

    # print(json.dumps(json_schema, indent=2)) 
    return system_prompt, json_schema

def llama_cpp_generation(llm, system_prompt, query, json_schema):
    completion = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {   "role": "user", 
                "content": query
            },
        ],
        response_format={
            "type":"json_object",
            "schema": json_schema
        },
        temperature=0.2 # default value
    )
    return completion['choices'][0]["message"]["content"]

# Functions about pseudo JSON output
def save_json(llm_output, file_path, file_name):

    try:
        
        # Check if `llm_output` is already a dictionary or JSON string
        if isinstance(llm_output, str):
            # If it's a string, parse it as JSON
            json_data = json.loads(llm_output)
        elif isinstance(llm_output, dict):
            # If it's already a dictionary, use it as is
            json_data = llm_output
        else:
            raise TypeError("llm_output must be a JSON-formatted string or dictionary.")
        
        # If successful, save as JSON
        text_file_name = file_path + file_name + '.json'
        with open(text_file_name, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
        print(f"Output saved as JSON to '{text_file_name}'.")
    
    except json.JSONDecodeError:
        # If llm_output is not a valid JSON, save as plain text
        text_file_name = file_path + file_name + '.txt'
        with open(text_file_name, 'w') as text_file:
            text_file.write(llm_output)
        print(f"Output is not valid JSON. Saved as text to '{text_file_name}'.")

def validate_json_struct(llm_output, json_schema='flat'):
    if json_schema == 'flat':

        class RoverCommands(BaseModel):
            execution_order: int = Field(description="Execution order of the command")
            command_type: Literal["displacement", "rotation"] = Field(description="Type of command to execute")
            move_direction: Optional[Literal["forward", "backward"]] = Field(None, description="Direction of movement")
            distance: Optional[int] = Field(None, description="Distance to cover, in meters")
            turn_direction: Optional[Literal["right", "left", "u-turn", "complete rotation"]] = Field(None, description="Rotation direction")
            angle: Optional[int] = Field(None, description="Rotation angle in degrees")
            
            # Conditional logic: If the command type is "displacement", "distance" and "direction" should be set
            # If the command type is "rotation", "angle" and "direction" should be set
            # ajouter une validation des types de commandes
            def validate(self):
                if self.command_type == "displacement" and not self.move_direction:
                    raise ValueError("Displacement command must have direction.")
                elif self.command_type == "rotation" and not self.turn_direction:
                    raise ValueError("Rotation command must have direction.")
                
        class MissionPlan(BaseModel):
            commands: List[RoverCommands] = Field(description="List of commands to be executed by the rover")
    
    if json_schema == "nested":

        class DisplacementCommand(BaseModel):
            move_direction: Literal["forward", "backward"] = Field(description="Direction of movement")
            distance: Optional[int] = Field(None, description="Distance to cover, in meters")

        class RotationCommand(BaseModel):
            turn_direction: Literal["right", "left", "u-turn", "complete rotation"] = Field(description="Rotation direction")
            angle: Optional[int] = Field(None, description="Rotation angle in degrees")

        class RoverCommands(BaseModel):
            execution_order: int = Field(description="Execution order of the command")
            command_type: Literal["displacement", "rotation"] = Field(description="Type of command to execute")
            command_info: Union[DisplacementCommand, RotationCommand] = Field(description="Specific details of the command") # Union is used to specify that the command_info field can be of either DisplacementCommand or RotationCommand type

            # Conditional logic: If the command type is "displacement", "distance" and "direction" should be set
            # If the command type is "rotation", "angle" and "direction" should be set
            # ajouter une validation des types de commandes
            def validate(self):
                if self.command_type == "displacement" and not self.command_info.move_direction:
                    raise ValueError("Displacement command must have direction and distance.")
                elif self.command_type == "rotation" and not self.command_info.turn_direction:
                    raise ValueError("Rotation command must have direction and angle.")
                
        class MissionPlan(BaseModel):
            commands: List[RoverCommands] = Field(description="List of commands to be executed by the rover")

    try:
        parsed_output = json.loads(llm_output)["commands"]
        
        # Validate each item in the list using the Pydantic model
        for command in parsed_output:
            rover_command = RoverCommands(**command) # Convert the dictionary command to a Pydantic model instance
            rover_command.validate()                # Custom validation logic

        print("Validation succeeded! The output is well-formed.")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Invalid JSON format: {e}")
    except ValidationError as e: 
        print(f"Validation error: {e}")
    except ValueError as e: # erreurs générique
        print(f"Custom validation error: {e}")

def extract_first_json(response_text):
    # This pattern captures a balanced JSON block by finding a substring from the first '{' to its corresponding '}'
    json_pattern = r"\{(?:[^{}]*|\{[^{}]*\})*\}"
    matches = re.findall(json_pattern, response_text)
    
    if matches:
        return matches[0]  # Return only the first matched JSON block
    else:
        print("No JSON found in the response.")
        return None