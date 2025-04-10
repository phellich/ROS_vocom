import numpy as np
import wave
import pyaudio
import json
import whisper
import warnings
import json
from pydantic import BaseModel, Field # https://medium.com/@marcnealer/a-practical-guide-to-using-pydantic-8aafa7feebf6
from typing import Literal, Optional, List
import pvporcupine
import struct
import time

################################################
## SPEECH 2 TEXT                              ##
################################################

# Whisper configuration 
def whisper_config(model_name):
    """Initialise Whisper model."""
    warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
    whisper_model = whisper.load_model(model_name)
    return whisper_model

# Speech recognition
def recognize_speech_WHISPER(model, audio, audio_path=None, whisper_prompt=None):                        # Use prompt to recognize some specific words or enhance the recognition? https://platform.openai.com/docs/guides/speech-to-text
    """Processes audio data and checks for the wake word."""
    if audio_path:
        # Charger et préparer l'audio depuis un fichier si un chemin est fourni
        audio = whisper.load_audio(audio_path)
    else:
        # Convertir les données `bytes` en un tableau numpy de type float32 normalisé
        audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

    # https://github.com/openai/whisper/blob/main/whisper/transcribe.py
    text = model.transcribe(audio,
                            initial_prompt=whisper_prompt
                            )['text']

    return text

def wait_4_wake_word_porcupine(porcupine_key, model_path):
    """
    Processes audio and detects the Porcupine wake word.
    
    Args:
        porcupine_key (str): Porcupine access key.
        model_path (str): Path to the wake word model file (.ppn).

    Returns:
        bool: True if the wake word is detected.
    """
    
    # https://picovoice.ai/docs/quick-start/porcupine-python/
    # https://medium.com/@rohitkuyadav2003/building-a-hotword-detection-with-porcupine-and-python-f95de3b8278d 
    
    porcupine = None
    audio_stream = None
    p = None
    keyword_index = -1  # Initialize to avoid unbound variable issues
    try:
        # Create the Porcupine wake word detector
        porcupine = pvporcupine.create(
            access_key=porcupine_key,
            keyword_paths=[model_path]
        )
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        audio_stream = p.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )
        
        while True:
            # Read audio stream
            pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            
            # Process audio for wake word detection
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                break

    except Exception as e:
        print(f"ERROR OCCURED: {e}")
    finally:
        # Clean up resources
        if porcupine is not None:
            porcupine.delete()
        if audio_stream is not None:
            audio_stream.close()
        if p is not None:
            p.terminate()

        return keyword_index >= 0

def wait_4_sleep_word_porcupine(porcupine_key, model_path, timeout=30):
    """
    Processes audio to detect the sleep word. Captures audio chunks and returns the full audio 
    if the sleep word is detected or if the timeout is reached.
    
    Args:
        porcupine_key (str): Porcupine access key.
        model_path (str): Path to the sleep word model file (.ppn).
        timeout (int): Maximum duration (in seconds) to listen before returning audio.

    Returns:
        tuple: (audio_data (bytes), sample_width (int), frame_rate (int)).
    """
    porcupine = None
    audio_stream = None
    p = None
    keyword_index = -1  # Initialize to avoid unbound variable issues
    sampwidth = 2       # Default for pyaudio.paInt16 (16-bit audio)
    framerate = None
    audio_chunks = []

    try:
        porcupine = pvporcupine.create(
            access_key=porcupine_key,
            keyword_paths=[model_path]
        )
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        audio_stream = p.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )
        
        sampwidth = p.get_sample_size(pyaudio.paInt16)
        framerate = porcupine.sample_rate
        
        # Monitor audio for sleep word or timeout
        start_time = time.time()
        while True:
            # Read and store audio chunks
            pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
            audio_chunks.append(pcm)
            
            # Process audio chunk for sleep word detection
            pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)
            keyword_index = porcupine.process(pcm_unpacked)
            if keyword_index >= 0:
                break
            
            # Check if timeout is reached
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                break

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if porcupine is not None:
            porcupine.delete()
        if audio_stream is not None:
            audio_stream.close()
        if p is not None:
            p.terminate()
        
        audio_data = b"".join(audio_chunks)
        return audio_data, sampwidth, framerate

# Audio handling
def save_audio(audio, filepath, sampwidth, framerate):
    """Save audio to a WAV file."""
    wf = wave.open(filepath, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(audio)
    wf.close()

def open_wav_file(file_path):
    '''Open a WAV file and return the audio frames and metadata in mono.''' 
    with wave.open(file_path) as wav_file:
        metadata = wav_file.getparams()
        frames = wav_file.readframes(metadata.nframes)
        mono_frames = stereo_to_mono(frames, metadata)
    return mono_frames, metadata

def stereo_to_mono(frames, metadata):
    '''Convert stereo audio to mono by averaging the two channels.'''

    audio_data = np.frombuffer(frames, dtype=np.int16)                              # Convert the raw byte data to a NumPy array

    if metadata.nchannels == 2:
        audio_data = np.reshape(audio_data, (-1, 2))                                # Stereo: reshape into 2D array (n_samples, 2 channels)
        mono_data = audio_data.mean(axis=1).astype(np.int16)                        # Convert to mono by averaging the two channels
        return mono_data.tobytes()
    else:                                                                           # Already mono, no conversion needed
        return frames

################################################
## TEXT 2 COMMAND                             ##
################################################

def prepare_llama_cpp_generation():
    """Prepare the system prompt and JSON schema for LlamaCPP generation."""

    class RoverCommand(BaseModel):
        command: Literal["move", "turn", "drill", "cameras", "not_a_command"] = Field(description=("Type of command to execute."))
        direction: Optional[Literal["forward", "backward", "right", "left", "180_turn", "360_turn"]] = Field(None, description="Direction of movement or rotation, depending on the command")
        execution_speed: Optional[Literal["fast", "slow", "default"]] = Field(None, description="Execution speed for the mission")
        distance: Optional[float] = Field(None, description="Distance in meters, null otherwise")
        angle: Optional[float] = Field(None, description="Rotation angle in degrees, null otherwise")
        # camera_toggle: Optional[Literal["turn_on", "turn_off"]] = Field(None, description="Indicates whether to turn the cameras on or off, null otherwise")
        # drill_rotation_speed: Optional[int] = Field(None, description="Optional drill rotation speed, null otherwise") # in what unit??
        # drill_distance_ratio: Optional[float] = Field(None, description="Optional drill distance ratio, null otherwise") # ratio of what??

    class MissionPlan(BaseModel):
        commands: List[RoverCommand] = Field(description="List of commands to be executed by the rover")

    json_schema = MissionPlan.model_json_schema()
    # print(json_schema)

    system_prompt = """You are assisting a rover's navigation and control system to interpret user instructions accurately. Your task is to identify specific commands (such as "move," "turn", "drill" or "cameras") and their details (such as distance, angle or execution speed) from the provided instructions.

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

Input: "Advance quickly 12 meters, stop, turn on the cameras and finally drill a hole."
Output:
{
  "commands": [
    {
      "command": "move",
      "direction": "forward",
      "execution_speed": "fast",
      "distance": 12,
    },
    {
      "command": "cameras",
      "camera_toggle": "turn_on"
    },
    {
      "command": "drill"
    }
  ]
}

Input: "Please turn slowly left 90 degrees and then drive backwards a little, maybe one meter."
Output:
{
  "commands": [
    {
      "command": "turn",
      "direction": "left",
      "execution_speed": "slow",
      "angle": 90
    },
    {
      "command": "move",
      "direction": "backward",
      "distance": 1
    }
  ]
}"""

    return system_prompt, json_schema

def llama_cpp_generation(llm, system_prompt, query, json_schema, temperature = 0.2):
    """
    Generate responses using LlamaCPP with structured JSON schema.

    Args:
        llm (Llama): The initialized LlamaCPP object.
        system_prompt (str): The system prompt to guide the conversation.
        query (str): The user query for the assistant.
        json_schema (dict): JSON schema to structure the response.
        temperature (float): Sampling temperature. Default is 0.2.
    
    Returns:
        str: The generated response content.
    """
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
        temperature=temperature 
    )
    return completion['choices'][0]["message"]["content"]

def save_json(llm_output, file_path, file_name):
    """Save the LLM output as a JSON file or plain text if not valid JSON."""

    try:

        # Check if `llm_output` is already a dictionary or JSON string
        if isinstance(llm_output, str):                                         # If it's a string, parse it as JSON
            json_data = json.loads(llm_output) 
        elif isinstance(llm_output, dict):                                      # If it's already a dictionary, use it as is
            
            json_data = llm_output
        else:
            raise TypeError("llm_output must be a JSON-formatted string or dictionary.")

        text_file_name = file_path + file_name + '.json'
        with open(text_file_name, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

    except json.JSONDecodeError:                                                # If llm_output is not a valid JSON, save as plain text
        text_file_name = file_path + file_name + '.txt'
        with open(text_file_name, 'w') as text_file:
            text_file.write(llm_output)

    return text_file_name
