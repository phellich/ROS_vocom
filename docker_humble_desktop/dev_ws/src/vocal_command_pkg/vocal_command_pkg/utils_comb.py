import numpy as np
import wave
import pyaudio
import json
import vosk
import whisper
import warnings
import json
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Optional, List
import re
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# import ollama

################################################
## SPEECH 2 TEXT                              ##
################################################

# PyAudio stream
def open_microphone_stream(device_name="pulse"):
    """Open an audio stream from the microphone using PyAudio."""

    # Advanced Linux Sound Architecture (ALSA) : API for sound card device drivers.
    # arecord -l in terminal to list audio devices
    
    p = pyaudio.PyAudio()                                                           # https://people.csail.mit.edu/hubert/pyaudio/docs/#pyaudio.PyAudio.Stream.__init__

    metadata = {
        "chunk": 16384,                                                             # 4096, # buffer size pour flux audio (+ grand = + de latence, + petit = + de CPU)
        "sample_format": pyaudio.paInt16,
        "nchannel": 1,                                                              # 1 pour mono, 2 pour stéréo (inutile car pas besoin de spacialisation)
        "framerate": 16000                                                          # fréquence d'échantillonnage (Hz) (16k pour speech recognition en général, 44.1k pour musique)
    }

    # Trouver l'index du périphérique correspondant à PulseAudio ou un autre nom
    # device_index = None
    # for i in range(p.get_device_count()):
    #     device_info = p.get_device_info_by_index(i)
    #     if device_name.lower() in device_info.get('name', '').lower():
    #         device_index = i
    #         break

    # if device_index is None:
    #     print(f"Erreur: Impossible de trouver le périphérique '{device_name}'.")
    #     p.terminate()
    #     return None, None, None

    try:
        stream = p.open(format=metadata["sample_format"],
                    channels=metadata["nchannel"],
                    rate=metadata["framerate"],
                    input=True,
                    frames_per_buffer=metadata["chunk"])                            # input_device_index=device_index,  # Utiliser l'index trouvé
    except Exception as e:
        print(f"Error lors de l'ouverture du flux audio: {e}")
        p.terminate()
        return None, None, None

    # print(f"Flux audio ouvert avec le périphérique: {device_name} (index: {device_index})")
    return p, stream, metadata

def close_microphone_stream(p, stream):
    """Close the audio stream and PyAudio instance."""
    stream.stop_stream()
    stream.close()
    p.terminate()

# Vosk and Whisper configuration and speech recognition functions
def whisper_config(model_name):
    """Initialise Whisper model."""
    warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
    whisper_model = whisper.load_model(model_name)
    return whisper_model

def vosk_config(model_path):
    """Initialise Vosk model."""
    vosk.SetLogLevel(-1)                                                            # -1 pour désactiver les logs
    vosk_model = vosk.Model(model_path)
    return vosk_model

# Fonctions Speech recognition
def recognize_speech_WHISPER(model, audio, audio_path=None):                        # Use prompt to recognize some specific words or enhance the recognition? https://platform.openai.com/docs/guides/speech-to-text
    """Processes audio data and checks for the wake word."""
    if audio_path:
        # Charger et préparer l'audio depuis un fichier si un chemin est fourni
        audio = whisper.load_audio(audio_path)
    else:
        # Convertir les données `bytes` en un tableau numpy de type float32 normalisé
        audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

    text = model.transcribe(audio)['text']

    return text

def wait_4_wake_word(stream, word_2_rec, metadata, recognizer):
    """Processes audio and checks for the wake word. Print Vosk recognized text in real time."""
    while True:
        # durée d'un chunk = (chunk_size/framerate) = 16384/16000 = 1.024s
        audio = stream.read(metadata["chunk"]*4, exception_on_overflow=True)

        _, wake_word_detected = recognize_speech_VOSK(recognizer, audio, word_2_rec)

        if wake_word_detected:
            return True

def recognize_speech_VOSK(recognizer, audio, word_2_rec):
    """Processes the audio data and applies speech recognition using Vosk."""

    recognizer.AcceptWaveform(audio)
    result = json.loads(recognizer.Result())
    text = result.get("text", "")
    word_2_rec_bool = any(w in text.lower() for w in word_2_rec)
    if text:
        print(f"Vosk: {text}")
    return text, word_2_rec_bool

def wait_4_sleep_word(stream, word_2_rec, metadata, recognizer, timeout=30):
    """
    Processes audio and checks for the sleep word. 
    Concatenate audio chunks and return full audio and text recognized by Vosk when sleep word is detected or timeout time is reached."""

    audio_chunks, text_chunks = [], []
    while True:
        audio = stream.read(metadata["chunk"]*4, exception_on_overflow=True)

        text, wake_word_detected = recognize_speech_VOSK(recognizer, audio, word_2_rec)

        audio_chunks.append(audio)
        text_chunks.append(text)

        if len(b''.join(audio_chunks)) / metadata["framerate"] > timeout*2:         # If audio chunks accumulated are more than timeout seconds, return the results
            print(f"Max audio duration of {timeout}s reached. Processing audio...")
            return b''.join(audio_chunks), ' '.join(text_chunks)

        if wake_word_detected:
            return b''.join(audio_chunks), ' '.join(text_chunks)

# Fonctions d'audio
def save_audio(audio, filepath, p, metadata):
    """Save audio to a WAV file."""
    wf = wave.open(filepath, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(metadata["sample_format"]))
    wf.setframerate(metadata["framerate"])
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

def read_text_file(file_path):
    '''Read the text content from a file.
    Returns a series of text lines.'''
    with open(file_path, "r") as file:
        text = file.read()
    return text

################################################
## TEXT 2 COMMAND                             ##
################################################

def prepare_llama_cpp_generation():
    """Prepare the system prompt and JSON schema for LlamaCPP generation."""

    class RoverCommand(BaseModel):
        command: Literal["displacement", "rotation", "drill", "cameras", "not_a_command"] = Field(description=("Type of command to execute."))
        direction: Optional[Literal["forward", "backward", "right", "left", "180_turn", "360_turn"]] = Field(None, description="Direction of movement or rotation, depending on the command")
        distance: Optional[int] = Field(None, description="Positive integer representing distance in meters, null otherwise")
        angle: Optional[int] = Field(None, description="Positive integer representing rotation angle in degrees, null otherwise")
        drill_additional_info: Optional[str] = Field(None, description="Optional string with details for drill instruction, null otherwise")
        cameras_state: Optional[Literal["turn_on", "turn_off"]] = Field(None, description="Change of state of the cameras, either 'turn_on' or 'turn_off'")

    class MissionPlan(BaseModel):
        execution_speed: Literal["fast", "slow", "default"] = Field(description="Execution speed for the mission")
        commands: List[RoverCommand] = Field(description="List of commands to be executed by the rover")

    json_schema = MissionPlan.model_json_schema()

    system_prompt = """You are assisting a rover's navigation and control system to interpret user instructions accurately. Your task is to identify specific commands (such as "move," "turn", "drill" or "cameras") and their details (such as distance, angle or execution speed) from the provided instructions.

Examples:

Input: "Spin rapidly on yourself 1/3 to the right"
Output:
{
  "commands": [
    {
      "command": "turn",
      "direction": "right",
      "angle": 120
    }
  ],
  "execution_speed": "fast"
}

Input: "Please turn gradualy left 90 degrees and then drive backwards a little, maybe one meter."
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
  ],
  "execution_speed": "slow"
}

Input: "Advance quickly 12 meters, stop, turn on the cameras and drill a hole."
Output:
{
  "commands": [
    {
      "command": "move",
      "direction": "forward",
      "distance": 12
    },
    {
      "command": "cameras",
      "cameras_state": "turn_on"
    },
    {
      "command": "drill",
      "drill_additional_info": "a hole"
  ],
  "execution_speed": "fast"
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

# Functions about pseudo JSON output
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
        print(f"Output saved as JSON to '{text_file_name}'.")

    except json.JSONDecodeError:                                                # If llm_output is not a valid JSON, save as plain text
        text_file_name = file_path + file_name + '.txt'
        with open(text_file_name, 'w') as text_file:
            text_file.write(llm_output)
        print(f"Output is not valid JSON. Saved as text to '{text_file_name}'.")