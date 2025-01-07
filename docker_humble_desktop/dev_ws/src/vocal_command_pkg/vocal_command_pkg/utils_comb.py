import numpy as np
import wave
import pyaudio
import json
import whisper
import warnings
import json
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from collections import deque
import time
import pvporcupine
import struct

################################################
## SPEECH 2 TEXT                              ##
################################################

# Whisper configuration and speech recognition functions
def whisper_config(model_name):
    """Initialise Whisper model."""
    warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
    whisper_model = whisper.load_model(model_name)
    return whisper_model

# Fonctions Speech recognition
def recognize_speech_WHISPER(model, audio, audio_path=None, word_2_rec=None, whisper_model="Whisper Base.en", whisper_prompt=None):                        # Use prompt to recognize some specific words or enhance the recognition? https://platform.openai.com/docs/guides/speech-to-text
    """Processes audio data and checks for the wake word."""
    if audio_path:
        # Charger et préparer l'audio depuis un fichier si un chemin est fourni
        audio = whisper.load_audio(audio_path)
    else:
        # Convertir les données `bytes` en un tableau numpy de type float32 normalisé
        audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

    # https://github.com/openai/whisper/blob/main/whisper/transcribe.py

    # start = time.time()
    text = model.transcribe(audio)['text']
    # print(f"Param 0: {time.time()-start:.2f}s")

    # start = time.time()
    # text = model.transcribe(audio,
    #                         language="en",
    #                         task="transcribe"
    #                         )['text']
    # print(f"Param 1: {time.time()-start:.2f}s")

    # start = time.time()
    # text = model.transcribe(audio,
    #                         fp16=False
    #                         )['text']
    # print(f"Param 2: {time.time()-start:.2f}s")

    # start = time.time()
    # text = model.transcribe(audio,
    #                         temperature=0
    #                         )['text']
    # print(f"Param 3: {time.time()-start:.2f}s")

    # start = time.time()
    # text = model.transcribe(audio,
    #                         beam_size=1, 
    #                         best_of=1
    #                         )['text']
    # print(f"Param 4: {time.time()-start:.2f}s")

    # start = time.time()
    # text = model.transcribe(audio,
    #                         logprob_threshold=-3.0, # default=-1.0
    #                         no_speech_threshold=0.3 # default=0.6
    #                         )['text']
    # print(f"Param 5: {time.time()-start:.2f}s")

    # start = time.time()
    # text = model.transcribe(audio,
    #                         initial_prompt=whisper_prompt
    #                         )['text']
    # print(f"Param 6: {time.time()-start:.2f}s")

    word_2_rec_bool = False
    if word_2_rec:
        word_2_rec_bool = any(w in text.lower() for w in word_2_rec)
        if text:
            print(f"{whisper_model}: {text.lower()}")
    return text, word_2_rec_bool

def wait_4_wake_word_TEST(porcupine_key, model_path):
    """Processes audio and checks for the wake word. Print Vosk recognized text in real time."""
    # https://picovoice.ai/docs/quick-start/porcupine-python/
    # https://medium.com/@rohitkuyadav2003/building-a-hotword-detection-with-porcupine-and-python-f95de3b8278d 
    
    porcupine=None
    p=None
    audio_stream=None
    try:
        porcupine = pvporcupine.create(
            access_key=porcupine_key,
            keyword_paths=[model_path]
        )
        p=pyaudio.PyAudio()
        audio_stream=p.open(rate=porcupine.sample_rate,channels=1,format=pyaudio.paInt16,input=True,frames_per_buffer=porcupine.frame_length)
        while True:
            keyword=audio_stream.read(porcupine.frame_length)
            keyword=struct.unpack_from("h"*porcupine.frame_length,keyword)
            keyword_index=porcupine.process(keyword)
            if keyword_index>=0:
                print("hotword detected")
                break

    finally:
        if porcupine is not None:
            porcupine.delete()
        if audio_stream is not None:
            audio_stream.close()
        if p is not None:
            p.terminate()
        print("keyword_index: ", keyword_index)
        return True

def wait_4_sleep_word_TEST(porcupine_key, model_path, timeout=30):
    """
    Processes audio and checks for the sleep word. 
    Concatenate audio chunks and return full audio and text recognized by Vosk when sleep word is detected or timeout time is reached."""
        
    porcupine=None
    p=None
    audio_stream=None
    try:
        porcupine = pvporcupine.create(
            access_key=porcupine_key,
            keyword_paths=[model_path]
        )
        p=pyaudio.PyAudio()
        audio_stream=p.open(rate=porcupine.sample_rate,channels=1,format=pyaudio.paInt16,input=True,frames_per_buffer=porcupine.frame_length)
        sampwidth = p.get_sample_size(pyaudio.paInt16)
        framerate = porcupine.sample_rate
        print(f"Sample width: {sampwidth}, Frame rate: {framerate}")
        audio_chunks = []
        while True:
            keyword=audio_stream.read(porcupine.frame_length)
            audio_chunks.append(keyword)
            keyword=struct.unpack_from("h"*porcupine.frame_length,keyword)
            keyword_index=porcupine.process(keyword)
            if keyword_index>=0:
                print("hotword detected")
                break
            if len(b''.join(list(audio_chunks))) / porcupine.sample_rate > timeout*2:         # If audio chunks accumulated are more than timeout seconds, return the results
                print(f"Max audio duration of {timeout}s reached. Processing audio...")
                break

    finally:
        if porcupine is not None:
            porcupine.delete()
        if audio_stream is not None:
            audio_stream.close()
        if p is not None:
            p.terminate()
        print("keyword_index: ", keyword_index)
        return b"".join(audio_chunks), sampwidth, framerate
        
def wait_4_wake_word(stream, word_2_rec, metadata, model):
    """Processes audio and checks for the wake word. Print Vosk recognized text in real time."""

    # q= deque()
    while True:

        # if len(q) > 5:
        #     q.popleft() 
  
        audio = stream.read(metadata["chunk"], exception_on_overflow=True) 

        # q.append(audio)       

        # # start = time.time()
        _, wake_word_detected = recognize_speech_WHISPER(model, 
                                                        #  audio, 
                                                         b"".join(q), # audio,
                                                         word_2_rec=word_2_rec, 
                                                         whisper_model="Whisper Tiny.en",
                                                         whisper_prompt=f"Listen for wake word {word_2_rec[0]}.")
        # # print(f"Time for 5 chunk: {time.time()-start:.2f}s")

        if wake_word_detected:
            # q.clear()
            return True

def wait_4_sleep_word(stream, word_2_rec, metadata, model, timeout=30):
    """
    Processes audio and checks for the sleep word. 
    Concatenate audio chunks and return full audio and text recognized by Vosk when sleep word is detected or timeout time is reached."""

    audio_chunks = []
    while True:
        audio = stream.read(metadata["chunk"], exception_on_overflow=True)
        audio_chunks.append(audio)

        start = time.time()
        _, wake_word_detected = recognize_speech_WHISPER(model, 
                                                         b"".join(audio_chunks[-5:]), 
                                                         word_2_rec=word_2_rec,
                                                         whisper_model="Whisper Tiny.en",
                                                         whisper_prompt=f"Start listening for end of command word {word_2_rec[0]}.")
        print(f"Time for 5 chunk: {time.time()-start:.2f}s")

        if len(b''.join(list(audio_chunks))) / metadata["framerate"] > timeout*2:         # If audio chunks accumulated are more than timeout seconds, return the results
            print(f"Max audio duration of {timeout}s reached. Processing audio...")
            return b"".join(audio_chunks)

        if wake_word_detected:
            return b"".join(audio_chunks)

# Fonctions d'audio
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
        command: Literal["move", "turn", "drill", "cameras", "not_a_command"] = Field(description=("Type of command to execute."))
        direction: Optional[Literal["forward", "backward", "right", "left", "180_turn", "360_turn"]] = Field(None, description="Direction of movement or rotation, depending on the command")
        execution_speed: Optional[Literal["fast", "slow", "default"]] = Field(None, description="Execution speed for the mission")
        distance: Optional[float] = Field(None, description="Distance in meters, null otherwise")
        angle: Optional[float] = Field(None, description="Rotation angle in degrees, null otherwise")
        camera_toggle: Optional[Literal["turn_on", "turn_off"]] = Field(None, description="Indicates whether to turn the cameras on or off, null otherwise")
        drill_rotation_speed: Optional[int] = Field(None, description="Optional drill rotation speed, null otherwise") # in what unit??
        drill_distance_ratio: Optional[float] = Field(None, description="Optional drill distance ratio, null otherwise") # ratio of what??

    class MissionPlan(BaseModel):
        commands: List[RoverCommand] = Field(description="List of commands to be executed by the rover")

    json_schema = MissionPlan.model_json_schema()

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