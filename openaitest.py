import openai
import json
import sounddevice as sd
import numpy as np
import wave
import io
import socket
from time import sleep
import os
# OpenAI API key
client = openai.OpenAI(os.getenv("OPENAI_API_KEY"))

# Wake word
WAKE_WORD = "heymedmax"

# Initialize conversation history
conversation_history = [
    {
        "role": "system",
        "content": """
        You are a medical assistant specializing in small injury treatment. 
          Your responses should always be structured as follows:
            **A JSON array** with the appropriate EV3 motor commands.
            **Clear first-aid instructions** in human-friendly language.
        Write all of the JSON first, followed by two newlines, and then the instructions. Do not include text with the JSON, and don't include a heading stating it is the JSON.

        **Robot Instructions:**
        - To deploy a cotton ball: {"port": "C", "speed": 50, "degrees": -200}
        - To move the belt to the antiseptic dispenser: {"port": "D", "speed": 50, "degrees": 415}
        - To move the belt in order to put the completed cotton ball into my hand: {"port": "D", "speed": 50, "degrees": 515}
        - To squeeze the antiseptic bottle: {"port": "A", "speed": 25, "degrees": -180} (must be squeezed and unsqueezed twice per cotton ball)
        - To reset the belt: {"port":"D", "speed":50, "degrees":-930}

        Once initialized, you will receive injury scenarios and respond accordingly.
        """
    }
]

def send_message_to_ev3(messages):
    # Define the IP address and port of your EV3
    # These values will only work if the EV3 is configured to accept TCP/IP connections (e.g., Wi-Fi setup)
    ev3_ip = "169.254.77.240"  # Replace with your EV3's IP address
    port = 5555  # Port for TCP/IP communication (set in EV3 configuration)

    # Message to send
    message = "Hello World"

    sock = None  # Initialize the socket variable
    try:
        # Create a TCP socket
        print("Creating socket...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to the EV3
        print("Connecting to EV3 at {}:{}".format(ev3_ip, port))
        sock.connect((ev3_ip, port))
        print("Connected!")
        messageList=messages
        for message in messageList:
            print("Sending message: {}".format(message))
            sock.sendall(message.encode())
            print("Message sent successfully!")

    except OSError as e:
        print("Socket error: {}".format(e))

    finally:
        # Close the socket safely
        if sock:
            print("Closing connection...")
            sock.close()

def record_audio(duration=1, fs=16000):
    """Record audio for a specified duration and return the audio data"""
    print("Listening...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    return audio_data

def transcribe_audio(audio_data):
    """Send the audio data to Whisper for transcription"""
    # Save audio data to a temporary WAV file
    audio_file=io.BytesIO()
    with wave.open(audio_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(16000)
        wf.writeframes(audio_data.tobytes())
    
    audio_file.seek(0)
    # Send audio file to OpenAI Whisper API
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file = ("audio.wav", audio_file, "audio/wav"),
        response_format="text"
    )
    return transcript

def get_medical_response(scenario):
    """Ask OpenAI for a response if a medical scenario is detected"""
    user_message = {"role": "user", "content": scenario}
    conversation_history.append(user_message)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=conversation_history,
        max_tokens=500
    )

    response_text = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": response_text})

    try:
        json_part, treatment_steps = response_text.split("\n\n", 1)
        json_array = json.loads(json_part)
        return json_array, treatment_steps.strip()
    except (ValueError, json.JSONDecodeError):
        return [], response_text  # Return raw response if JSON parsing fails

def listen_for_wake_word():
    """Listen for the wake word 'Hey, medmax' and then process medical scenarios"""
    while True:
        audio_data = record_audio(duration=3)
        transcription = transcribe_audio(audio_data)
        print(f"Detected Speech: {transcription}")

        # Check for the wake word
        if WAKE_WORD.lower() in transcription.lower().replace(' ', ''):
            print("Wake word detected! Listening for a medical scenario...")
            
            audio_data = record_audio(duration=5)
            medical_scenario = transcribe_audio(audio_data)
            print(f"Medical Scenario Detected: {medical_scenario}")
            
            actions, treatment = get_medical_response(medical_scenario)

            print("\nJSON Actions:", json.dumps(actions, indent=2))
            print("\nTreatment Instructions:", treatment)
            formattedJson = []
            for j in actions:
                y = "{"
                for i in j:
                    if type(j[i]) == str:
                        y += f"\"{i}\":\"{j[i]}\", "
                    else:
                        y += f"\"{i}\":{j[i]}, "
                formattedJson.append(y.rstrip(", ") + "}")
            send_message_to_ev3(formattedJson)
        else:
            print("No wake word detected. Continuing to listen...")

# Start listening for the wake word and scenarios
listen_for_wake_word()
