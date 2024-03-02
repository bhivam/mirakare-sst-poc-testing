"""
Notes:

LLM for info bucketing:
- GPT 3.5 vs 4: Will the performance difference matter for our use case
  - Will have to build various test cases
  - Ask Vik if he can start making recordings
- Engineering prompt correctly. Will require trial and error.

Speech-to-Text for dictation:
- AssemblyAI vs Whisper vs Deepgram
  - We should collect many voice recordings and run some correctness and speed tests
- If we use whisper, there is also the optino of using whisper on machine models
  - This would reduce costs but require testing to see how much accuracy is lost using smaller models.

"""

from openai import OpenAI
import os
import time
import sys
import pyaudio
import wave
from pydub import AudioSegment

def record_audio():
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 44100  # Record at 44100 samples per second
    seconds = 20
    filename = "output.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    for _ in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    wf = AudioSegment.from_wav('output.wav')
    wf.export('output.mp3', format='mp3')


def main():

    if len(sys.argv) > 2:
        print("Usage: python3 audio_to_events.py <mp3 file>")
        sys.exit(0)

    audio_file_path = ""
    if len(sys.argv) == 2:
        audio_file_path = sys.argv[1]
        if audio_file_path[-4:] != ".mp3":
            print("argument given must be path to mp3 file")
            sys.exit(0)
    else:
        record_audio()
        audio_file_path = "output.mp3"

    audio_file = open(audio_file_path, "rb")
    file_creation_datetime = str(time.ctime(os.path.getctime(audio_file_path)))

    client = OpenAI()

    note_transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )

    note_transcription_with_date = "Note Date: " + file_creation_datetime + "\n" + note_transcription.text

    print(note_transcription_with_date)

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be given a note from a caregiver about a patient who they are taking care of. You have to split this note up into information about the following categories and return the categorized information to the user. The categories are Heart Rate, Blood Pressure, Sleep, Oxygen Level, Breathing Rate, Temperature, Glucose, Weight, Entertainment, Exercise, Meals, Medication, Therapy, Toileting. Any information that does not fit into these categories should be summarized into a note and returned to the user separately. Each event should be given in the following format: [Category | Time(am/pm format) | Date | Description]. At the beginning of the message you will get the date and time that the note is being taken. If any time information is missing from an event, use the date and time of the note and place an 'r' in front of it. The extra note should have category \"Other Note\"",
            },
            {
                "role": "user",
                "content": note_transcription_with_date,
            },
        ],
    )

    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
