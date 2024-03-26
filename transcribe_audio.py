from openai import OpenAI
import os
import csv


def transcribe_audio(path: str, oai_client: OpenAI):
    audio_file = open(path, "rb")

    transcription = oai_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    audio_file.close()

    return transcription.text


def main():
    client = OpenAI()

    test_files = os.listdir("test_files")

    csv_file = open("failure_data_set.csv", "a")
    writer = csv.writer(csv_file, delimiter=",")

    for test_file in test_files:
        print("processing", test_file)
        test_file_path = "test_files/" + test_file
        text = transcribe_audio(test_file_path, client)
        print(text)
        writer.writerow([text])


if __name__ == "__main__":
    main()
