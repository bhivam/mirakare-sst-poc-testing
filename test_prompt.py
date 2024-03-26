from openai import OpenAI
import csv
from datetime import datetime

FULL_PROMPT = "You will be given a note from a caregiver about their patient. You have to split this note up into a list of events. You must label each event. The categories are Heart Rate, Blood Pressure, Sleep, Oxygen Level, Breathing Rate, Temperature, Glucose, Weight, Entertainment, Exercise, Meals, Medication, Therapy, Toileting, and Other. Each event should be given in the following format: [Category | Time(am/pm format) | Date | Description]. At the beginning of the message you will get the date and time that the note is being taken. If any time information is missing from an event, use the date and time of the note and place an 'r' in front of it. Indicate new events with a new line."

ONLY_EVENTS_PROMPT = "You are going to take in a note from a caregiver about their patient describing one or more events that have occured. Split the note you recieve into a list of events."

EVENT_CATEGORIZATION = "You will take in a list of events from a care giver. Categorize each event Using the following categories: Heart Rate, Blood Pressure, Sleep, Oxygen Level, Breathing Rate, Temperature, Glucose, Weight, Entertainment, Exercise, Food/Drink, Medication, Therapy, Toileting, Other. The format of the output should be: Category | Associated Event. The output should be delimited using new lines"


def main():

    inputs = []

    with open("failure_data_set.csv", "r") as file:
        rows = csv.reader(file)
        for row in rows:
            if len(row) == 0:
                continue
            inputs.append(row[0])

    client = OpenAI()

    for input in inputs:
        note_transcription_with_date = (
            "Note Date: " + str(datetime.now()) + "\n" + input
        )

        print(note_transcription_with_date)

        completion1 = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": ONLY_EVENTS_PROMPT,
                },
                {
                    "role": "user",
                    "content": note_transcription_with_date,
                },
            ],
        )

        completion2 = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": EVENT_CATEGORIZATION
                    },
                    {
                        "role": "user",
                        "content": completion1.choices[0].message.content
                    }
                    ]
                )

        print(completion2.choices[0].message.content)

        print("===============")


if __name__ == "__main__":
    main()
