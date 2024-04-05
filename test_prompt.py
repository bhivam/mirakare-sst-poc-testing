from openai import OpenAI
import csv
from datetime import datetime

FULL_PROMPT = "You are going to take in a note from a caregiver about their patient describing one or more events that have occured. Split the note you recieve into a list of events. Categorize each event Using the following categories: Heart Rate, Blood Pressure, Sleep, Oxygen Level, Breathing Rate, Temperature, Glucose, Weight, Entertainment, Exercise, Food/Drink, Medication, Therapy, Toileting, Other. The format of the output should be: Category | Time (am/pm) | Associated Event. The output should be delimited using new lines. If the Time cannot be found use 'n/a'."

ONLY_EVENTS_PROMPT = "You are going to take in a note from a caregiver about their patient describing one or more events that have occured. Split the note you recieve into a list of events."

EVENT_CATEGORIZATION = "You will take in a list of events from a care giver. Categorize each event Using the following categories: Heart Rate, Blood Pressure, Sleep, Oxygen Level, Breathing Rate, Temperature, Glucose, Weight, Entertainment, Exercise, Food/Drink, Medication, Therapy, Toileting, Other. The format of the output should be: Category | Time (am/pm) | Associated Event. The output should be delimited using new lines. If the Time cannot be found use 'n/a'."

"""
NOTES:
note 1:
- "Mira's diaper was changed" categorized as "Other". define toiletting.
- "Mira was brought downstairs for breakfast" categorized as "Food/Drink". explain that "Food/Drink" category has to do with consumption of food and drink, not just any event that related to food or drink. 

note 2:
- Should medication be lobbed together like this or split apart?
- Same comment on toiletting

note 3:
- "therapy | albuterol treatment" is albuterol treatment therapy or medication?
- time for 10 milliliters of milk of magnesia followed by 6 ounces of water seems arbitrarily picked. 

note 4:
- "Mira came home" this was categorized as entertainment.
- "Mira went to school" as entertainment
- This should be in "other" or category such as transportation

note 5: 

"""


def main():

    inputs = []
    desired_outputs = []

    with open("data_set.csv", "r") as file:
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

        completion0 = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:personal::9AiKHtIp",
            messages=[
                {"role": "system", "content": FULL_PROMPT},
                {"role": "user", "content": note_transcription_with_date}
            ],
        )
        print("FINE TUNED GPT 3.5")
        print(completion0.choices[0].message.content)

        completion1 = client.chat.completions.create(
            model="gpt-4-turbo-preview",
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
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": EVENT_CATEGORIZATION},
                {
                    "role": "user",
                    "content": str(completion1.choices[0].message.content),
                },
            ],
        )

        desired_outputs.append(completion2.choices[0].message.content)
        print("GPT 4.0")
        print(completion2.choices[0].message.content)

        print("===============")

    with open("data_set.csv", "w") as file:
        w = csv.writer(file)
        for input, output in zip(inputs, desired_outputs):
            w.writerow([input, output])


if __name__ == "__main__":
    main()
