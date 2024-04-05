from openai import OpenAI 

def main():
    client = OpenAI()
    
    # file = open("ft_data.json", "rb")
    # client.files.create(file=file, purpose="fine-tune")

    # file_id = client.files.list(purpose="fine-tune").data[0].id
    # client.fine_tuning.jobs.create(training_file=file_id, model="gpt-3.5-turbo")

    print(client.fine_tuning.jobs.list().data[-1])

if __name__ == '__main__':
    main()
