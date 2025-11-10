import sys
from datetime import datetime
from dotenv import load_dotenv
import json
import os
from openai import OpenAI
import uuid

JOB_ID_LOG_FILE = "job_id.log"

def generate_batch_file(in_file, out_file):
    with open(in_file) as f:
        prompts = json.load(f)

    with open(out_file, 'w') as f:
        for prompt in prompts:
            messages = []
            for p in prompt:
                messages.append({"role": "assistant", "content": ""})
                messages.append({"role": "user", "content": p})
            request = {
                "custom_id": str(uuid.uuid4()),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini-2024-07-18",
                    "messages": messages[1:],
                    "max_tokens": 50,
                    "temperature": 0.7
                }
            }
            f.write(json.dumps(request) + "\n")

def submit_batch_file(file_path, nickname):
    load_dotenv()
    OpenAI.api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI()

    batch_input_file = client.files.create(
        file=open(file_path, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    job_info = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "story generation batch job"
        }
    )

    job_id = job_info.id

    if os.path.exists(JOB_ID_LOG_FILE):
        with open(JOB_ID_LOG_FILE, "a") as f:
            f.write(f"Nickname: {nickname}, Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Job ID: {job_id}\n")
    else:
        with open(JOB_ID_LOG_FILE, "w") as f:
            f.write(f"Nickname: {nickname}, Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Job ID: {job_id}\n")

    print(f"Batch job submitted with ID: {job_id}")
    os.remove(file_path)

def get_status(code):
    load_dotenv()
    OpenAI.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI()
    print("Job Status:", client.batches.retrieve(code).status)

def get_result(code, out_file):
    load_dotenv()
    OpenAI.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI()

    file_response = client.files.content(client.batches.retrieve(code).output_file_id)

    response_text = []
    for line in file_response.text.split('\n'):
        if line:
            response = json.loads(line)
            response_text.append(response["response"]["body"]["choices"][0]["message"]["content"])

    with open(out_file, "w") as f:
        json.dump(response_text, f, indent=4)

if __name__ == "__main__":
    operation = sys.argv[1]

    if operation == 'submit':
        in_file = sys.argv[2]
        nickname = sys.argv[3]
        tmp_file = f'tmp_batch_{datetime.now().strftime("%Y%m%d%H%M%S")}.jsonl'
        generate_batch_file(in_file, tmp_file)
        submit_batch_file(tmp_file, nickname)
    elif operation == 'results':
        code = sys.argv[2]
        out_file = sys.argv[3]
        get_result(code, out_file)
    elif operation == 'status':
        code = sys.argv[2]
        get_status(code)
    else:
        print("Invalid operation. Must be 'submit', 'results', or 'status.")