import json
import uuid
import sys

def generate_batch_file(in_file, out_file):
    with open(in_file) as f:
        prompts = json.load(f)

    with open(out_file, "w") as f:
        for prompt in prompts:
            messages = []
            for p in prompt:
                messages.append({"role": "assistant", "content": ""})
                messages.append({"role": "user", "content": p})
            request = {
                "custom_id": str(uuid.uuid4()),
                "body": {
                    "messages": messages[1:],
                    "max_tokens": 50,
                    "temperature": 0.7,
                },
            }
            f.write(json.dumps(request) + "\n")

if __name__ == "__main__":
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    generate_batch_file(in_file, out_file)