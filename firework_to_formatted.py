import sys
import json

if __name__ == "__main__":
    in_file = sys.argv[1]
    out_file = sys.argv[2]

    with open(in_file) as f:
        lines = f.readlines()

    outputs = []
    for line in lines:
        data = json.loads(line)
        content = data["response"]["choices"][0]["message"].get("content", "")
        reasoning = data["response"]["choices"][0]["message"].get("reasoning_content", "")
        output = f"{content} {reasoning}"
        outputs.append(output)

    with open(out_file, "w") as f:
        json.dump(outputs, f, indent=4)



    
    