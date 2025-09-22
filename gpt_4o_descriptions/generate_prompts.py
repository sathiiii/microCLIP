import os
import re
import openai
from openai import OpenAI
import json
from tqdm import tqdm
from argparse import ArgumentParser

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def extract_list_items(text):
    items = re.findall(r"\d+[\).\s-]+\s*(.+?)(?=\n\d+[\).\s-]+|\Z)", text.strip(), re.DOTALL)
    return [item.strip().rstrip(".") + "." for item in items if item]

prompt_templates = { 
    "cars": [
        "Describe what {} {}, a type of car, looks like.",
        "What are the primary characteristics of {} {}?",
        "Description of the exterior of {} {}.",
        "What are the identifying characteristics of {} {}, a type of car?",
        "Describe an image from the internet of {} {}."
    ],
    "pets": [
        "Describe what a pet {} looks like.",
        "Visually describe {} '{}', a type of pet."
    ],
    "dtd": [
        "What does a “{}” surface look like?",
        "What does a “{}” texture look like? ",
        "What does a “{}” pattern look like?"
    ],
    "resisc45": [
        "Describe a satellite photo of {} {}",
        "Describe {} {} as it would appear in an aerial image",
        "How can you identify {} {} in an aerial photo?",
        "Describe the satellite photo of {} {}"
    ],
    "flowers": [
        "Describe how to identify {} {}, a type of flower",
        "What does {} {} flower look like?"
    ]
}

def get_article(word):
    return "an" if word[0].lower() in "aeiou" else "a"

def fill_prompt_templates(template_list, category_name):
    filled = []
    article = get_article(category_name)
    
    for template in template_list:
        placeholder_count = template.count("{}")
        
        if placeholder_count == 1:
            filled.append(template.format(category_name))
        elif placeholder_count == 2:
            filled.append(template.format(article, category_name))
            
    return filled

def parse_args():
    parser = ArgumentParser(description="Generate prompts for categories.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset.")
    parser.add_argument("--classes", type=str, default="configs/classes.json", help="Path to the classes file.")
    args = parser.parse_args()
    if not os.path.exists(args.classes):
        raise FileNotFoundError(f"Classes file {args.classes} does not exist.")
    with open(args.classes, 'r') as f:
        args.classes = json.load(f)
    if args.dataset not in args.classes:
        raise ValueError(f"Dataset {args.dataset} not found in classes file.")
    return args

def main(args):
    openai.api_key = OPENAI_API_KEY
    output_dir = os.path.dirname(os.path.abspath(__file__))
    # json_name = os.path.join(output_dir, f"{args.dataset}.json")

    # all_responses = {}
    dataset = args.dataset
    classes = args.classes[dataset]

    tasks = []
    
    for category in classes:
        prompts = fill_prompt_templates(prompt_templates[dataset], category)

        for index, curr_prompt in enumerate(prompts):
            task = {
                "custom_id": f"{category}_{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "temperature": 0.1,
                    "response_format": { 
                        "type": "json_object"
                    },
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant."
                        },
                        {
                            "role": "user",
                            "content": f"Please provide 10 sentences in a numbered list format that answer the following prompt in a detailed and descriptive manner:\n\n{curr_prompt}"
                        }
                    ],
                }
            }
            tasks.append(task)
            
    file_name = os.path.join(output_dir, f"{dataset}_batch_prompts.jsonl")
    with open(file_name, 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')
            
    # client = OpenAI()
            
    # batch_file = client.files.create(
    #     file=open(file_name, "rb"),
    #     purpose="batch"
    # )
    # batch_job = client.batches.create(
    #     input_file_id=batch_file.id,
    #     endpoint="/v1/chat/completions",
    #     completion_window="24h"
    # )
            
        #     response = openai.ChatCompletion.create(
        #         model="gpt-4o",
        #         messages=[
        #             {"role": "system", "content": "You are a helpful assistant."},
        #             {"role": "user", "content": prompt}
        #         ],
        #         temperature=0.1,
        #         max_tokens=512,
        #     )
        #     result = response.choices[0].message.content.strip()
        #     all_result += extract_list_items(result)

        #     # Update prompt token count live
        #     if "usage" in response:
        #         total_prompt_tokens += response["usage"]["prompt_tokens"]
        #         progress.set_postfix({"prompt_tokens": total_prompt_tokens})

        # all_responses[category] = all_result

    # with open(json_name, 'w') as f:
    #     json.dump(all_responses, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)