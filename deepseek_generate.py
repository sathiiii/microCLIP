import os
import re
import json
import torch
import argparse
from tqdm import tqdm

from deepseek_vl2.utils.io import load_pil_images
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from transformers import AutoModelForCausalLM

vowel_list = ['A', 'E', 'I', 'O', 'U']

PROMPT_TEMPLATES = [
    "What does {article_class} {class_name} in an image look like? Describe the visual features in at least 30 words, in sentences.",
    "Describe the visual appearance of {article_class} {class_name} typically found in an image. Use full sentences, 30 words or more.",
    "In an image, how can one visually identify {article_class} {class_name}? Provide a detailed description in full sentences, totaling at least 30 words.",
    "Explain how {article_class} {class_name} is depicted in visuals. Focus on textures, shapes, and colors. Write 30+ words.",
    "How would {article_class} {class_name} look in an image? Provide a thorough and detailed description in sentences."
]
ESAT_PROMPT_TEMPLATES = [
    "What does {article_class} {class_name} in an aerial image look like? Describe the visual features in at least 30 words, in sentences.",
    "Describe the visual appearance of {article_class} {class_name} typically found in a satellite image. Use full sentences, 30 words or more.",
    "In a satellite image, how can one visually identify {article_class} {class_name}? Provide a detailed description in full sentences, totaling at least 30 words.",
    "Explain how {article_class} {class_name} is depicted in visuals. Focus on textures, shapes, and colors. Write 30+ words.",
    "How would {article_class} {class_name} look in an aerial image? Provide a thorough and detailed description in sentences."
]

def get_article(word):
    return "an" if word[0].upper() in vowel_list else "a"

def clean_description(text):
    artifacts = [
        r"<\|endoftext\|>",
        r"<\\uff5cend\\u2581of\\u2581sentence\\uff5c>",
        r"<\|.*?\|>",
        r"[^\x00-\x7F]+",
        r"<endofsentence>",
    ]
    for pattern in artifacts:
        text = re.sub(pattern, '', text)

    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip()

    if not text.endswith('.'):
        text += '.'

    return text

def generate_descriptions(prompts, class_names, output_path, samples_per_prompt=2):
    final_output = {}
    for class_name in tqdm(class_names, desc="Generating Descriptions"):
        class_descriptions = []
        for prompt_template in prompts:
            for _ in range(samples_per_prompt):
                prompt = prompt_template.format(
                    article_class=get_article(class_name),
                    class_name=class_name
                )

                conversation = [
                    {"role": "<|User|>", "content": prompt, "images": []},
                    {"role": "<|Assistant|>", "content": ""}
                ]

                pil_images = load_pil_images(conversation)
                prepare_inputs = vl_chat_processor(
                    conversations=conversation,
                    images=pil_images,
                    force_batchify=True,
                    system_prompt=""
                ).to(vl_gpt.device)

                inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
                inputs_embeds, past_key_values = vl_gpt.incremental_prefilling(
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    chunk_size=512
                )

                outputs = vl_gpt.generate(
                    inputs_embeds=inputs_embeds,
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    past_key_values=past_key_values,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=True,
                    use_cache=True,
                    temperature=0.7
                )

                decoded = tokenizer.decode(
                    outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(),
                    skip_special_tokens=False
                )
                sentences = clean_description(decoded).split('.')
                sentences = [s.strip() + '.' for s in sentences if len(s.strip()) > 0]
                class_descriptions.extend(sentences)

        final_output[class_name] = class_descriptions

    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=2)

def get_class_names(dataset):
    if dataset.lower() == "office_home":
        image_root = os.path.join('data', dataset, "images")
        domain_folders = [d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))]
        class_path = os.path.join(image_root, domain_folders[0])
        class_folders = [c for c in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, c))]
        def clean_name(name):
            return name.replace('_', ' ').replace('-', ' ').strip().title()
        return [clean_name(cls) for cls in class_folders]
    elif dataset.lower() == "aircrafts":
        with open('configs/classes.json', 'rb') as f:
            class_names = json.load(f)['fgvc']
        return [name + ' airplane' for name in class_names]
    elif dataset.lower() == "eurosat":
        with open('configs/classes.json', 'rb') as f:
            class_names = json.load(f)['eurosat']
        return class_names
    else:
        raise NotImplementedError(f"Class name extraction not implemented for dataset: {dataset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset (e.g., office_home)")

    args = parser.parse_args()

    model_path = "deepseek-ai/deepseek-vl2-small"
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    class_names = get_class_names(args.dataset)
    output_path = os.path.join('data', args.dataset, "class_descriptions.json")

    if args.dataset.lower() == "eurosat":
        generate_descriptions(ESAT_PROMPT_TEMPLATES, class_names, output_path)
    else:
        generate_descriptions(PROMPT_TEMPLATES, class_names, output_path)
