# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import os
import os.path as osp
import re
from io import BytesIO
import json
import time

import requests
import torch
from PIL import Image

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        print("downloading image from url", args.video_file)
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def get_prompt(question, options):
  prompt = f"<video>\n {question} Answer with just a single letter corresponding to the option."
  option_letters = 'ABCD'
  for i, option in enumerate(options):
    prompt += f"\n{option_letters[i]}. {option}"
  return prompt

def eval_model(args):
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base)

    results = []
    correct_pd = 0

    with open(args.anno_path, 'r') as file:
        data = json.load(file)
      
    # Attempt to load existing results to avoid re-processing
    results = []
    processed_indices = set()
    if osp.exists(args.output_path):
        try:
            with open(args.output_path, 'r') as json_file:
                existing_results = json.load(json_file)
                # Extract processed indices
                for r in existing_results:
                    processed_indices.add(r['index'])
                results = existing_results
                print(f"Loaded {len(results)} previously processed results.")
        except json.JSONDecodeError:
            print("Output file exists but is not valid JSON, starting fresh.")
            results = []
    else:
        results = []

    correct_pd = sum(1 for r in results if r['pd'] == r['gt_option'])

    # Count how many are already processed (skipped) and how many remain
    already_processed_count = len(processed_indices)
    total_count = len(data)
    to_process_count = total_count - already_processed_count

    # We'll measure time only from this run and for items processed this run
    start_time = time.time()
    processed_this_run = 0

    for i, item in enumerate(data):
        # Skip if this index is already processed
        if i in processed_indices:
            print(f"Skipping index {i}, already processed.")
            continue
        item_start_time = time.time()
        video_file = args.video_dir + item['video'][1:]

        from llava.mm_utils import opencv_extract_frames
        images, num_frames = opencv_extract_frames(video_file, args.num_video_frames)

        qs = get_prompt(item['question'], item['options'])
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if DEFAULT_IMAGE_TOKEN not in qs:
                print("no <image> tag found in input. Automatically append one at the beginning of text.")
                # do not repeatively append the prompt.
                if model.config.mm_use_im_start_end:
                    qs = (image_token_se + "\n") * len(images) + qs
                else:
                    qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs
        print("input: ", qs)

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args.conv_mode, args.conv_mode
                )
            )
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        print(images_tensor.shape)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[
                    images_tensor,
                ],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        print("Pred:", outputs)

        # Calculate latency for this item
        item_end_time = time.time()
        item_latency = item_end_time - item_start_time

        results.append({
            'pd': outputs, 
            'gt': item['gt_option'], 
            'needle_time': item['needle_time'],
            'length': item['length'],
            'latency': item_latency
        })

        if outputs == item['gt_option']:
          correct_pd += 1
        accuracy = (correct_pd / len(results)) * 100 if results else 0
        elapsed_time = time.time() - start_time
        avg_time_per_item = elapsed_time / (len(results) + 1)
        remaining_time = avg_time_per_item * (len(data) - len(results) - 1)

        print(f"Correct: {item['gt_option']}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Item Latency: {item_latency}")
        print(f"Elapsed time: {elapsed_time:.2f}s")
        print(f"Estimated remaining time: {remaining_time:.2f}s")

        with open(args.output_path, 'w') as json_file:
          json.dump(results, json_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA-2.7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--video-file", type=str, default=None)
    parser.add_argument("--num-video-frames", type=int, default=6)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--anno-path", type=str, required=True)
    parser.add_argument("--video-dir", type=str, required=True)
    args = parser.parse_args()

    eval_model(args)
