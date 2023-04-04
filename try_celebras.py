# Program: VTSTech-GPT.py 2023-04-03 9:23:22 PM
# Description: Python script that generates text with Cerebras GPT pretrained and Corianas finetuned models
# Author: Written by Veritas//VTSTech (veritas@vts-tech.org)
# GitHub: https://github.com/Veritas83
# Homepage: www.VTS-Tech.org
# Dependencies: transformers, colorama, Flask, torch
# pip install transformers colorama Flask torch
# Models are stored at C:\Users\%username%\.cache\huggingface\hub
import argparse
import time
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from colorama import Fore, Style, init

global start_time, end_time, build, model_size, prompt_text
init(autoreset=True)
build = "v0.2-r09"
tok = random.seed()
eos_token_id = tok
parser = argparse.ArgumentParser(description="Generate text with Cerebras GPT models")
parser.add_argument(
    "-m",
    "--model",
    choices=["111m", "256m", "590m", "1.3b", "2.7b", "6.7b", "13b"],
    help="Choose the model size to use (default: 111m)",
    type=str.lower,
    default="111m"
)
parser.add_argument(
    "-p",
    "--prompt",
    type=str,
    default="AI is",
    help='Text prompt to generate from (default: "AI is")',
)
parser.add_argument("-s", "--size", type=int, default=None)
parser.add_argument("-l", "--length", type=int, default=None)
parser.add_argument("-tk", "--topk", type=float, default=None)
parser.add_argument("-tp", "--topp", type=float, default=None)
parser.add_argument("-ty", "--typp", type=float, default=None)
parser.add_argument("-tm", "--temp", type=float, default=None)
args = parser.parse_args()
if args.model:
    model_size = args.model
if args.prompt:
    prompt_text = args.prompt
if args.length is not None:
    max_length = int(args.length)
else:
    max_length = args.length
top_p = args.topp
top_k = args.topk
typ_p = args.typp
temp = args.temp


def get_model(model_size):
    if model_size == "111m":
        return "cerebras/Cerebras-GPT-111M"
    elif model_size == "256m":
        return "cerebras/Cerebras-GPT-256M"
    elif model_size == "590m":
        return "cerebras/Cerebras-GPT-590M"
    elif model_size == "1.3b":
        return "cerebras/Cerebras-GPT-1.3B"
    elif model_size == "2.7b":
        return "cerebras/Cerebras-GPT-2.7B"
    elif model_size == "6.7b":
        return "cerebras/Cerebras-GPT-6.7B"
    elif model_size == "13b":
        return "cerebras/Cerebras-GPT-13B"
    return "cerebras/Cerebras-GPT-111M"


model_name = get_model(model_size)


def banner():
    print(Style.BRIGHT + f"VTSTech-GPT {build} - www: VTS-Tech.org git: Veritas83")
    print("Using Model : " + Fore.RED + f"{model_name}")
    print("Using Prompt: " + Fore.YELLOW + f"{prompt_text}")
    print(
        "Using Params: "
        + Fore.YELLOW
        + f"max_new_tokens:{max_length} do_sample:True use_cache:True no_repeat_ngram_size:2 top_k:{top_k} top_p:{top_p} typical_p:{typ_p} temp:{temp}"
    )


def CerbGPT(prompt_text):
    global start_time, end_time, build
    temp = None
    top_k = None
    top_p = None
    start_time = time.time()
    model_name = get_model(model_size)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        print("!! Tokenizer Error")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except:
        print("!! Model Error")
    opts = {}
    if temp is not None:
        opts["temperature"] = temp
    if top_k is not None:
        opts["top_k"] = top_k
    if top_p is not None:
        opts["top_p"] = top_p
    if typ_p is not None:
        opts["typical_p"] = typ_p
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    generated_text = pipe(
        prompt_text,
        max_new_tokens=max_length,
        do_sample=True,
        use_cache=True,
        no_repeat_ngram_size=2,
        **opts,
    )[0]
    end_time = time.time()
    return generated_text["generated_text"]


if __name__ == "__main__":
    banner()
    print(CerbGPT(prompt_text))
