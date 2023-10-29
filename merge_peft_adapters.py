from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--peft_model_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()

def main():
    args = get_args()

    print(f"Loading base model: {args.base_model_name_or_path}")
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     args.base_model_name_or_path,
    #     return_dict=True,
    #     torch_dtype=torch.float16,
    # )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        # load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map={"":0}, # at least started writing
        # offload_folder="offload/",
        # device_map="auto",
    )

    print(f"Loading PEFT: {args.peft_model_path}")
    model = PeftModel.from_pretrained(base_model, args.peft_model_path, offload_folder="offload/")
    model.to(args.device)
    print(f"Running merge_and_unload")
    model = model.merge_and_unload() # https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py#L382

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    model.save_pretrained(f"{args.output_dir}")
    tokenizer.save_pretrained(f"{args.output_dir}")
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__" :
    main()
