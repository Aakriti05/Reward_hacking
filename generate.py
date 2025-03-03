import torch
import json
import argparse
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name="Qwen/Qwen2.5-Math-7B"):
    """Loads the specified LLM model and tokenizer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    return model, tokenizer, device


def generate_response(model, tokenizer, question, diversity_param, device):
    """Generates a diverse Chain-of-Thought (CoT) response using nucleus sampling."""
    messages = [
        {"role": "system", "content": "Please provide a detailed step-by-step solution."},
        {"role": "user", "content": question}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    eos_token_id = tokenizer.eos_token_id  # Ensure generation stops at EOS token

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=diversity_param,  # Adjust randomness
            top_p=0.8,  # Nucleus sampling
            repetition_penalty=1.2,
            eos_token_id=eos_token_id  # Stop generating when EOS token is encountered
        )
    
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def load_dataset_and_generate_responses(args):
    """Loads dataset, generates responses, and saves them."""
    print(f"Loading dataset: {args.dataset_name} (split: {args.dataset_split})")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    # Load model
    print(f"Loading model: {args.model_name}")
    model, tokenizer, device = load_model(args.model_name)

    # Load existing data if resuming
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            output_data = json.load(f)
        processed_questions = {entry["question"] for entry in output_data}
        print(f"Resuming from last saved state. Found {len(output_data)} processed questions.")
    else:
        output_data = []
        processed_questions = set()


    for i, sample in enumerate(dataset.select(range(args.num_samples))):
        question = sample["problem"]

        if question in processed_questions:
            print("Skipping: ", i)
            continue  # Skip already processed questions

        response_1 = generate_response(model, tokenizer, question, args.diversity_1, device)
        response_2 = generate_response(model, tokenizer, question, args.diversity_2, device)

        output_data.append({
            "question": question,
            "response_1": response_1,
            "response_2": response_2
        })

        print(f"Processed {i+1}/{args.num_samples} questions.", flush=True)
        #print("Response 1:", response_1)
        #print("Response 2:", response_2)

        # Save responses
        with open(args.output_file, "w") as f:
            json.dump(output_data, f, indent=4)
    print(f"Responses saved in {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CoT responses using Qwen LLM.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-7B", help="LLM model name")
    parser.add_argument("--dataset_name", type=str, default="SynthLabsAI/Big-Math-RL-Verified", help="Dataset name")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split (train/test/val)")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of samples to process")
    parser.add_argument("--diversity_1", type=float, default=1.0, help="Diversity parameter for first response")
    parser.add_argument("--diversity_2", type=float, default=1.5, help="Diversity parameter for second response")
    parser.add_argument("--output_file", type=str, default="responses_7B.json", help="Output JSON file")

    args = parser.parse_args()
    load_dataset_and_generate_responses(args)
