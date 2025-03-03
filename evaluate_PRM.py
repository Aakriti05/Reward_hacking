import torch
import json
import argparse
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import os


def load_prm_model(model_name, device):
    """Loads the Preference Reward Model (PRM) and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    return model, tokenizer


def make_step_rewards(logits, token_masks):
    """Extracts step-wise reward scores from model logits."""
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels

    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]  # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # Extract reward scores
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


def evaluate_responses(model, tokenizer, response_data, output_file):
    """Evaluates responses using the PRM model and saves results."""
    
    # Resume from last saved state if output file exists
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            output_data = json.load(f)
        processed_questions = {entry["question"] for entry in output_data}
        print(f"Resuming evaluation. Found {len(output_data)} already processed questions.")
    else:
        output_data = []
        processed_questions = set()

    step_sep_id = tokenizer.encode("<extra_0>")[0]

    for i, item in enumerate(response_data):
        question = item["question"]
        responses = [item["response_1"], item["response_2"]]

        if question in processed_questions:
            continue  # Skip already processed questions

        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "<extra_0>".join(responses) + "<extra_0>"},
        ]

        conversation_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        input_ids = tokenizer.encode(conversation_str, return_tensors="pt").to(model.device)
        outputs = model(input_ids=input_ids)

        token_masks = (input_ids == step_sep_id)
        step_rewards = make_step_rewards(outputs[0], token_masks)

        # Compute average reward per response
        avg_rewards = [sum(r) / len(r) if len(r) > 0 else 0 for r in step_rewards]

        # Assign 1 to preferred response, 0 to dispreferred response
        if avg_rewards[0] > avg_rewards[1]:
            preferred, dispreferred = responses[0], responses[1]
            labels = [1, 0]
        else:
            preferred, dispreferred = responses[1], responses[0]
            labels = [0, 1]

        output_data.append({
            "question": question,
            "preferred_response": preferred,
            "dispreferred_response": dispreferred,
            "labels": labels
        })

        print(f"Evaluated {i+1}/{len(response_data)} questions.", labels, flush=True)

        # Save after every evaluation to prevent data loss
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)

    print(f"Evaluation saved in {output_file}")


def main():
    """Main function to parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate responses using Qwen PRM model.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-PRM-7B", help="PRM model name")
    parser.add_argument("--input_file", type=str, default="responses.json", help="Path to input JSON file")
    parser.add_argument("--output_file", type=str, default="evaluated_responses.json", help="Path to output JSON file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model_name} on {device}")
    model, tokenizer = load_prm_model(args.model_name, device)

    # Load responses
    with open(args.input_file, "r") as f:
        response_data = json.load(f)

    evaluate_responses(model, tokenizer, response_data, args.output_file)


if __name__ == "__main__":
    main()
