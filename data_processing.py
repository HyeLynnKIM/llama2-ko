from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("beomi/open-llama-2-ko-7b")


def format_kullm(sample):
    instruction = f"### 질문: {sample['instruction']}"
    input = f"{sample['input']}" if len(sample["input"]) > 0 else None
    output = f"### 답변\n{sample['output']}"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, input, output] if i is not None])
    return prompt


# template dataset to add prompt to each sample
def template_dataset(sample):
    sample["text"] = f"당신은 유능한 어시스턴트 입니다. 주어지는 질문에 대해서 올바르게 답해주세요. {format_kullm(sample)}{tokenizer.eos_token}"
    return sample


# apply prompt template per sample
dataset = load_dataset("nlpai-lab/kullm-v2", split="train")

# Shuffle the dataset
dataset_shuffled = dataset.shuffle(seed=42)
dataset = dataset.map(template_dataset, remove_columns=list(dataset.features))

print(dataset[0])