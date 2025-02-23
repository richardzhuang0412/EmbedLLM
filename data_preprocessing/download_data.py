import os
from huggingface_hub import hf_hub_download

# Hugging Face dataset repository details
repo_id = "RZ412/EmbedLLM"
files = ["train.csv", "test.csv", "val.csv", "model_order.csv", "question_order.csv"] 

output_dir = "../data/"
os.makedirs(output_dir, exist_ok=True)

for file in files:
    print(f"Downloading {file}...")
    downloaded_file = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=file, local_dir=output_dir)
    print(f"Saved {file} to {downloaded_file}")