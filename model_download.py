import os
import torch
from urllib.parse import urlparse
from transformers import AutoModelForCausalLM, AutoTokenizer, snapshot_download

def save_model(model_name, local_dir='./local_model'):
    # Set the random seed for reproducibility
    torch.random.manual_seed(0)

    # Get the model info
    model_info = snapshot_download(repo_id=model_name, return_info=True)

    # Print the model download URL and extract the domain
    download_url = model_info['url']
    parsed_url = urlparse(download_url)
    domain = parsed_url.netloc
    print(f"Downloading model from: {download_url}")
    print(f"Domain to open for network configuration: {domain}")

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Download the model and tokenizer locally
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save the model and tokenizer locally
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)

    # Calculate the size of the downloaded files for the current model only
    total_size_new_files = 0
    for filename in os.listdir(local_dir):
        file_path = os.path.join(local_dir, filename)
        # Check if the file was created in this run
        if os.path.isfile(file_path):
            total_size_new_files += os.path.getsize(file_path)

    print(f"Total size of downloaded files for '{model_name}': {total_size_new_files / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    # Replace 'microsoft/Phi-3.5-mini-instruct' with any model name you want to save
    model_name = "microsoft/Phi-3.5-mini-instruct"
    save_model(model_name)
