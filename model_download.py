import os
import torch
from urllib.parse import urlparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def save_model(model_name, base_dir='./models'):
    # Set the random seed for reproducibility
    torch.random.manual_seed(0)

    # Create a specific folder for the model
    model_dir = os.path.join(base_dir, model_name.replace('/', '_'))  # Replace slashes for folder naming
    os.makedirs(model_dir, exist_ok=True)

    # Load the model configuration to get model information
    config = AutoConfig.from_pretrained(model_name)
    download_url = f"https://huggingface.co/{model_name}/resolve/main/pytorch_model.bin"
    parsed_url = urlparse(download_url)
    domain = parsed_url.netloc
    
    print(f"Downloading model from: {download_url}")
    print(f"Domain to open for network configuration: {domain}")

    # Download the model and tokenizer locally
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save the model and tokenizer locally in the model-specific folder
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Calculate the size of the downloaded files for the current model only
    total_size_new_files = 0
    for filename in os.listdir(model_dir):
        file_path = os.path.join(model_dir, filename)
        # Add size of the new files
        if os.path.isfile(file_path):
            total_size_new_files += os.path.getsize(file_path)

    # Convert size to GB and print the total size and path
    total_size_gb = total_size_new_files / (1024 * 1024 * 1024)  # Convert bytes to GB
    print(f"Total size of downloaded files for '{model_name}' in '{model_dir}': "
          f"{total_size_gb:.2f} GB")

if __name__ == "__main__":
    # Replace 'microsoft/Phi-3.5-mini-instruct' with any model name you want to save
    model_name = "microsoft/Phi-3.5-mini-instruct"
    save_model(model_name)
