import os
import json
import torch
import boto3
from botocore.exceptions import NoCredentialsError
from urllib.parse import urlparse
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

def upload_to_s3(model_dir, model_name, s3_details):
    # Initialize S3 client
    s3 = boto3.client(
        's3',
        region_name=s3_details['region']
    )

    # Upload each file in the model directory to S3
    for filename in os.listdir(model_dir):
        file_path = os.path.join(model_dir, filename)
        if os.path.isfile(file_path):
            try:
                s3.upload_file(file_path, s3_details['bucket_name'], f"{model_name.replace('/', '_')}/{filename}")
                print(f"Uploaded {filename} to S3 bucket {s3_details['bucket_name']}")
            except FileNotFoundError:
                print(f"The file {file_path} was not found")
            except NoCredentialsError:
                print("Credentials not available")


def main():
    # Load the configuration from config.json
    config_path = 'config.json'
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # Get S3 details
    s3_details = config['s3']

    # Iterate through models in the configuration
    for model_name, options in config['models'].items():
        print(f"Model name: {model_name}")
        if options.get("download", False):
            model_dir = save_model(model_name)
            if options.get("save_to_s3", False):
                upload_to_s3(model_dir, model_name, s3_details)

if __name__ == "__main__":
    main()

