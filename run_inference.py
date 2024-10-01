import json
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

def prepare_images(image_urls):
    images = []
    for url in image_urls:
        response = requests.get(url, stream=True)
        image = Image.open(response.raw)
        images.append(image)
    return images

def generate_response(model, processor, images, prompt):
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    tokenized_prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(tokenized_prompt, images, return_tensors="pt").to("cuda:0")
    
    generation_args = {
        "max_new_tokens": 1000,
        "temperature": 0.0,
        "do_sample": False,
    }
    
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    
    # Remove input tokens
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response

def main():
    config = load_config('config.json')
    
    for model_name, model_info in config['models'].items():
        if model_info['run_inference']:
            model_id = model_name
            model_path = model_info['path']
            model_type = model_info['type']
            
            print(f"Loading model: {model_name} from {model_path}")
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, num_crops=4)
            
            if model_type == "vision":
                # Prepare image URLs from SlideShare
                image_urls = [
                    f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg"
                    for i in range(1, 10)
                ]
                images = prepare_images(image_urls)
                prompt = "<|image_1|>\n<|image_2|>\nCan you provide a summary of the contents in these images?"
                response = generate_response(model, processor, images, prompt)
                print(f"Response for {model_name}: {response}")
            else:
                # Handle text models (can add specific logic here if needed)
                prompt = "What can you tell me about the capabilities of text models?"
                messages = [{"role": "user", "content": prompt}]
                tokenized_prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(tokenized_prompt, return_tensors="pt").to("cuda:0")
                
                generate_ids = model.generate(**inputs)
                response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
                print(f"Response for {model_name}: {response}")

if __name__ == "__main__":
    main()
