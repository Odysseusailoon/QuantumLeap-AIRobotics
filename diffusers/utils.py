from datasets import Dataset
import os
from PIL import Image

def create_dataset(image_dir, prompt_file):
    images = []
    prompts = []
    
    # Load prompts from file
    with open(prompt_file, 'r') as f:
        prompt_list = f.readlines()
    
    # Load and process images
    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, img_file)
            image = Image.open(image_path)
            # Resize image to 512x512
            image = image.resize((512, 512))
            images.append(image)
            
    # Create dataset
    dataset_dict = {
        "image": images,
        "prompt": prompt_list
    }
    
    return Dataset.from_dict(dataset_dict)