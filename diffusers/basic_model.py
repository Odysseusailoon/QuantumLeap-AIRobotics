import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
import requests

# Check GPU availability
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load the pipeline
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16,  # Use float16 for memory efficiency
    variant="fp16"
)

# Move to GPU
pipe = pipe.to("cuda")

# Enable memory optimizations
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()


def generate_video_from_image(image_path, num_frames=14, fps=7):
    # Load and prepare image
    image = load_image(image_path)
    image = image.resize((1024, 576))  # SVD preferred resolution
    
    # Generate video frames
    generator = torch.Generator(device="cuda").manual_seed(42)
    frames = pipe(
        image,
        num_frames=num_frames,
        num_inference_steps=25,
        generator=generator,
    ).frames[0]
    
    # Save video
    output_path = "generated_video.mp4"
    export_to_video(frames, output_path, fps=fps)
    return output_path

# Example usage
image_path = "your_image.jpg"
video_path = generate_video_from_image(image_path)
print(f"Video saved to: {video_path}")

def generate_video_advanced(
    image_path,
    num_frames=14,
    fps=7,
    motion_bucket_id=127,  # Controls motion intensity (1-255)
    noise_aug_strength=0.1,  # Controls deviation from input image
    seed=42
):
    image = load_image(image_path)
    image = image.resize((1024, 576))
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    frames = pipe(
        image,
        num_frames=num_frames,
        num_inference_steps=25,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        generator=generator,
    ).frames[0]
    
    output_path = f"video_motion_{motion_bucket_id}_noise_{noise_aug_strength}.mp4"
    export_to_video(frames, output_path, fps=fps)
    return output_path

class VideoGenerator:
    def __init__(self):
        self.pipe = None
        
    def load_model(self):
        if torch.cuda.is_available():
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid",
                torch_dtype=torch.float16,
                variant="fp16"
            ).to("cuda")
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()
        else:
            raise RuntimeError("GPU not available")
    
    def clear_memory(self):
        if self.pipe is not None:
            del self.pipe
            torch.cuda.empty_cache()
            self.pipe = None
    
    def __enter__(self):
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_memory()

def process_multiple_images(image_paths, output_dir="videos/"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    with VideoGenerator() as generator:
        for idx, image_path in enumerate(image_paths):
            try:
                output_path = os.path.join(
                    output_dir, 
                    f"video_{idx}.mp4"
                )
                
                image = load_image(image_path)
                image = image.resize((1024, 576))
                
                frames = generator.pipe(
                    image,
                    num_frames=14,
                    num_inference_steps=25,
                ).frames[0]
                
                export_to_video(frames, output_path, fps=7)
                print(f"Generated video {idx+1}: {output_path}")
                
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                continue