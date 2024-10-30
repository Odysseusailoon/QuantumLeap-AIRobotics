from tqdm import tqdm
import logging
from diffusers.utils import load_image, export_to_video
from diffusers.basic_model import VideoGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_video_with_monitoring(
    image_path,
    output_path,
    num_retries=3
):
    for attempt in range(num_retries):
        try:
            with VideoGenerator() as generator:
                image = load_image(image_path)
                image = image.resize((1024, 576))
                
                logger.info(f"Generating video: Attempt {attempt + 1}")
                with tqdm(total=25, desc="Generating frames") as pbar:
                    def callback(step, timestep, latents):
                        pbar.update(1)
                    
                    frames = generator.pipe(
                        image,
                        num_frames=14,
                        num_inference_steps=25,
                        callback=callback,
                        callback_steps=1
                    ).frames[0]
                
                export_to_video(frames, output_path, fps=7)
                logger.info(f"Video successfully generated: {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == num_retries - 1:
                logger.error("All attempts failed")
                return False
            
            torch.cuda.empty_cache()
            continue