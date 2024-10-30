from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
import torch
from torch.utils.data import DataLoader
import accelerate

def train_model(
    pretrained_model_name="runwayml/stable-diffusion-v1-5",
    dataset=None,
    output_dir="fine_tuned_model",
    num_epochs=100,
    learning_rate=1e-5,
    batch_size=1
):
    # Load the pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name,
        torch_dtype=torch.float16
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    # Prepare optimizer
    optimizer = torch.optim.AdamW(
        pipeline.unet.parameters(),
        lr=learning_rate
    )
    
    # Prepare dataloader
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            # Get the inputs
            images = batch["image"].to(device)
            prompts = batch["prompt"]
            
            # Forward pass
            with torch.no_grad():
                latents = pipeline.vae.encode(images).latent_dist.sample()
                latents = latents * pipeline.vae.config.scaling_factor
            
            # Train step
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipeline.scheduler.num_train_timesteps, (batch_size,))
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
            
            # Predict the noise
            noise_pred = pipeline.unet(noisy_latents, timesteps, prompts)["sample"]
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            pipeline.save_pretrained(f"{output_dir}/checkpoint-{epoch+1}")
            
    # Save final model
    pipeline.save_pretrained(output_dir)