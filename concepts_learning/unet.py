import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        # Create sinusoidal position embeddings
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :].to(t.device)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # Project to higher dimension
        embeddings = self.proj(embeddings)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Time embedding projection
        self.time_mlp = nn.Linear(time_dim, out_channels)
        
        # Residual connection if channels don't match
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        residual = self.shortcut(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(t)[:, :, None, None]
        x = F.silu(x + time_emb)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x + residual)
        
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_mlp = TimeEmbedding(time_dim)
        
        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        
        # Encoder
        self.down1 = ResidualBlock(64, 128, time_dim)
        self.down2 = ResidualBlock(128, 256, time_dim)
        self.down3 = ResidualBlock(256, 512, time_dim)
        
        # Bottleneck
        self.bottleneck1 = ResidualBlock(512, 512, time_dim)
        self.bottleneck2 = ResidualBlock(512, 512, time_dim)
        
        # Decoder
        self.up1 = ResidualBlock(1024, 256, time_dim)
        self.up2 = ResidualBlock(512, 128, time_dim)
        self.up3 = ResidualBlock(256, 64, time_dim)
        
        # Downsampling
        self.pool = nn.MaxPool2d(2)
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final conv
        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, t):
        # Time embedding
        t = self.time_mlp(t)
        
        # Initial conv
        x0 = self.init_conv(x)
        
        # Encoder
        x1 = self.down1(x0, t)
        x1_pool = self.pool(x1)
        
        x2 = self.down2(x1_pool, t)
        x2_pool = self.pool(x2)
        
        x3 = self.down3(x2_pool, t)
        x3_pool = self.pool(x3)
        
        # Bottleneck
        x_bot = self.bottleneck1(x3_pool, t)
        x_bot = self.bottleneck2(x_bot, t)
        
        # Decoder with skip connections
        x_up1 = self.upsample(x_bot)
        x_up1 = torch.cat([x_up1, x3], dim=1)
        x_up1 = self.up1(x_up1, t)
        
        x_up2 = self.upsample(x_up1)
        x_up2 = torch.cat([x_up2, x2], dim=1)
        x_up2 = self.up2(x_up2, t)
        
        x_up3 = self.upsample(x_up2)
        x_up3 = torch.cat([x_up3, x1], dim=1)
        x_up3 = self.up3(x_up3, t)
        
        # Final conv
        out = self.final_conv(x_up3)
        
        return out
    
def test_unet():
    # Create a sample batch
    batch_size = 4
    channels = 3
    height = 64
    width = 64
    
    # Initialize model
    model = UNet(in_channels=channels, out_channels=channels)
    
    # Create sample inputs
    x = torch.randn(batch_size, channels, height, width)
    t = torch.randint(0, 1000, (batch_size,))
    
    # Forward pass
    output = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Time shape: {t.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check if output dimensions match input
    assert output.shape == x.shape, f"Output shape {output.shape} doesn't match input shape {x.shape}"
    return "UNet test passed!"

# Now let's add the diffusion scheduling components that are typically used with UNet

class DiffusionScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        """
        Initialize the diffusion scheduler with linear noise schedule
        """
        self.num_timesteps = num_timesteps
        
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # Calculate alphas
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculate diffusion parameters
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """
        Reverse diffusion process (single step)
        """
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        # Predict noise
        model_output = model(x, t)
        
        # Calculate mean
        model_mean = sqrt_recip_alphas_cumprod_t * (
            x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        )
        
        # Add noise only if t > 0
        if t_index > 0:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(betas_t) * noise
        else:
            return model_mean

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """
        Generate samples using the reverse diffusion process
        """
        device = next(model.parameters()).device
        b = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step'):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)
        
        return img

class DiffusionTrainer:
    def __init__(self, model, scheduler, optimizer, device):
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = device
        
    def train_step(self, batch):
        """
        Single training step
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Get batch
        x_start = batch.to(self.device)
        batch_size = x_start.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=self.device)
        
        # Generate noise
        noise = torch.randn_like(x_start)
        
        # Get noisy images
        x_noisy = self.scheduler.q_sample(x_start, t, noise)
        
        # Predict noise
        noise_pred = self.model(x_noisy, t)
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Backprop
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# Example training loop
    def train_diffusion(
        model,
        dataloader,
        num_epochs=100,
        device="cuda",
        lr=2e-4,
    ):
        scheduler = DiffusionScheduler()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        trainer = DiffusionTrainer(model, scheduler, optimizer, device)
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch in tqdm(dataloader, desc=f'Epoch {epoch}'):
                loss = trainer.train_step(batch)
                epoch_losses.append(loss)
                
            avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f'Epoch {epoch} average loss: {avg_loss:.4f}')

def load_and_preprocess_image(image_path, size=64):
    # Load and preprocess a single image
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def denormalize_image(tensor):
    # Convert normalized tensor back to image
    tensor = tensor.clone().detach().cpu()
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.clamp(0, 1)
    return tensor

def test_diffusion_process():
    # Initialize models
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = UNet(in_channels=3, out_channels=3).to(device)
    scheduler = DiffusionScheduler(num_timesteps=100)  # Reduced timesteps for faster testing

    # Load your image
    image_path = "/Users/asuka/Downloads/FELV-cat.jpg"  # Replace with your image path
    original_image = load_and_preprocess_image(image_path).to(device)

    # Add value checks
    print(f"Original image range: {original_image.min():.2f} to {original_image.max():.2f}")

    def safe_denormalize(tensor):
        # Safely denormalize and handle NaN values
        tensor = tensor.clone().detach().cpu()
        tensor = torch.nan_to_num(tensor, nan=0.0)  # Replace NaN with 0
        tensor = tensor * 0.5 + 0.5
        tensor = torch.clamp(tensor, 0, 1)  # Ensure values are in [0,1]
        return tensor

    # Test reverse diffusion with value checking
    print("Testing reverse diffusion...")
    sample_shape = original_image.shape
    generated = scheduler.p_sample_loop(model, sample_shape)
    print(f"Generated image range: {generated.min():.2f} to {generated.max():.2f}")

    # Plot with safe denormalization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(safe_denormalize(original_image[0]).permute(1, 2, 0))
    plt.title("Original")
    
    plt.subplot(1, 2, 2)
    plt.imshow(safe_denormalize(generated[0]).permute(1, 2, 0))
    plt.title("Generated")
    
    plt.savefig('generation_result.png')
    plt.close()

if __name__ == "__main__":
    # Test UNet architecture
    print("Testing UNet architecture...")
    test_result = test_unet()
    print(test_result)

    # Test diffusion process
    print("\nTesting diffusion process...")
    test_diffusion_process()