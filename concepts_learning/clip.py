import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from transformers import BertModel

class ImageEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # Load pretrained ResNet but remove the final classification layer
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Add projection head
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        projected = self.projection(features)
        return F.normalize(projected, dim=-1)  # L2 normalize

class TextEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # Load pretrained BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Add projection head
        self.projection = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get BERT [CLS] token output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [batch_size, 768]
        projected = self.projection(cls_output)
        return F.normalize(projected, dim=-1)  # L2 normalize

class CLIP(nn.Module):
    def __init__(self, output_dim=512, temperature=0.07):
        super().__init__()
        self.image_encoder = ImageEncoder(output_dim)
        self.text_encoder = TextEncoder(output_dim)
        self.temperature = temperature
        
    def forward(self, images, input_ids, attention_mask):
        # Get normalized embeddings
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        
        # Compute cosine similarity
        logits = torch.matmul(image_features, text_features.t()) / self.temperature
        
        # Create labels for contrastive learning
        labels = torch.arange(len(images), device=images.device)
        
        # Compute loss in both directions
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2

    def get_similarity(self, images, input_ids, attention_mask):
        with torch.no_grad():
            image_features = self.image_encoder(images)
            text_features = self.text_encoder(input_ids, attention_mask)
            similarity = torch.matmul(image_features, text_features.t())
        return similarity
