import json
import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pinecone import Pinecone, ServerlessSpec
import torch.nn as nn
from huggingface_hub import hf_hub_download


# Defining RNN model with GRU and Soft Gating
class MusicRec(nn.Module):
    def __init__(self, target_feature_count= 19):
        super().__init__()
        self.target_feature_count= target_feature_count
        self.gru = nn.GRU(19*384, 512, batch_first=True)
        self.name_gate = nn.Linear(384, 19)  # 19 features to modulate
        self.head = nn.Sequential(
            nn.Dropout(0.3), # dropout for overfitting
            nn.Linear(512, target_feature_count * 384))  # Predict all features

    def forward(self, x, name_emb):
        # x: [batch, 5, 19, 384]
        # name_emb: [batch, 384]

        # Soft gating 
        gates= torch.sigmoid(self.name_gate(name_emb)) # [batch, 19]
        x = x * gates.unsqueeze(1).unsqueeze(-1)  # [batch, 5, 19, 384]

        # flatten x to [batch, 5, 19*384]
        x = x.flatten(2)  

        # Process sequence
        _, hidden = self.gru(x)
        
        pred = self.head(hidden.squeeze(0))  # [batch, 19*384]
        return pred.view(-1, self.target_feature_count, 384)  # Reshape to [batch, 19, 384]


class Recommend:


    def init_model(self, model_path="quantized_model.pth", from_hf=True):
        """
        Initialize model from local path or Hugging Face Hub.
        
        Args:
            model_path: Local path or HF filename (if from_hf=True)
            from_hf: If True, download from Hugging Face Hub first
        """
        config = {
            "batch_size": 128,
            "lr": 3e-4,
            "epochs": 10,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }
        
        # Download from Hugging Face Hub if needed
        if from_hf:
            model_path = hf_hub_download(
                repo_id="ThistleBristle/MusicRec",
                filename=model_path,
                cache_dir="models"  # Optional: organized storage
            )
        
        # Original initialization logic
        model = MusicRec().to(config["device"])
        model.load_state_dict(torch.load(model_path, map_location=config["device"]))  # Note: map_location added
        print(f"Model loaded from {'HF' if from_hf else 'local'}: {model_path}")
        return model

    def predict_playlist(self, model ,input_sequences, title_embs, targets):
        """
        Args:
            input_sequences: [N, 5, 19, 384] tensor
            title_embs: [N, 384] tensor
            targets: [N, 19, 384] tensor (for evaluation)
        Returns:
            Dictionary of predictions and metrics
        """
        model.eval()
        results = {
            'predictions': [],
            'similarities': [],
            'mean_similarity': None
        }
        
        with torch.no_grad():
            for i in range(len(input_sequences)):
                # Get batch (unsqueeze for batch dim)
                input_seq = input_sequences[i].unsqueeze(0)  # [1, 5, 19, 384]
                title_emb = title_embs[i].unsqueeze(0)     # [1, 384]
                
                # Predict
                pred = model(input_seq, title_emb)  # [1, 19, 384]
                pred = pred.squeeze(0)              # [19, 384]
                
                # Store results
                results['predictions'].append(pred)
                
                # Calculate similarity if targets provided
                if targets is not None:
                    sim = F.cosine_similarity(
                        pred.flatten(),
                        targets[i].flatten(),
                        dim=0
                    ).item()
                    results['similarities'].append(sim)
        
        if targets is not None:
            results['mean_similarity'] = np.mean(results['similarities'])
        
        return results
            
    
