import torch
import torch.nn as nn

class ResumeMLPAdapter(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=2048, output_dim=4096):
        """
        MLP Adapter to transform Resume + Job Description embeddings into a tailored resume embedding.
        
        Args:
            input_dim (int): Dimension of a single embedding (e.g., 4096 for Llama-3).
                             The actual input to the first layer will be 2 * input_dim.
            hidden_dim (int): Dimension of hidden layers.
            output_dim (int): Dimension of the output vector.
        """
        super(ResumeMLPAdapter, self).__init__()
        
        self.input_dim = input_dim
        
        # We concatenate Resume (input_dim) + JD (input_dim) -> 2 * input_dim
        combined_dim = input_dim * 2
        
        self.network = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Skip connection for validation before training (Mock Trainer)
        # Projects combined input down to output_dim linearly
        self.skip_projection = nn.Linear(combined_dim, output_dim)
        
        # Initialize weights (optional, PyTorch defaults are usually fine for random init)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, resume_emb, jd_emb, use_skip_connection=True):
        """
        Forward pass.
        
        Args:
            resume_emb (torch.Tensor): Tensor of shape (batch, input_dim) or (input_dim,)
            jd_emb (torch.Tensor): Tensor of shape (batch, input_dim) or (input_dim,)
            use_skip_connection (bool): If True, uses the linear skip projection (useful for untrained models).
            
        Returns:
            torch.Tensor: Transformed embedding.
        """
        # Determine device from module parameters
        device = next(self.parameters()).device

        # Ensure inputs are tensors and move to device
        if not isinstance(resume_emb, torch.Tensor):
            resume_emb = torch.tensor(resume_emb, dtype=torch.float32)
        if not isinstance(jd_emb, torch.Tensor):
            jd_emb = torch.tensor(jd_emb, dtype=torch.float32)
            
        resume_emb = resume_emb.to(device)
        jd_emb = jd_emb.to(device)
            
        # Add batch dimension if missing
        if resume_emb.dim() == 1:
            resume_emb = resume_emb.unsqueeze(0)
        if jd_emb.dim() == 1:
            jd_emb = jd_emb.unsqueeze(0)
            
        # Concatenate: [Batch, Resume_Dim + JD_Dim]
        combined = torch.cat((resume_emb, jd_emb), dim=1)
        
        if use_skip_connection:
            # For "untrained" debugging: simpler linear transform
            return self.skip_projection(combined)
        else:
            return self.network(combined)
