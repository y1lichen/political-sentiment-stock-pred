import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        
    def forward(self, market_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([market_feat, text_feat], dim=1))  
        return g * text_feat + (1 - g) * market_feat

class DualBranchNet(nn.Module):
    def __init__(self, text_dim: int, market_dim: int, hidden_dim: int = 128, lstm_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout), 
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.market_lstm = nn.LSTM(input_size=market_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.fusion = GatedFusion(hidden_dim)
        self.fusion_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 3))

    def forward(self, market_x: torch.Tensor, text_x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.market_lstm(market_x)
        return self.fusion_head(self.fusion(h_n[-1], self.text_mlp(text_x)))