import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    def __init__(self, hidden_dim: int, text_gate_bias: float = -2.0):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        nn.init.constant_(self.gate[0].bias, text_gate_bias)
        
    def forward(self, market_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([market_feat, text_feat], dim=1))  
        return g * text_feat + (1 - g) * market_feat

class DualBranchNet(nn.Module):
    def __init__(self, text_dim: int, market_dim: int, hidden_dim: int = 128, lstm_layers: int = 1, dropout: float = 0.2, num_classes: int = 3, text_gate_bias: float = -2.0):
        super().__init__()
        self.text_lstm = nn.LSTM(input_size=text_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.text_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.market_lstm = nn.LSTM(input_size=market_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.fusion = GatedFusion(hidden_dim, text_gate_bias=text_gate_bias)
        self.fusion_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, market_x: torch.Tensor, text_x: torch.Tensor) -> torch.Tensor:
        if text_x.dim() == 2:
            text_x = text_x.unsqueeze(1)
        _, (market_h, _) = self.market_lstm(market_x)
        _, (text_h, _) = self.text_lstm(text_x)
        return self.fusion_head(self.fusion(market_h[-1], self.text_head(text_h[-1])))
