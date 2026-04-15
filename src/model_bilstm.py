import torch
import torch.nn as nn
import torchvision.models as models
from src.model import get_model

class CNN_BiLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1, dropout=0.3, config=None, num_classes=1):
        super(CNN_BiLSTM, self).__init__()

        backbone = get_model(num_classes=num_classes, config=config)

        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_dim = 1280

        self.bilstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)

        features = self.cnn(x)
        features = self.pool(features)
        features = features.view(batch_size * seq_len, -1)

        features = features.view(batch_size, seq_len, self.feature_dim)

        lstm_out, _ = self.bilstm(features)
        lstm_out = lstm_out[:, -1, :]

        out = self.fc(lstm_out)

        return out