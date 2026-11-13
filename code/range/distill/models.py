import math
import torch
import torch.nn as nn

class RMALCRNNTeacher(nn.Module):

    def __init__(self, d=192):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(6, 32))
        self.view_bias = nn.Parameter(torch.zeros(6, 32))
        self.attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.proj = nn.Linear(192, 64)
        self.lrr_u = nn.Linear(64, 8, bias=False)
        self.lrr_v = nn.Linear(64, 8, bias=False)
        self.conv = nn.Sequential(nn.Conv1d(64, 96, 5, padding=2), nn.BatchNorm1d(96), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(96, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.rnn = nn.LSTM(128, 96, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.LayerNorm(192), nn.Dropout(0.35), nn.Linear(192, 2))

    def forward(self, x):
        b, t, _ = x.shape
        v = x.view(b, t, 6, 32)
        v = v * torch.sigmoid(self.gate)[None, None] + self.view_bias[None, None]
        v = v.reshape(b * t, 6, 32)
        v = self.attn(v, v, v, need_weights=False)[0].reshape(b, t, 192)
        z = self.proj(v)
        u = self.lrr_u(z)
        q = self.lrr_v(z)
        a = torch.relu(torch.einsum('btr,bsr->bts', u, q) / math.sqrt(8.0))
        z = z + 0.1 * torch.bmm(a, z)
        h = self.conv(z.transpose(1, 2)).transpose(1, 2)
        h, _ = self.rnn(h)
        return self.fc(h.mean(1))

class RMALStudentCore(nn.Module):

    def __init__(self, in_dim=192, proj_dim=64):
        super().__init__()
        self.proj = nn.Linear(in_dim, proj_dim)
        self.temporal = nn.Sequential(nn.Conv1d(proj_dim, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(96), nn.ReLU(), nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))

    def forward(self, x):
        z = torch.relu(self.proj(x)).transpose(1, 2)
        z = self.temporal(z).mean(dim=2)
        return self.head(z)

def parameter_count(model):
    return sum((p.numel() for p in model.parameters()))
