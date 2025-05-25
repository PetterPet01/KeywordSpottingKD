import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    super().__init__()
    self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                               groups=in_channels, bias=False)
    self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.depthwise(x)
    x = self.pointwise(x)
    x = self.bn(x)
    return self.relu(x)

class AudioEmbeddingNet(nn.Module):
  def __init__(self, embedding_dim=64, num_classes=None):
    super().__init__()
    self.features = nn.Sequential(
      DepthwiseSeparableConv(1, 16),
      DepthwiseSeparableConv(16, 32, stride=2),
      DepthwiseSeparableConv(32, 64, stride=2)
    )
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.embedding = nn.Linear(64, embedding_dim)
    self.classifier = nn.Linear(embedding_dim, num_classes) if num_classes else None

  def forward(self, x):
    x = self.features(x)
    x = self.pool(x)
    x = x.view(x.size(0), -1)
    emb = self.embedding(x)
    if self.classifier:
      return emb, self.classifier(emb)
    return emb

