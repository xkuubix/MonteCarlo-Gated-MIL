import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange, reduce

class GatedAttentionMIL(nn.Module):
    def __init__(self, resnet_type="resnet18", attention_dim=128, output_dim=1, dropout_rate=0.5):
        super(GatedAttentionMIL, self).__init__()

        self.resnet = getattr(models, resnet_type)(weights="IMAGENET1K_V1")
        self.feature_dim = list(self.resnet.children())[-1].in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  
        self.flatten = nn.Flatten()

        self.attention_V = nn.Linear(self.feature_dim, attention_dim)
        self.attention_U = nn.Linear(self.feature_dim, attention_dim)
        self.attention_weights = nn.Linear(attention_dim, 1)

        self.feature_dropout = nn.Dropout(dropout_rate)
        self.attention_dropout = nn.Dropout(dropout_rate)

        self.classifier = nn.Linear(self.feature_dim, output_dim)

    def _extract_features(self, x):
        x = rearrange(x, "b n c h w -> (b n) c h w")  
        features = self.resnet(x)
        features = self.flatten(features)  
        return features

    def _apply_dropout(self, features):
        return self.feature_dropout(features)

    def _compute_attention(self, features, batch_size, num_patches):
        A_V = self.attention_V(features)
        A_U = self.attention_U(features)
        A = torch.tanh(A_V) * torch.sigmoid(A_U)
        A = self.attention_dropout(A)  
        A = self.attention_weights(A)  
        A = F.softmax(A, dim=1)  
        features = rearrange(features, "(b n) d -> b n d", b=batch_size, n=num_patches)
        return A, features

    def _classify(self, aggregated_features):
        output = self.classifier(aggregated_features)
        output = torch.sigmoid(output)
        return output

    def forward(self, x):
        batch_size, num_patches, C, H, W = x.shape
        features = self._extract_features(x)
        features = self._apply_dropout(features)
        A, features = self._compute_attention(features, batch_size, num_patches)
        aggregated_features = reduce(A * features, "b n d -> b d", "sum")
        output = self._classify(aggregated_features)
        return output, A

    def predict(self, x, mc_samples=10):
        self.train()
        self.apply(lambda m: m.eval() if isinstance(m, nn.BatchNorm2d) else None)

        batch_size, num_patches, C, H, W = x.shape
        features = self._extract_features(x)
        outputs = []
        attentions = []

        for _ in range(mc_samples):
            dropped_features = self._apply_dropout(features)
            A, dropped_features = self._compute_attention(dropped_features, batch_size, num_patches)
            aggregated_features = reduce(A * dropped_features, "b n d -> b d", "sum")
            output = self._classify(aggregated_features)
            outputs.append(output)
            attentions.append(A)
            torch.cuda.empty_cache()
            gc.collect()  

        outputs = torch.stack(outputs, dim=0)
        attentions = torch.stack(attentions, dim=0)

        return outputs, attentions