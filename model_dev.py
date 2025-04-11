import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os, random, numpy as np

SEED = 42
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float32)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class MultiHeadGatedAttentionMIL(nn.Module):
    def __init__(
            self,
            num_classes=2,
            backbone='r18',
            pretrained=True,
            L=512,
            D=128,
            feature_dropout=0.1,
            attention_dropout=0.1,
            shared_attention=True):

        super().__init__()
        self.L = L
        self.D = D
        self.num_classes = num_classes
        self.shared_attention = shared_attention

        # Feature extractor
        if pretrained:
            if backbone == 'r18':
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                self.feature_extractor = models.resnet18(weights=weights)
            elif backbone == 'r34':
                weights = models.ResNet34_Weights.IMAGENET1K_V1
                self.feature_extractor = models.resnet34(weights=weights)
            elif backbone == 'r50':
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                self.feature_extractor = models.resnet50(weights=weights)
        else:
            self.feature_extractor = models.resnet18()

        self.feature_extractor.fc = Identity()

        # Attention mechanism (Shared or Separate)
        if shared_attention:
            self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        else:
            self.attention_V = nn.ModuleList([
                nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
                for _ in range(self.num_classes)
            ])
            self.attention_U = nn.ModuleList([
                nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
                for _ in range(self.num_classes)
            ])
        self.attention_weights = nn.ModuleList([
            nn.Linear(self.D, 1) for _ in range(self.num_classes)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(self.L, 1, bias=False) for _ in range(self.num_classes)
        ])
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.attention_dropouts = nn.ModuleList([
            nn.Dropout(attention_dropout) for _ in range(self.num_classes)
        ])


    def forward(self, x, N=None):
        bs, num_instances, ch, w, h = x.shape
        x = x.view(bs * num_instances, ch, w, h)
        H = self.feature_extractor(x)
        H = self.feature_dropout(H)
        H = H.view(bs, num_instances, -1) # (bs, num_instances, L)
        M = []
        A_all = []

        for i in range(self.num_classes):
            if self.shared_attention:
                A_V = self.attention_V(H)
                A_U = self.attention_U(H)
            else:
                A_V = self.attention_V[i](H)
                A_U = self.attention_U[i](H)

            A = self.attention_weights[i](A_V * A_U)
            A = torch.transpose(A, 2, 1)  # (bs, 1, num_instances)
            A = self.attention_dropouts[i](A)
            A = F.softmax(A, dim=2)
            A_all.append(A)
        
        As = torch.cat(A_all, dim=1) # bs, num_classes, num_instances)
        if not self.training and N:
            with torch.enable_grad():
                As = As.requires_grad_(True)
                As.retain_grad()
                M = torch.matmul(As, H) # (bs, num_classes, L)
                Y, As, importance = self.causal_counterfactual_dropout(As, H, M, N)
        else:
            M = torch.matmul(As, H) # (bs, num_classes, L)
            Y = [self.classifiers[i](M[:, i, :]) for i in range(self.num_classes)]
            Y = torch.cat(Y, dim=-1)  # (bs, num_classes)
    
        return Y, As
    

    def causal_counterfactual_dropout(self, As, H, M, N=50):
        Y = [self.classifiers[i](M[:, i, :]) for i in range(self.num_classes)]
        Y = torch.cat(Y, dim=-1)  # (bs, num_classes)
        Y = Y.requires_grad_(True)
        Y.retain_grad()
        # compute gradients of the output w.r.t. attention weights
        grads = torch.autograd.grad( # (bs, num_classes, num_instances)
            outputs=Y.sum(dim=-1),
            inputs=As,
            retain_graph=True,
            create_graph=True
        )[0]

        importance = torch.sigmoid(grads * As) 
        importance = importance / (importance.sum(dim=-1, keepdim=True) + 1e-8)
        
        counterfactual_Ys = []
        counterfactual_attentions = []

        for _ in range(N):
            dropout_mask = torch.bernoulli(importance).bool()
            A_cf = As * (~dropout_mask)  # (bs, num_classes, num_instances)
            # A_all += torch.ones_like(A_all) # tak jak w kolaboracyjnym jednym????????
            M_cf = torch.matmul(A_cf, H)    # (bs, num_classes, L)
            Y_cf = [self.classifiers[i](M_cf[:, i, :]) for i in range(self.num_classes)]
            Y_cf = torch.cat(Y_cf, dim=-1)
            counterfactual_Ys.append(Y_cf.unsqueeze(0))
            counterfactual_attentions.append(A_cf.unsqueeze(0))

        # (n_samples, bs, num_classes),
        # (n_samples, bs, num_classes, num_instances)
        # (bs, num_classes, num_instances)
        return (
            torch.cat(counterfactual_Ys, dim=0),
            torch.cat(counterfactual_attentions, dim=0),
            importance.detach()
        )


# %%
if __name__ == '__main__':
    model = MultiHeadGatedAttentionMIL().to('cuda')
    sample_input = torch.randn(1,4,3,64,64).to('cuda')
    sample_input.requires_grad = True
    model.eval()
    output, attention_weights = model(sample_input, N=10)
    print(output.shape)
# %%