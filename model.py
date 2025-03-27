import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class Flatten(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        input_shape = x.shape
        output_shape = [input_shape[i] for i in range(self.dim)] + [-1]
        return x.view(*output_shape)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class GatedAttentionMIL(nn.Module):

    def __init__(
                self,
                num_classes=1,
                backbone='r18',
                pretrained=True,
                L=512,
                D=128,
                K=1,
                feature_dropout=0.1,
                attention_dropout=0.1):

        super().__init__()
        self.L = L  
        self.D = D
        self.K = K
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
        self.num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = Identity()

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Dropout(attention_dropout)
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid(),
            nn.Dropout(attention_dropout)
        )
        self.attention_weights = nn.Linear(self.D, self.K)
        self.classifier = nn.Sequential(
                                        nn.Linear(self.L * self.K,
                                                  num_classes))
        self.feature_dropout = nn.Dropout(feature_dropout)

    def forward(self, x):
        bs, num_instances, ch, w, h = x.shape
        x = x.view(bs*num_instances, ch, w, h)
        H = self.feature_extractor(x)
        H = self.feature_dropout(H)
        H = H.view(bs, num_instances, -1)
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_weights(torch.mul(A_V, A_U))
        A = torch.transpose(A, 2, 1)
        A = F.softmax(A, dim=2)
        m = torch.matmul(A, H)
        Y = self.classifier(m)
        return Y, A

    def mc_inference(self, input_tensor, n=30, device='cuda'):
        self.eval()
        self.to(device)
        input_tensor = input_tensor.to(device)

        def enable_dropout(m):
            if isinstance(m, torch.nn.Dropout):
                m.train()

        self.apply(enable_dropout)

        # Feature extraction only once
        with torch.no_grad():
            bs, num_instances, ch, w, h = input_tensor.shape
            input_tensor = input_tensor.view(bs * num_instances, ch, w, h)
            H = self.feature_extractor(input_tensor)
            H = H.view(bs, num_instances, -1)

        predictions = []
        attention_weights = []
        
        with torch.no_grad():
            for _ in range(n):
                H_dropout = self.feature_dropout(H)  # Apply dropout in each pass

                A_V = self.attention_V(H_dropout)
                A_U = self.attention_U(H_dropout)
                A = self.attention_weights(torch.mul(A_V, A_U))
                A = torch.transpose(A, 2, 1)
                A = F.softmax(A, dim=2)

                m = torch.matmul(A, H_dropout)
                outputs = self.classifier(m)
                outputs = F.sigmoid(outputs)

                predictions.append(outputs)
                attention_weights.append(A)

                torch.cuda.empty_cache()
                gc.collect()

        predictions = torch.stack(predictions)  # Shape: (n, batch_size, num_classes)
        attention_weights = torch.stack(attention_weights)  # Shape: (n, batch_size, num_instances, K)
        return predictions, attention_weights



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
        # TODO: adjust L for resnet50
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

        # Attention weight layers (Always separate per class)
        self.attention_weights = nn.ModuleList([
            nn.Linear(self.D, 1) for _ in range(self.num_classes)
        ])

        # Classifiers per class
        self.classifiers = nn.ModuleList([
            nn.Linear(self.L, 1, bias=False) for _ in range(self.num_classes)
        ])

        # Dropout layers
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.attention_dropouts = nn.ModuleList([
            nn.Dropout(attention_dropout) for _ in range(self.num_classes)
        ])

    def forward(self, x):
        bs, num_instances, ch, w, h = x.shape
        x = x.view(bs * num_instances, ch, w, h)
        H = self.feature_extractor(x)
        H = self.feature_dropout(H)
        H = H.view(bs, num_instances, -1)

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
            M.append(torch.matmul(A, H))

        M = torch.cat(M, dim=1)  # (bs, num_classes, L)
        A_all = torch.cat(A_all, dim=1)  # (bs, num_classes, num_instances)

        Y = [self.classifiers[i](M[:, i, :]) for i in range(self.num_classes)]
        Y = torch.cat(Y, dim=-1)  # (bs, num_classes)

        return Y, A_all


    def mc_inference(self, input_tensor, n=30, device='cuda'):
        """
        Performs Monte Carlo (MC) Dropout inference for uncertainty estimation.
        
        Args:
            input_tensor (torch.Tensor): Input of shape (bs, num_instances, ch, w, h).
            n (int): Number of stochastic forward passes.
            device (str): Device ('cuda' or 'cpu').

        Returns:
            predictions (torch.Tensor): Stacked predictions (n, bs, num_classes).
            attention_weights (torch.Tensor): Stacked attention weights (n, bs, num_classes, num_instances).
        """
        self.eval()
        self.to(device)
        input_tensor = input_tensor.to(device)

        # Enable dropout layers during inference
        def enable_dropout(m):
            if isinstance(m, torch.nn.Dropout):
                m.train()

        self.apply(enable_dropout)

        # Feature extraction (done once)
        with torch.no_grad():
            bs, num_instances, ch, w, h = input_tensor.shape
            input_tensor = input_tensor.view(bs * num_instances, ch, w, h)
            H = self.feature_extractor(input_tensor)
            H = H.view(bs, num_instances, -1)  # Shape: (bs, num_instances, L)

        predictions = []
        attention_weights_list = []

        with torch.no_grad():
            for _ in range(n):
                H_dropout = self.feature_dropout(H)  # Apply dropout to extracted features (MCDO)

                M = []
                A_all = []

                for i in range(self.num_classes):
                    if self.shared_attention:
                        A_V = self.attention_V(H_dropout)
                        A_U = self.attention_U(H_dropout)
                    else:
                        A_V = self.attention_V[i](H_dropout)
                        A_U = self.attention_U[i](H_dropout)

                    A = self.attention_weights[i](A_V * A_U)
                    A = torch.transpose(A, 2, 1)  # (bs, 1, num_instances)
                    A = self.attention_dropouts[i](A)  # Apply dropout to attention (MCDO)
                    A = F.softmax(A, dim=2)

                    A_all.append(A)
                    M.append(torch.matmul(A, H_dropout))

                M = torch.cat(M, dim=1)  # (bs, num_classes, L)
                A_all = torch.cat(A_all, dim=1)  # (bs, num_classes, num_instances)

                Y = [self.classifiers[i](M[:, i, :]) for i in range(self.num_classes)]
                Y = torch.cat(Y, dim=-1)  # (bs, num_classes)

                predictions.append(Y)
                attention_weights_list.append(A_all)

                torch.cuda.empty_cache()
                gc.collect()

        predictions = torch.stack(predictions)  # (n, bs, num_classes)
        attention_weights_list = torch.stack(attention_weights_list)  # (n, bs, num_classes, num_instances)

        return predictions, attention_weights_list


