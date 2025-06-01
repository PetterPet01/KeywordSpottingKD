# File: proto_net.py (Corrected)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable  # Variable is deprecated, use torch.Tensor directly with requires_grad


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):  # Check for d being None or inconsistent
        raise ValueError(f"Dimension mismatch in euclidean_dist: x.size(1)={d}, y.size(1)={y.size(1)}")

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    # Using mean for distance, common in some prototypical networks but sum is also used.
    # Original code uses mean.
    return torch.pow(x - y, 2).mean(2)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()
        self.encoder = encoder

    def loss(self, support, query):
        # Use torch.Tensor directly, Variable is not needed for PyTorch versions >= 0.4
        xs = support  # support: (n_class, n_support, C, H, W)
        xq = query  # query: (n_class, n_query, C, H, W)

        n_class = xs.size(0)
        if xq.size(0) != n_class:  # Ensure query also has n_class ways
            raise ValueError(f"Query n_class ({xq.size(0)}) mismatch with support n_class ({n_class})")
        n_support = xs.size(1)
        n_query = xq.size(1)

        # Ensure support and query are not empty
        if n_support == 0:
            raise ValueError("Support set cannot be empty (n_support=0)")
        if n_query == 0:
            raise ValueError("Query set cannot be empty (n_query=0)")

        # Create target indices for loss calculation
        # target_inds shape: (n_class, n_query, 1)
        # It indicates that for the i-th class, all its query samples belong to class i.
        target_inds = torch.arange(0, n_class, device=xs.device).view(n_class, 1, 1).expand(n_class, n_query, 1)

        # Reshape support and query samples to be flat lists of samples
        # xs: (n_class, n_support, C, H, W) -> (n_class * n_support, C, H, W)
        # xq: (n_class, n_query, C, H, W)   -> (n_class * n_query, C, H, W)

        sample_shape = xs.size()[2:]  # Should be (Channel, Height, Width), e.g., (1, 40, 101)

        xs_flat = xs.reshape(n_class * n_support, *sample_shape)
        xq_flat = xq.reshape(n_class * n_query, *sample_shape)

        # Concatenate all samples (support and query) into a single batch
        all_samples = torch.cat([xs_flat, xq_flat], 0)
        # all_samples shape: ( (n_class * n_support) + (n_class * n_query), C, H, W )

        # Pass all samples through the encoder
        # Encoder q_bc_resnet_encoder.QBcResNetEncoderASM expects (Batch, 1, H, W) if C=1
        encoded_samples = self.encoder(all_samples)  # Output: (Total_Samples, feature_dim)

        if encoded_samples.dim() != 2:
            raise ValueError(
                f"Encoder output should be 2D (Batch, features), got {encoded_samples.dim()}D, shape {encoded_samples.shape}")

        feature_dim = encoded_samples.size(-1)

        # Separate encoded support and query features
        encoded_support_flat = encoded_samples[:n_class * n_support]
        encoded_query_flat = encoded_samples[n_class * n_support:]

        # Calculate prototypes from support features
        # Reshape encoded_support_flat to (n_class, n_support, feature_dim) to average over n_support dimension
        prototypes = encoded_support_flat.view(n_class, n_support, feature_dim).mean(dim=1)
        # prototypes shape: (n_class, feature_dim)

        # Calculate distances between query embeddings and prototypes
        # encoded_query_flat shape: (n_class * n_query, feature_dim)
        # prototypes shape: (n_class, feature_dim)
        # dists shape: (n_class * n_query, n_class)
        dists = euclidean_dist(encoded_query_flat, prototypes)

        # Log softmax over distances (negated for max to pick smallest dist)
        # Output log_p_y shape: (n_class * n_query, n_class)
        log_p_y_flat = F.log_softmax(-dists, dim=1)

        # Reshape for gathering loss according to target_inds
        # log_p_y shape: (n_class, n_query, n_class)
        log_p_y = log_p_y_flat.view(n_class, n_query, n_class)

        # Gather the log probabilities of the true classes
        # target_inds shape: (n_class, n_query, 1)
        # loss_val shape after gather: (n_class, n_query, 1)
        loss_val = -log_p_y.gather(dim=2, index=target_inds).squeeze(dim=2).view(-1).mean()

        # Calculate accuracy
        # y_hat shape: (n_class, n_query) -> predicted class indices for each query sample
        _, y_hat = log_p_y.max(dim=2)

        # target_inds.squeeze(dim=2) shape: (n_class, n_query)
        acc_val = torch.eq(y_hat, target_inds.squeeze(dim=2)).float().mean()

        return loss_val, y_hat, acc_val


def load_protonet_conv(x_dim, hid_dim, z_dim):
    # x_dim is typically (channels, height, width)
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),  # x_dim[0] is input channels
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()  # Flattens the output of conv blocks to (Batch, features)
    )

    return Protonet(encoder)