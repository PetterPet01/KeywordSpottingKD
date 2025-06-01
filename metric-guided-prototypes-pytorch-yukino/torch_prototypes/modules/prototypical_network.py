import torch
import torch.nn as nn
from torch_prototypes.metrics.distortion import Pseudo_Huber
from torch_scatter import scatter_mean
import numpy as np

# Attempt to import KDTree, but make it an optional dependency for the module itself
try:
    from sklearn.neighbors import KDTree

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KDTree = None  # Placeholder


class LearntPrototypes(nn.Module):
    """
    Learnt Prototypes Module. Classification module based on learnt prototypes, to be wrapped around a backbone
    embedding network.
    """

    def __init__(
            self,
            model,
            n_prototypes,
            embedding_dim,
            prototypes=None,
            squared=False,
            ph=None,
            dist="euclidean",
            use_manual_distance=False,  # <<< ADDED PARAMETER
            device="cuda",
    ):
        """
        Args:
            model (nn.Module): feature extracting network
            n_prototypes (int): number of prototypes to use
            embedding_dim (int): dimension of the embedding space
            prototypes (tensor): Prototype tensor of shape (n_prototypes x embedding_dim),
            squared (bool): Whether to use the squared Euclidean distance or not
            ph (float): if specified, the distances function is huberized with delta parameter equal to the specified value
            dist (str): default 'euclidean', other possibility 'cosine'
            use_manual_distance (bool): If True, calculates distances manually instead of using torch.cdist or nn.CosineSimilarity.
            device (str): device on which to declare the prototypes (cpu/cuda)
        """
        super(LearntPrototypes, self).__init__()
        self.model = model

        self._prototypes_are_learnable = prototypes is None

        if prototypes is None:
            initial_prototypes = torch.rand((n_prototypes, embedding_dim), device=device)
        else:
            initial_prototypes = prototypes.to(device)

        self.prototypes = nn.Parameter(
            initial_prototypes,
            requires_grad=self._prototypes_are_learnable
        )

        self.n_prototypes = n_prototypes
        self.embedding_dim = embedding_dim
        self.squared = squared
        self.dist = dist
        self.ph = None if ph is None else Pseudo_Huber(delta=ph)
        self.use_manual_distance = use_manual_distance  # <<< STORE PARAMETER

        self.kdtree = None
        self.use_kdtree_for_inference = False
        self._kdtree_prototypes_normalized = False

        self.original_prototypes_data = None
        self.original_n_prototypes = None
        self.is_augmented_for_few_shot = False

    def enable_kdtree_inference(self):
        self.use_kdtree_for_inference = True;
        self.build_kdtree()

    def disable_kdtree_inference(self):
        self.use_kdtree_for_inference = False

    def build_kdtree(self):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required to use KD-Tree. Please install it.")

        if self.prototypes is None or self.prototypes.shape[0] == 0:
            print("Warning: Cannot build KD-Tree with no prototypes.")
            self.kdtree = None
            return

        prototypes_np = self.prototypes.data.cpu().numpy()
        self._kdtree_prototypes_normalized = False

        if self.dist == "cosine":
            norm = np.linalg.norm(prototypes_np, axis=1, keepdims=True)
            prototypes_np = prototypes_np / (norm + 1e-8)
            self._kdtree_prototypes_normalized = True

        self.kdtree = KDTree(prototypes_np)
        print(f"KD-Tree built with {prototypes_np.shape[0]} prototypes.")

    def _forward_kdtree(self, embeddings):
        if self.kdtree is None:
            raise RuntimeError("KD-Tree not built. Call build_kdtree() first or disable use_kdtree_for_inference.")

        query_embeddings_np = embeddings.detach().cpu().numpy()

        if self.dist == "cosine" and self._kdtree_prototypes_normalized:
            norm = np.linalg.norm(query_embeddings_np, axis=1, keepdims=True)
            query_embeddings_np = query_embeddings_np / (norm + 1e-8)

        kdtree_euclidean_dists_np, kdtree_indices_np = self.kdtree.query(query_embeddings_np, k=1)

        kdtree_euclidean_dists = torch.from_numpy(kdtree_euclidean_dists_np.squeeze(-1)).to(embeddings.device,
                                                                                            dtype=torch.float32)
        kdtree_indices = torch.from_numpy(kdtree_indices_np.squeeze(-1)).to(embeddings.device).long()

        scores = torch.full((embeddings.size(0), self.prototypes.shape[0]), float('-inf'),
                            device=embeddings.device, dtype=embeddings.dtype)

        final_dists = kdtree_euclidean_dists

        if self.dist == "cosine":
            # KD-Tree with L2 norm on normalized vectors: d^2 = (n1-n2)^T(n1-n2) = n1^T n1 - 2 n1^T n2 + n2^T n2
            # Since n1, n2 are normalized, n1^T n1 = 1, n2^T n2 = 1. So d^2 = 2 - 2 n1^T n2 = 2(1 - cos_sim)
            # So, (kdtree_euclidean_dists)^2 / 2 = 1 - cos_sim. This is the "distance" we want.
            cosine_dists = (kdtree_euclidean_dists.pow(2)) / 2.0
            final_dists = cosine_dists

        if self.ph is not None:
            final_dists = self.ph(final_dists.to(torch.float32))

        if self.squared:  # This applies after PH, consistent with non-kdtree path
            final_dists = final_dists.pow(2)

        source_values = -final_dists.to(scores.dtype)
        scores[torch.arange(embeddings.size(0), device=embeddings.device), kdtree_indices] = source_values

        return scores

    def forward(self, *input, **kwargs):
        embeddings = self.model(*input, **kwargs)

        original_shape = embeddings.shape
        two_dim_data = False
        b, c_dim, h, w = -1, -1, -1, -1

        if len(embeddings.shape) == 4:
            two_dim_data = True
            b, c_dim, h, w = embeddings.shape
            embeddings_reshaped = (
                embeddings.view(b, c_dim, h * w)
                .transpose(1, 2)
                .contiguous()
                .view(b * h * w, c_dim)
            )
        else:
            embeddings_reshaped = embeddings

        current_prototypes = self.prototypes.to(embeddings_reshaped.device)

        if not self.training and self.use_kdtree_for_inference:
            scores = self._forward_kdtree(embeddings_reshaped)
        else:
            current_prototypes_casted = current_prototypes.to(embeddings_reshaped.dtype)

            # <<< MODIFICATION FOR MANUAL DISTANCE >>>
            if self.use_manual_distance:
                if self.dist == "cosine":
                    # Manual Cosine Similarity: (A dot B) / (||A|| * ||B||)
                    # Then distance is 1 - similarity
                    # embeddings_reshaped: (N, D), current_prototypes_casted: (M, D)
                    norm_embeddings = torch.linalg.norm(embeddings_reshaped, dim=1, keepdim=True)  # (N, 1)
                    norm_prototypes = torch.linalg.norm(current_prototypes_casted, dim=1, keepdim=True)  # (M, 1)

                    # Add epsilon for numerical stability
                    normalized_embeddings = embeddings_reshaped / (norm_embeddings + 1e-8)
                    normalized_prototypes = current_prototypes_casted / (norm_prototypes + 1e-8)

                    # Cosine similarity: (N, D) @ (D, M) -> (N, M)
                    sim = torch.matmul(normalized_embeddings, normalized_prototypes.t())
                    dists = 1 - sim
                else:  # Manual Euclidean distance (L2 norm)
                    # dist(x,y) = sqrt(sum((x-y)^2))
                    # Efficient computation: dist(x,y)^2 = sum(x^2) - 2*sum(xy) + sum(y^2)
                    # embeddings_reshaped: (N, D), current_prototypes_casted: (M, D)
                    sum_sq_embeddings = torch.sum(embeddings_reshaped.pow(2), dim=1, keepdim=True)  # (N, 1)
                    sum_sq_prototypes = torch.sum(current_prototypes_casted.pow(2), dim=1, keepdim=True).t()  # (1, M)

                    # Dot product: (N, D) @ (D, M) -> (N, M)
                    dot_product = torch.matmul(embeddings_reshaped, current_prototypes_casted.t())

                    # Squared Euclidean distances: (N, 1) - 2*(N, M) + (1, M) results in (N, M)
                    dists_sq = sum_sq_embeddings - 2 * dot_product + sum_sq_prototypes

                    # Clamp to avoid sqrt of negative due to precision errors
                    dists_sq = torch.clamp(dists_sq, min=0.0)
                    dists = torch.sqrt(dists_sq)  # Standard L2 distance
            else:
                # Original PyTorch optimized functions
                if self.dist == "cosine":
                    sim = nn.CosineSimilarity(dim=-1)(
                        embeddings_reshaped[:, None, :], current_prototypes_casted[None, :, :]
                    )
                    dists = 1 - sim
                else:  # Euclidean
                    dists = torch.cdist(embeddings_reshaped, current_prototypes_casted, p=2)
            # <<< END MODIFICATION FOR MANUAL DISTANCE >>>

            # Common post-processing for distances (applies to both manual and torch.cdist paths)
            if self.ph is not None:
                dists = self.ph(dists.to(torch.float32))

            if self.squared:  # If squared is True, square the (possibly Huberized) distance
                # This applies AFTER the base distance (L2 or 1-cos_sim) is calculated
                dists = dists.pow(2)

            scores = -dists.to(embeddings_reshaped.dtype)

        if two_dim_data:
            scores = (
                scores.view(b, h * w, self.prototypes.shape[0])
                .transpose(1, 2)
                .contiguous()
                .view(b, self.prototypes.shape[0], h, w)
            )
        return scores

    def augment_with_few_shot_prototypes(self, support_embeddings, support_labels):
        if self.is_augmented_for_few_shot:
            print("Warning: Model is already augmented. Reverting to original before augmenting again.")
            self.revert_to_original_prototypes()

        self.original_prototypes_data = self.prototypes.data.clone()
        self.original_n_prototypes = self.prototypes.shape[0]

        support_labels = support_labels.to(support_embeddings.device).long()
        num_new_classes = (torch.max(support_labels) + 1).item()

        new_prototypes = scatter_mean(
            support_embeddings, support_labels.unsqueeze(1), dim=0, dim_size=num_new_classes
        ).detach()

        combined_prototypes = torch.cat([self.original_prototypes_data.to(new_prototypes.device), new_prototypes],
                                        dim=0)

        self.prototypes = nn.Parameter(combined_prototypes, requires_grad=False)

        print(f"Augmented with {num_new_classes} new prototypes. Total prototypes: {self.prototypes.shape[0]}")
        self.is_augmented_for_few_shot = True

        if self.use_kdtree_for_inference:
            print("Rebuilding KD-Tree with augmented prototypes...")
            self.build_kdtree()

    def revert_to_original_prototypes(self):
        if not self.is_augmented_for_few_shot or self.original_prototypes_data is None:
            print("Model was not augmented or no original prototypes saved. No action taken.")
            return

        self.prototypes = nn.Parameter(self.original_prototypes_data,
                                       requires_grad=self._prototypes_are_learnable)

        self.original_prototypes_data = None
        self.is_augmented_for_few_shot = False
        print(f"Reverted to original {self.prototypes.shape[0]} prototypes.")

        if self.use_kdtree_for_inference:
            print("Rebuilding KD-Tree with original prototypes...")
            self.build_kdtree()


class HypersphericalProto(nn.Module):
    """
    Implementation of Hypershperical Prototype Networks (Mettes et al., 2019)
    """

    def __init__(self, model, num_classes, prototypes):
        """
        Args:
            model (nn.Module): backbone feature extracting network
            num_classes (int): number of classes
            prototypes (tensor): pre-defined prototypes, tensor has shape (num_classes x embedding_dimension)
        """
        super(HypersphericalProto, self).__init__()
        self.model = model
        self.prototypes = nn.Parameter(prototypes).requires_grad_(False)
        self.num_classes = num_classes

    def forward(self, *input, **kwargs):
        embeddings = self.model(*input, **kwargs)
        current_prototypes = self.prototypes.to(embeddings.device)

        if len(embeddings.shape) == 4:  # Flatten 2D data
            two_dim_data = True
            b, c, h, w = embeddings.shape
            embeddings = (
                embeddings.view(b, c, h * w)
                .transpose(1, 2)
                .contiguous()
                .view(b * h * w, c)
            )
        else:
            two_dim_data = False

        dists = 1 - nn.CosineSimilarity(dim=-1)(
            embeddings[:, None, :], current_prototypes[None, :, :]
        )
        scores = -dists.pow(2)

        if two_dim_data:  # Un-flatten 2D data
            scores = (
                scores.view(b, h * w, self.num_classes)
                .transpose(1, 2)
                .contiguous()
                .view(b, self.num_classes, h, w)
            )
        return scores


class DeepNCM(nn.Module):
    """
    Implementation of Deep Nearest Mean Classifiers (Gueriero et al., 2017)
    """

    def __init__(self, model, num_classes, embedding_dim):
        """
        Args:
            model (nn.Module): backbone feature extracting network
            num_classes (int): number of classes
            embedding_dim (int): number of dimensions of the embedding space
        """
        super(DeepNCM, self).__init__()
        self.model = model
        prot_device = "cuda"
        self.prototypes = nn.Parameter(
            torch.rand((num_classes, embedding_dim), device=prot_device)
        ).requires_grad_(False)
        self.num_classes = num_classes
        self.counter = torch.zeros(num_classes, device=prot_device)
        self._check_device = True

    def forward(self, *input_target, **kwargs):
        input_data = input_target[:-1]
        y_true = input_target[-1]

        embeddings = self.model(*input_data, **kwargs)

        current_device = embeddings.device
        self.prototypes.data = self.prototypes.data.to(current_device)
        self.counter = self.counter.to(current_device)
        y_true = y_true.to(current_device)

        if len(embeddings.shape) == 4:
            two_dim_data = True
            b, c, h, w = embeddings.shape
            embeddings = (
                embeddings.view(b, c, h * w)
                .transpose(1, 2)
                .contiguous()
                .view(b * h * w, c)
            )
            y_true = y_true.view(b * h * w)
        else:
            two_dim_data = False

        if self.training:
            y_true_flat = y_true.reshape(-1)
            represented_classes = torch.unique(y_true_flat)

            new_prototypes_batch = scatter_mean(
                embeddings, y_true_flat.unsqueeze(1), dim=0, dim_size=self.num_classes
            ).detach()

            batch_counts = torch.bincount(y_true_flat, minlength=self.num_classes).float().unsqueeze(1).to(
                current_device)

            for cls_idx in represented_classes:
                if batch_counts[cls_idx] > 0:
                    prev_proto_weighted = self.counter[cls_idx] * self.prototypes.data[cls_idx, :]
                    new_proto_batch_weighted = batch_counts[cls_idx] * new_prototypes_batch[cls_idx, :]

                    total_count_for_cls = self.counter[cls_idx] + batch_counts[cls_idx]

                    self.prototypes.data[cls_idx, :] = (
                                                                   prev_proto_weighted + new_proto_batch_weighted) / total_count_for_cls
                    self.counter[cls_idx] = total_count_for_cls

        dists = torch.cdist(embeddings, self.prototypes.data)

        if two_dim_data:
            dists = (
                dists.view(b, h * w, self.num_classes)
                .transpose(1, 2)
                .contiguous()
                .view(b, self.num_classes, h, w)
            )

        return -dists.pow(2)