# File: proto_MAML_train_quaternion.py

# Ensure Models directory is in Python path if running from a different root
import sys
import os
# Assuming this script is in a directory like 'PROTO-MAML_Project/scripts'
# and 'Models' is 'PROTO-MAML_Project/Models'
# Or if script is in PROTO-MAML/ and Models is PROTO-MAML/Models
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from data_loader import ESC50Dataset
from proto_net import Protonet  # Original Protonet class
from utils import episodic_sampling, manage_checkpoints  # Original utils
from Models.q_bc_resnet_encoder import QBcResNetEncoderASM  # Our new encoder

import torch
import pandas as pd
import numpy as np
import learn2learn as l2l
from tqdm import tqdm

# --- Configuration ---
BASE_DATA_PATH = '../ESC-50/audio'  # Specify your dataset audio path
META_DATA_PATH = '../ESC-50/meta/esc50.csv'  # Specify your dataset meta path
CHECKPOINT_DIR_ROOT = '../MC_checkpoints_Quaternion'  # Specify your checkpoint path root

# Encoder Configuration (matches QUATERNION setup where applicable)
ENCODER_SCALE = 6  # From QUATERNION example: QBCResnet6QORGASM
ENCODER_DROPOUT = 0.2
ENCODER_USE_SUBSPECTRAL = True  # As per QBCResnet6QORGASM
N_MELS = 40  # Matched to QBcResNetEncoderASM's expected input

# MAML/Protonet Configuration
N_WAY = 5
K_SHOT = 5  # For both support and query for simplicity in this example, or adjust sampling
# K_SHOT_QUERY = 5 # if different
EPISODES_PER_EPOCH = 10  # Number of tasks per meta-update
META_LR = 1e-3
FAST_LR = 0.2  # Inner loop learning rate for MAML
EPOCHS = 100  # Reduced for quick test, original was 10000
MAML_STEPS = 8  # Inner loop adaptation steps
FIRST_ORDER_MAML = True  # Use first-order MAML approximation

# --- Setup ---
checkpoint_suffix = f'{N_WAY}-way-{K_SHOT}-shot'
checkpoint_dir = os.path.join(CHECKPOINT_DIR_ROOT, checkpoint_suffix)
os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

np.random.seed(66)
torch.manual_seed(66)
if device == "cuda":
    torch.cuda.manual_seed_all(66)

# --- Data Loading ---
meta_data = pd.read_csv(META_DATA_PATH)

classes = np.arange(meta_data['target'].nunique())  # Generalize for datasets not 0-49
np.random.shuffle(classes)
train_classes = classes[:25]
val_classes = classes[25:40]  # Unused in this MAML setup, but kept for consistency
test_classes = classes[40:]  # Unused in this MAML setup

train_meta = meta_data[meta_data.target.isin(train_classes)]
# val_meta = meta_data[meta_data.target.isin(val_classes)]
# test_meta = meta_data[meta_data.target.isin(test_classes)]

# Pass N_MELS to dataset
train_dataset = ESC50Dataset(BASE_DATA_PATH, train_meta, target_n_mels=N_MELS)
# val_dataset = ESC50Dataset(BASE_DATA_PATH, val_meta, target_n_mels=N_MELS)
# test_dataset = ESC50Dataset(BASE_DATA_PATH, test_meta, target_n_mels=N_MELS)

train_dataset.meta_data = train_dataset.meta_data.reset_index(drop=True)
# val_dataset.meta_data = val_dataset.meta_data.reset_index(drop=True)
# test_dataset.meta_data = test_dataset.meta_data.reset_index(drop=True)

# --- Model Definition ---
# 1. Create the Quaternion BC-ResNet Encoder
quaternion_encoder = QBcResNetEncoderASM(
    scale=ENCODER_SCALE,
    dropout=ENCODER_DROPOUT,
    use_subspectral=ENCODER_USE_SUBSPECTRAL
).to(device)

# 2. Create Protonet using this encoder
#    Protonet itself does not have trainable parameters beyond its encoder.
proto_model = Protonet(encoder=quaternion_encoder).to(device)

# 3. Wrap with MAML
#    The original code used GBML with MetaCurvatureTransform. Using MAML for simplicity first.
# maml = l2l.algorithms.MAML(proto_model, lr=FAST_LR, first_order=FIRST_ORDER_MAML, allow_nograd=True) # allow_nograd for Protonet if it has no direct params

# Using GBML as in the original if MetaCurvature is desired (requires more setup if transform has params)
# For GBML, the module passed should be the one whose parameters are meta-learned.
# Protonet's "loss" method is what MAML/GBML will differentiate through.
# The `proto_model` (which is Protonet instance) is the module here.
maml = l2l.algorithms.GBML(
    module=proto_model,  # Pass the Protonet instance
    transform=l2l.optim.ModuleTransform(lambda p: torch.optim.SGD(p, lr=FAST_LR)),  # Simpler transform
    # transform=l2l.optim.MetaCurvatureTransform, # If using MetaCurvature
    lr=FAST_LR,  # This lr is for the transform, not the meta-optimizer
    adapt_transform=False,  # Adapt the transform's parameters? Usually False for SGD.
    first_order=FIRST_ORDER_MAML,
).to(device)

meta_opt = torch.optim.Adam(maml.parameters(), lr=META_LR)  # Optimize MAML's (cloned model's) parameters
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=meta_opt, T_max=EPOCHS // 10 if EPOCHS >= 10 else 1,
                                                       eta_min=1e-5)

# --- Training Loop ---
print(f"Starting training with Quaternion BC-ResNet Encoder for {EPOCHS} epochs.")
print(f"  Encoder: scale={ENCODER_SCALE}, dropout={ENCODER_DROPOUT}, subspectral_norm={ENCODER_USE_SUBSPECTRAL}")
print(f"  MAML: {N_WAY}-way, {K_SHOT}-shot, {MAML_STEPS} adaptation steps, fast_lr={FAST_LR}")
print(f"  Meta-optimizer: Adam lr={META_LR}")

for epoch in range(EPOCHS):
    meta_train_loss = 0.0
    meta_train_acc = 0.0

    PRE_VAL_epoch = []
    TRAIN_ACC_epoch = []
    POST_VAL_epoch = []

    for episode in tqdm(range(EPISODES_PER_EPOCH), desc=f'Epoch {epoch + 1}/{EPOCHS}', leave=False):
        task_model = maml.clone()  # Create a clone for the current task

        # Sample support and query sets for the task
        # episodic_sampling expects k_shot for support, and +1 for query (total k_shot+1 items per class)
        # The Protonet.loss logic handles n_support and n_query based on input tensor shapes.
        # Let's ensure episodic_sampling provides k_shot for support and k_shot for query.
        # For this, we need k_shot*2 samples per class.
        # Or, modify episodic_sampling / Protonet.loss to be more flexible.

        # The provided episodic_sampling:
        # ways = random.sample(base_classes.tolist(), c_way)
        # indicies = random.sample(indicies, k_shot + 1) # k_shot for support, 1 for query
        # support = data[indicies[0]][0] if k_shot == 1 else torch.stack([data[i][0] for i in indicies[:-1]])
        # query = data[indicies[-1]][0]
        # So, it's k_shot support, 1 query image per class.
        # Protonet.loss expects query of shape (n_class, n_query_per_class, C, H, W)
        # And support of shape (n_class, n_support_per_class, C, H, W)

        # Let's use k_shot for support and k_shot for query as a common setup
        # We'll need to modify episodic_sampling or how we call it.
        # For now, let's use k_shot for support, and 1 for query (as per original episodic_sampling)
        # And Protonet.loss will see n_query = 1.

        s_v, q_v = episodic_sampling(N_WAY, K_SHOT, train_classes, train_dataset)
        s_v, q_v = s_v.to(device), q_v.to(device)
        # s_v shape: (N_WAY, K_SHOT, 1, N_MELS, N_FRAMES) or (N_WAY, 1, N_MELS, N_FRAMES) if K_SHOT=1
        # q_v shape: (N_WAY, 1, 1, N_MELS, N_FRAMES) (since episodic_sampling gives 1 query item)
        # Protonet.loss expects q_v to be (N_WAY, N_QUERY_ITEMS, 1, N_MELS, N_FRAMES)
        # Current q_v is (N_WAY, 1, N_MELS, N_FRAMES) after squeeze(1) if needed.
        # Reshape q_v for Protonet.loss: (N_WAY, 1 query item, C, H, W)
        if q_v.dim() == 4:  # (N_WAY, C, H, W) -> (N_WAY, 1, C, H, W)
            q_v = q_v.unsqueeze(1)

        # Evaluate before adaptation (optional, for tracking)
        with torch.no_grad():
            pre_loss, _, pre_acc = task_model.module.loss(s_v, q_v)  # module gives access to Protonet
            PRE_VAL_epoch.append(pre_acc.item())

        # Adapt the model to the task
        for _ in range(MAML_STEPS):
            # The original RDFT logic is complex. For standard MAML, adapt on support set, evaluate on query.
            # Protonet.loss is defined for a support/query pair.
            # MAML typically adapts by minimizing loss on the support set (or a part of it).
            # Here, the adaptation is on s_f, q_f from the original code, which seems like a leave-one-out on support.
            # Let's use the support set (s_v) to compute prototypes and a small query set (q_v) for adaptation loss.

            # Standard MAML adaptation: use (s_v, q_v) for adaptation loss
            # The original code uses a leave-one-out from s_v for adaptation. This is fine for Protonet's structure.

            # Using the RDFT-like adaptation from the prompt:
            adapt_loss_sum = 0
            adapt_acc_sum = 0
            num_adapt_samples = 0
            if K_SHOT > 1:  # Leave-one-out only makes sense if K_SHOT > 1
                for j_adapt in range(s_v.size(1)):  # Iterate over support samples to leave one out
                    s_f = torch.stack([torch.cat((s_v[i, :j_adapt], s_v[i, j_adapt + 1:])) for i in range(s_v.size(0))])
                    q_f_single = s_v[:, j_adapt, :].unsqueeze(
                        1)  # The left-out sample becomes the query (N_WAY, 1, C,H,W)

                    loss, _, acc = task_model.module.loss(s_f, q_f_single)
                    task_model.adapt(loss)  # Adapt MAML's fast weights
                    adapt_loss_sum += loss.item()
                    adapt_acc_sum += acc.item()
                    num_adapt_samples += 1
            else:  # K_SHOT = 1, cannot do leave-one-out. Adapt on (s_v, q_v) directly.
                loss, _, acc = task_model.module.loss(s_v, q_v)
                task_model.adapt(loss)
                adapt_loss_sum = loss.item()
                adapt_acc_sum = acc.item()
                num_adapt_samples = 1

            if num_adapt_samples > 0:
                TRAIN_ACC_epoch.append(adapt_acc_sum / num_adapt_samples)

        # Evaluate after adaptation on the original query set q_v
        # This loss contributes to the meta-objective
        post_loss, _, post_acc = task_model.module.loss(s_v, q_v)
        POST_VAL_epoch.append(post_acc.item())

        meta_train_loss += post_loss  # Accumulate loss for meta-update
        meta_train_acc += post_acc.item()

    # Meta-update
    meta_opt.zero_grad()
    meta_loss_avg = meta_train_loss / EPISODES_PER_EPOCH
    meta_loss_avg.backward()
    meta_opt.step()
    scheduler.step()

    avg_pre_acc = np.mean(PRE_VAL_epoch) if PRE_VAL_epoch else 0
    avg_train_acc = np.mean(TRAIN_ACC_epoch) if TRAIN_ACC_epoch else 0
    avg_post_acc = np.mean(POST_VAL_epoch) if POST_VAL_epoch else 0

    print(f"Epoch [{epoch + 1}/{EPOCHS}] Meta-Loss: {meta_loss_avg.item():.4f} | "
          f"Avg Pre-Adapt Acc: {avg_pre_acc:.4f} | "
          f"Avg Adapt Acc: {avg_train_acc:.4f} | "
          f"Avg Post-Adapt Acc: {avg_post_acc:.4f}")

    if (epoch + 1) % 10 == 0:  # Save checkpoint every 10 epochs
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'maml_state_dict': maml.state_dict(),  # Save MAML state (includes proto_model)
            'meta_optimizer_state_dict': meta_opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_classes': train_classes,  # Save class split for consistent testing
            'config': {  # Save key configurations
                'encoder_scale': ENCODER_SCALE, 'n_mels': N_MELS,
                'n_way': N_WAY, 'k_shot': K_SHOT, 'maml_steps': MAML_STEPS
            }
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        manage_checkpoints(checkpoint_dir, max_checkpoints=5)  # Keep last 5 checkpoints

print("Training finished.")