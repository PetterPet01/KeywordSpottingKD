# File: proto_MAML_train_gsc.py
import sys
import os
# Add parent directory to path if Models, etc., are there
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader_gsc import SpeechCommandsDatasetMAML, N_CLASSES_GSC, DEFAULT_LABELS, idx_to_label, label_to_idx
from utils_gsc import build_label_to_indices_map, episodic_sampling_gsc  # Using our new utils
from proto_net import Protonet  # Original Protonet class
from Models.q_bc_resnet_encoder import QBcResNetEncoderASM  # Your Quaternion encoder
from utils import manage_checkpoints  # Original utils for checkpoint management

import torch
import numpy as np
import learn2learn as l2l
from tqdm import tqdm
import pandas as pd  # Only if needed for other things, not core data loading

# --- Configuration ---
GSC_DATASET_PATH = "../data"  # Specify GSC dataset root path
CHECKPOINT_DIR_ROOT = '../MC_checkpoints_Quaternion_GSC_Yukino'

# Encoder Configuration (from QUATERNION setup)
ENCODER_SCALE = 6
ENCODER_DROPOUT = 0.2
ENCODER_USE_SUBSPECTRAL = True
# N_MELS is defined in data_loader_gsc (40)

# MAML/Protonet Configuration
N_WAY = 5
K_SHOT_SUPPORT = 5
K_SHOT_QUERY = 5  # Number of query samples per class in a task
EPISODES_PER_EPOCH = 20  # Number of tasks per meta-update (increased for GSC)
META_LR = 1e-3
FAST_LR = 0.1  # Inner loop learning rate for MAML
EPOCHS = 200  # Number of meta-epochs
MAML_STEPS = 5  # Inner loop adaptation steps
FIRST_ORDER_MAML = True

# Class split for GSC (35 classes total)
# Example: 20 for meta-train, 8 for meta-validation, 7 for meta-test
NUM_META_TRAIN_CLASSES = 20
NUM_META_VAL_CLASSES = 8  # Can be used for selecting best meta-model
NUM_META_TEST_CLASSES = N_CLASSES_GSC - NUM_META_TRAIN_CLASSES - NUM_META_VAL_CLASSES

# --- Setup ---
checkpoint_suffix = f'{N_WAY}-way-{K_SHOT_SUPPORT}s-{K_SHOT_QUERY}q-shot'
checkpoint_dir = os.path.join(CHECKPOINT_DIR_ROOT, checkpoint_suffix)
os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

np.random.seed(42)
torch.manual_seed(42)
if device == "cuda":
    torch.cuda.manual_seed_all(42)

# --- Data Loading & Splitting ---
print("Loading Google Speech Commands V2 dataset...")
# We need three dataset instances for train/val/test subsets of GSC
# For meta-learning, we split classes, not dataset instances directly.
# So, we'll load the 'training' subset of GSC, then split its classes.

# GSC 'training' subset usually contains most of the data.
# 'validation' and 'testing' contain specific files listed in validation_list.txt, testing_list.txt.
# For meta-learning, it's common to take all available classes and split them.
# Let's use the 'training' subset and split its classes.
# Or, combine all files and then split classes. For simplicity, using 'training' subset files.

full_train_dataset = SpeechCommandsDatasetMAML(subset='testing', path=GSC_DATASET_PATH,
                                               download=True)  # 'None' uses all but val/test lists
print(f"Full dataset loaded with {len(full_train_dataset)} samples.")

# Get all unique label indices present in the loaded dataset
all_present_label_indices = sorted(list(set(idx for _, idx in full_train_dataset)))
if len(all_present_label_indices) < N_CLASSES_GSC:
    print(
        f"Warning: Expected {N_CLASSES_GSC} classes, but only found {len(all_present_label_indices)} in the dataset files used.")
    # This might happen if `subset=None` doesn't actually capture all class examples or if some are missing.
    # Or if SpeechCommandsDatasetMAML filters out some files.

# Shuffle and split the *label indices*
np.random.shuffle(all_present_label_indices)  # shuffle the list of label_indices

meta_train_label_indices = all_present_label_indices[:NUM_META_TRAIN_CLASSES]
meta_val_label_indices = all_present_label_indices[
                         NUM_META_TRAIN_CLASSES: NUM_META_TRAIN_CLASSES + NUM_META_VAL_CLASSES]
meta_test_label_indices = all_present_label_indices[NUM_META_TRAIN_CLASSES + NUM_META_VAL_CLASSES:]

print(f"Meta-Train Label Indices ({len(meta_train_label_indices)}): {meta_train_label_indices}")
print(f"Meta-Val Label Indices ({len(meta_val_label_indices)}): {meta_val_label_indices}")
print(f"Meta-Test Label Indices ({len(meta_test_label_indices)}): {meta_test_label_indices}")

# Build label_to_indices map for the full_train_dataset (contains all samples we'll use)
# This map is crucial for episodic_sampling_gsc
label_to_indices_map = build_label_to_indices_map(full_train_dataset)
# wow
# --- Model Definition ---
quaternion_encoder = QBcResNetEncoderASM(
    scale=ENCODER_SCALE,
    dropout=ENCODER_DROPOUT,
    use_subspectral=ENCODER_USE_SUBSPECTRAL
).to(device)

proto_model = Protonet(encoder=quaternion_encoder).to(device)

maml = l2l.algorithms.MAML(proto_model, lr=FAST_LR, first_order=FIRST_ORDER_MAML, allow_nograd=True)
maml.to(device)  # Ensure MAML itself is on device

meta_opt = torch.optim.Adam(maml.parameters(), lr=META_LR)
scheduler = torch.optim.lr_scheduler.StepLR(meta_opt, step_size=EPOCHS // 10 if EPOCHS >= 10 else 1, gamma=0.7)

# --- Training Loop ---
print(f"Starting training with GSC V2 & Quaternion BC-ResNet Encoder for {EPOCHS} epochs.")
print(f"  {N_WAY}-way, {K_SHOT_SUPPORT}-support, {K_SHOT_QUERY}-query")

best_val_acc = 0.0

for epoch in range(EPOCHS):
    meta_train_loss_epoch = 0.0
    meta_train_acc_epoch = 0.0

    task_model = maml.clone()  # Clone for the epoch; new clone per task inside the loop

    for episode in tqdm(range(EPISODES_PER_EPOCH), desc=f'Epoch {epoch + 1}/{EPOCHS} [Meta-Train]', leave=False):

        # Sample a task for meta-training
        s_v, q_v = episodic_sampling_gsc(
            N_WAY, K_SHOT_SUPPORT, K_SHOT_QUERY,
            meta_train_label_indices,
            label_to_indices_map,
            full_train_dataset  # Pass the dataset instance
        )
        s_v, q_v = s_v.to(device), q_v.to(device)

        # Inner loop adaptation
        for _ in range(MAML_STEPS):
            # Adapt using support set to form prototypes, query set for loss
            # Protonet's loss function inherently uses support to form prototypes
            # and then computes loss against query.
            # For MAML, the loss on the *query set given support-derived prototypes* is differentiated.
            adapt_loss, _, _ = task_model.module.loss(s_v, q_v)  # task_model.module is Protonet
            task_model.adapt(adapt_loss)  # Adapt MAML's fast weights

        # Evaluate adapted model on the query set for meta-update
        eval_loss, _, eval_acc = task_model.module.loss(s_v, q_v)

        meta_train_loss_epoch += eval_loss.item()
        meta_train_acc_epoch += eval_acc.item()

        # Accumulate gradients for meta-update
        # The gradient is from eval_loss on the cloned, adapted model
        eval_loss.backward()  # This accumulates gradients in task_model's parameters

    # Meta-update: Apply accumulated gradients to the original maml model
    # learn2learn's MAML.adapt handles gradient accumulation on the clone.
    # The meta-optimizer step is on the main `maml` model's parameters.
    # The backward pass for eval_loss should affect the *original* `maml` parameters
    # through the clone mechanism, or we need to manually transfer gradients.
    # The standard L2L MAML:
    #   meta_opt.zero_grad()
    #   for task:
    #       learner = maml.clone()
    #       ... adapt learner ...
    #       evaluation_error = loss_func(learner(X_query), y_query)
    #       evaluation_error.backward() # Grads flow back to maml.parameters()
    #   meta_opt.step()

    # Corrected meta-update logic for L2L MAML:
    # Gradients should be accumulated on the original `maml` model parameters.
    # The current loop structure does one backward pass per task on `task_model`.
    # We need to sum these gradients on `maml`.
    # A simpler way with L2L is to compute the average loss and do one backward.

    meta_opt.zero_grad()  # Zero gradients for the main MAML model

    # Re-run adaptation and evaluation for meta-gradient computation
    # This is typical for MAML: one pass for adaptation, another for meta-gradient
    # Or, ensure that task_model.adapt correctly allows gradients to flow to original maml model.
    # The L2L MAML `adapt` method (if `allow_nograd=False` or `allow_unused=True` in some setups)
    # creates a computation graph that, when `eval_loss.backward()` is called,
    # allows gradients to flow back to the parameters of the *original* `maml` model.
    # So the current loop structure for backward() inside the episode loop is fine.

    # Average the gradients by scaling the accumulated gradients before optimizer step
    for p in maml.parameters():
        if p.grad is not None:
            p.grad.data.mul_(1.0 / EPISODES_PER_EPOCH)

    meta_opt.step()
    scheduler.step()

    avg_meta_train_loss = meta_train_loss_epoch / EPISODES_PER_EPOCH
    avg_meta_train_acc = meta_train_acc_epoch / EPISODES_PER_EPOCH

    print(f"Epoch [{epoch + 1}/{EPOCHS}] Meta-Train Loss: {avg_meta_train_loss:.4f} | "
          f"Meta-Train Acc: {avg_meta_train_acc:.4f}")

    # Meta-Validation (optional, but good for model selection)
    if (epoch + 1) % 5 == 0 and meta_val_label_indices:  # Validate every 5 epochs
        meta_val_loss_epoch = 0.0
        meta_val_acc_epoch = 0.0
        val_episodes = EPISODES_PER_EPOCH // 2  # Fewer episodes for validation

        # Use a fresh clone for validation, don't use the one from training loop
        val_task_model = maml.clone()  # Fresh clone from current meta-learned maml
        val_task_model.eval()  # Set to eval mode for encoder batchnorms etc.

        for _ in tqdm(range(val_episodes), desc=f'Epoch {epoch + 1}/{EPOCHS} [Meta-Val]', leave=False):
            s_v, q_v = episodic_sampling_gsc(
                N_WAY, K_SHOT_SUPPORT, K_SHOT_QUERY,
                meta_val_label_indices,
                label_to_indices_map,
                full_train_dataset
            )
            s_v, q_v = s_v.to(device), q_v.to(device)

            # Adapt on support, Test on query (no gradient tracking needed for validation)
            with torch.no_grad():  # No gradients needed for validation adaptation steps
                # Create a temporary clone for adaptation during validation
                temp_val_learner = val_task_model.clone()
                for _ in range(MAML_STEPS):
                    adapt_loss, _, _ = temp_val_learner.module.loss(s_v, q_v)
                    # Perform a "simulated" adapt step if `adapt` requires grads
                    # For Protonet, adaptation means re-calculating prototypes from s_v.
                    # The MAML wrapper's `adapt` might need to be called carefully.
                    # If Protonet's "adaptation" is just re-deriving prototypes from s_v,
                    # then the loss on q_v is all we need.
                    # Let's assume MAML's adapt step is simple SGD on fast weights.
                    # For validation, we can just evaluate the pre-adaptation state, or post.
                    # We should evaluate post-adaptation performance.
                    # To do this without gradients, manually update fast weights or use non-diff adapt.
                    # Easiest: Clone, adapt with gradients temporarily enabled, then eval.

            # Re-clone for this specific validation task to adapt
            current_val_learner = maml.clone()
            current_val_learner.eval()  # Ensure batch norms etc are in eval mode

            for _ in range(MAML_STEPS):
                # Note: loss computation for adaptation might use training=True internally in Protonet/encoder
                # For pure validation, Protonet's encoder should be in eval() mode for BN.
                # The `task_model.module.loss` will use its encoder. Ensure encoder is .eval()
                # maml.eval() should propagate this.
                adapt_loss_val, _, _ = current_val_learner.module.loss(s_v, q_v)
                current_val_learner.adapt(adapt_loss_val)  # Adapt this specific val learner

            with torch.no_grad():
                val_loss, _, val_acc = current_val_learner.module.loss(s_v, q_v)

            meta_val_loss_epoch += val_loss.item()
            meta_val_acc_epoch += val_acc.item()

        avg_meta_val_loss = meta_val_loss_epoch / val_episodes
        avg_meta_val_acc = meta_val_acc_epoch / val_episodes
        print(f"Epoch [{epoch + 1}/{EPOCHS}] Meta-Val Loss: {avg_meta_val_loss:.4f} | "
              f"Meta-Val Acc: {avg_meta_val_acc:.4f}")

        if avg_meta_val_acc > best_val_acc:
            best_val_acc = avg_meta_val_acc
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'maml_state_dict': maml.state_dict(),
                'meta_optimizer_state_dict': meta_opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'meta_train_label_indices': meta_train_label_indices,
                'meta_val_label_indices': meta_val_label_indices,
                'meta_test_label_indices': meta_test_label_indices,
                'config': {
                    'encoder_scale': ENCODER_SCALE, 'n_mels': N_MELS,
                    'n_way': N_WAY, 'k_shot_support': K_SHOT_SUPPORT, 'k_shot_query': K_SHOT_QUERY,
                    'maml_steps': MAML_STEPS
                }
            }, checkpoint_path)
            print(f"Saved new best model to {checkpoint_path} with Val Acc: {best_val_acc:.4f}")

    if (epoch + 1) % 20 == 0:  # Save periodic checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'maml_state_dict': maml.state_dict(),
        }, checkpoint_path)
        manage_checkpoints(checkpoint_dir, max_checkpoints=3)  # Keep last 3 periodic checkpoints + best

print("Training finished.")
if meta_val_label_indices:
    print(f"Best Meta-Validation Accuracy: {best_val_acc:.4f}")