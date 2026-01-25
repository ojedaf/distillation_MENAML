# %% [markdown]
# # 🎓 MenaML: Distillation Workshop
#
# **Author:** [Your Name]  
# **Date:** 2026
#
# ---
#
# ## Learning Objectives
#
# By the end of this workshop, you will:
# 1. Understand the core concepts behind Distillation
# 2. Implement a Teacher-Student training framework from scratch
# 3. Compare different training strategies: baseline, hard labels, soft labels
# 4. Analyze the trade-offs between model size and performance
#
# ---
# %% [markdown]
# ## 1. Introduction: What is Knowledge Distillation?
#
# **Knowledge Distillation** is a technique whereby a student neural network learns from another, usually already pre-trained neural network. KD can be used for compressing the model, in which case the student is smaller. It can also be used for improving model performance, where the student is the same or even larger than the teacher. In either case, the the student is trained to mimic the behavior of one or more teacher models.
#
# ### Why do we need it?
#
# - **Deployment constraints**: Large models are expensive to run on edge devices, mobile phones, or in real-time applications
# - **Inference speed**: Smaller models are faster
# - **Cost reduction**: Less compute = less money and energy
# - **Increased performance**: Sometimes KD is used to increase the performance of our model.
# %% [markdown]
# ## Why Does KD work?
#
# ### The "Dark Knowledge"
#
# In their seminal 2015 paper, [Hinton et al](https://arxiv.org/pdf/1503.02531). observed that the **soft probability outputs** of a teacher model contain more information than hard labels.
#
# **Example**: For a cat image, hard label says `[0, 0, 1, 0, ...]` (just "cat")  
# But soft labels might say `[0.01, 0.05, 0.85, 0.09, ...]` revealing that the image also looks a bit like a dog or tiger!
#
# This extra information about class relationships is the "dark knowledge" that helps the student learn better.
#
# ![Distillation Diagram](https://intellabs.github.io/distiller/imgs/knowledge_distillation.png)
#
#
# ### Reweighing Training Examples
#
# Even if we ignore probabilities for all classes other than the true class, we get an effect where some examples are weighted higher than other ones according to what probability the teacher assigns to the true class. [Tang et al](https://arxiv.org/pdf/2002.03532) showed that this *importance weighing* is an important component of how KD works.
# %% [markdown]
# ## 2. Setup and Imports
# %%
import collections
import os
import time
from typing import Dict, List, Optional, Tuple, Union

# Third party
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import tqdm.auto as tqdm

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
# %% [markdown]
# ## 3. Dataset: CIFAR-10
#
# We'll use CIFAR-10: 60,000 32x32 color images in 10 classes.
# %%
# Data augmentation and normalization
cifar_mean = np.asarray([0.4914, 0.4822, 0.4465])
cifar_std = np.asarray([0.2023, 0.1994, 0.2010])
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar_mean, std=cifar_std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar_mean, std=cifar_std),
])

# Download and load datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

trainloader = DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0,
)
testloader = DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=0,
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(f"Training samples: {len(trainset)}")
print(f"Test samples: {len(testset)}")
# %%
# Visualize some samples
def imshow(ax, img: torch.Tensor) -> None:
    """Helper function to un-normalize and display an image.

    Args:
        ax: Matplotlib axes to plot on.
        img (torch.Tensor): Tensor image of shape (C, H, W).

    Returns:
        None
    """
    img = img * cifar_std[:, None, None] + cifar_mean[:, None, None]
    img = np.clip(img, 0., 1.)
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))

# Get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Show images
fig, axes = plt.subplots(1, 8, figsize=(6, 1))
for i in range(8):
    # axes[i].imshow(np.transpose((images[i] / 2 + 0.5).numpy(), (1, 2, 0)))
    imshow(axes[i], images[i])
    axes[i].set_title(classes[labels[i]])
    axes[i].axis('off')
plt.tight_layout()
plt.show()
# %% [markdown]
# ## 4. Model Definitions
#
# ### Teacher Model: ResNet-18 (11M parameters)
# ### Student Model: Small CNN (< 1M parameters)
#
# The goal is to transfer knowledge from the large teacher to the tiny student.
# %%
class TeacherCNN(nn.Module):
    """
    Teacher: ResNet-18 adapted for CIFAR-10 (32x32 images)
    """
    def __init__(self, num_classes: int = 10, weights: Optional[str] = None):
        super(TeacherCNN, self).__init__()
        self.model = models.resnet18(weights=weights)
        # Modify first conv layer for 32x32 images (no aggressive downsampling)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove maxpool for small images
        self.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.fc = nn.Identity()

    def forward(self, x, output_features=False):
        x = self.model(x)
        logits = self.fc(x)
        if output_features:
            return logits, x
        return logits

def get_teacher_model(num_classes: int = 10, weights: Optional[str] = 'IMAGENET1K_V1'):
    """
    Teacher: ResNet-18 adapted for CIFAR-10 (32x32 images)
    """
    model = TeacherCNN(num_classes=num_classes, weights=weights)
    return model


class StudentCNN(nn.Module):
    """
    Student: A small CNN with ~100K parameters
    """
    def __init__(self, num_classes: int = 10, output_dim: int = 256):
        super(StudentCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16 -> 8

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8 -> 4

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4 -> 2

            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 2 -> 1
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 1 * 1, output_dim),  # 512 to match teacher
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(output_dim, num_classes),
        )

    def forward(self, x, output_features=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        logits = self.fc2(x)
        if output_features:
            return logits, x
        return logits


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# %% [markdown]
# ### Activity 1. Instantiate the Teacher and Student models
# %%
# TODO
teacher = None
student = None
# %% [markdown]
# #### Solution
# %%
# Create models and compare sizes
teacher = get_teacher_model().to(device)
student = StudentCNN().to(device)
# %% [markdown]
# ### Compare Student and Teacher Model Sizes
# %%
teacher_params = count_parameters(teacher)
student_params = count_parameters(student)

print(f"Teacher (ResNet-18) parameters: {teacher_params:,}")
print(f"Student (Small CNN) parameters: {student_params:,}")
print(f"\nCompression ratio: {teacher_params / student_params:.1f}x smaller")
# %% [markdown]
# ## 5. Training Utilities
# %%
def train_epoch(model, teacher, trainloader, criterion, optimizer, device):
    """Trains the model using supervised training or distillation.

    Distillation is used only when the teacher is present. In that case, the
    teacher is frozen (no gradients).
    """
    model.train()
    if teacher is not None:
      teacher.eval()  # Teacher is always in eval mode

    running_loss = 0.0
    running_hard_loss = 0.0
    running_soft_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm.tqdm(trainloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass for the trained model
        optimizer.zero_grad()
        logits = model(inputs)

        # Compute loss
        if teacher is not None:
            # Get teacher predictions (no gradient needed)
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            loss, hard_loss, soft_loss = criterion(
                logits, teacher_logits, labels
            )
            running_hard_loss += hard_loss.item()
            running_soft_loss += soft_loss.item()
        else:
          loss = criterion(logits, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    n = len(trainloader)
    return dict(
        total_loss=running_loss/n,
        hard_loss=running_hard_loss/n,
        soft_loss=running_soft_loss/n,
        train_acc=100.*correct/total
    )


def evaluate(model, testloader, device):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return dict(test_acc=100. * correct / total)


def train_loop(
    model: nn.Module,
    criterion: nn.Module,
    checkpoint_name: str,
    model_name: str,
    experiment_name: str,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    teacher: Optional[nn.Module] = None,
    num_epochs: int = 5,
    learning_rate: float = 0.01,
    overwrite: bool = False,
) -> dict[str, list[float]]:
  """Main training loop.

  Args:
      model: Model to train.
      criterion: Loss function.
      checkpoint_name: Filename to save checkpoint.
      model_name: Display name for the model.
      experiment_name: Display name for the experiment.
      train_data_loader: Training loader.
      test_data_loader: Test loader.
      teacher: Optional teacher model.
      num_epochs: Number of epochs.
      learning_rate: Learning rate.
      overwrite: Whether to overwrite existing checkpoints.

  Returns:
      dict[str, list[float]]: Training history.
  """
  # Setup optimizer and scheduler
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=num_epochs, eta_min=0.001
  )

  print(experiment_name)
  print("=" * 60)
  history = collections.defaultdict(list)
  checkpoint_file_name = f'{checkpoint_name}.pth'

  # Check if checkpoint exists safely
  if os.path.exists(checkpoint_file_name) and not overwrite:
    raise ValueError(f'The checkpoint {checkpoint_file_name} already exists!')

  best_test_acc = 0.0
  for epoch in range(num_epochs):
      train_start = time.time()

      # Run training epoch
      metrics = train_epoch(
          model, teacher, train_data_loader, criterion, optimizer, device
      )
      train_time = time.time() - train_start

      # Evaluation
      eval_start = time.time()
      metrics.update(evaluate(model, test_data_loader, device))
      eval_time = time.time() - eval_start

      # Update learning rate
      scheduler.step()

      # Store metrics
      for k, v in metrics.items():
          history[k].append(v)

      train_acc = metrics.pop('train_acc')
      test_acc = metrics.pop('test_acc')

      # Log progress
      print(
          f"Epoch {epoch+1:2d}/{num_epochs}",
          ' |'.join(f" {k}: {v:.4f}" for k, v in metrics.items()),
          f"| Train Acc {train_acc:.2f}% | Test Acc {test_acc:.2f}%"
          f"| Epoch time {train_time:.2f}s | Eval time {eval_time:.2f}s"
      )

      # Save best model
      if test_acc > best_test_acc:
          best_test_acc = test_acc
          torch.save(model.state_dict(), checkpoint_file_name)

  print(f"\n✅ {model_name} final test accuracy: {best_test_acc:.2f}%")
  return history
# %% [markdown]
# ## 6. Train the Teacher Model
#
# First, we need a well-trained teacher. In practice, you might use a pre-trained model, but we'll train one from scratch for educational purposes.
#
# **Note**: To save time, we'll provide the checkpoint, but you can check the training code and train your own teacher later.
# %% [markdown]
# ### 6.1. Download the Teacher Model
# %%
!gdown --id 1Ko41G-TVerBr1tY0cSr4m1h1s9PvHRXw
# %% [markdown]
# ### 6.2. Train the Teacher Model
# %%
# # Training configuration
# TEACHER_EPOCHS = 15  # Increase for better teacher (50+ recommended)
# LEARNING_RATE = 0.01

# teacher = get_teacher_model(weights='IMAGENET1K_V1').to(device)
# teacher_history = train_loop(
#     model=teacher,
#     criterion=nn.CrossEntropyLoss(),
#     train_data_loader=trainloader,
#     test_data_loader=testloader,
#     num_epochs=TEACHER_EPOCHS,
#     learning_rate=LEARNING_RATE,
#     checkpoint_name="teacher_resnet18.pth",
#     model_name="Teacher",
#     experiment_name="Teacher Model (ResNet-18)",
#     overwrite=False,

# )
# %% [markdown]
# ## 7. Baseline: Train Student WITHOUT Distillation
#
# Let's start by training the student network with only hard labels. This will allow us to appreciate the value of distillation.
# %% Training Hypers
STUDENT_EPOCHS = 5
LR = 0.01
# %%
# Train student from scratch (no teacher)
teacher_dim = teacher.fc.in_features
student_baseline = StudentCNN(output_dim=teacher_dim).to(device)

baseline_history = train_loop(
    model=student_baseline,
    criterion=nn.CrossEntropyLoss(),
    train_data_loader=trainloader,
    test_data_loader=testloader,
    num_epochs=STUDENT_EPOCHS,
    learning_rate=LR,
    checkpoint_name="baseline_student_v1",
    model_name="Baseline Student",
    experiment_name="Training Student Baseline (NO distillation)",
    overwrite=False,
    # overwrite=True,
)
# %% [markdown]
# ## 8. The Core: Knowledge Distillation Loss
#
# ### The Distillation Loss Function
# The key innovation is combining two losses:
# $$L_{total} = \alpha \cdot L_{hard} + (1 - \alpha) \cdot L_{soft}$$
# Where:
# - $L_{hard}$ = Cross-entropy with true labels (standard supervised loss)
# - $L_{soft}$ = KL divergence between teacher and student soft predictions
# - $\alpha$ = Weight balancing the two losses (typically 0.1-0.5)
# %% [markdown]
# ### Activity 2. Implement the `DistillationLoss` class in accordance with the previously provided description.
#

# %%
class DistillationLoss(nn.Module):
    """Knowledge Distillation Loss.

    Combines:
    1. Hard label loss: classification cross entropy.
    2. Soft label loss: KL divergence between the student and teacher logits.

    The T^2 factor compensates for the gradient magnitude reduction when using
    temperature.
    """
    def __init__(self, alpha: float):
        super().__init__()
        # TODO

    def forward(
          self,
          student_logits: torch.Tensor,
          teacher_logits: torch.Tensor,
          labels: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO
        return total_loss, hard_loss, soft_loss
# %% [markdown]
# #### Solution
# %%
class DistillationLoss(nn.Module):
    """Knowledge Distillation Loss.

    Combines:
    1. Hard label loss: classification cross entropy.
    2. Soft label loss: KL divergence between the student and teacher logits.

    The T^2 factor compensates for the gradient magnitude reduction when using
    temperature.
    """
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(
          self,
          student_logits: torch.Tensor,
          teacher_logits: torch.Tensor,
          labels: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Hard label loss (standard cross-entropy)
        hard_loss = self.ce_loss(student_logits, labels)

        # Soft label loss (KL divergence)
        # Student: log-softmax
        student_logits = F.log_softmax(student_logits, dim=1)
        # Teacher: softmax (target distribution)
        teacher_probs = F.softmax(teacher_logits, dim=1)

        # KL divergence * T^2 (to match gradient magnitude)
        soft_loss = self.kl_loss(student_logits, teacher_probs)

        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return total_loss, hard_loss, soft_loss
# %% [markdown]
# ## 9. Train Student with Distillation
# %% [markdown]
# ### Training with distillation (Using big network as teacher)
# %% [markdown]
# #### Activity 3. Implement the training code that uses the distillation loss defined above to distill knowledge from a large pretrained teacher model to a smaller student model.
# %%
# Distillation configuration
STUDENT_EPOCHS = 5
ALPHA = 0.1        # Weight for hard loss (try: 0.1, 0.3, 0.5)
LR = 0.01

# Load trained teacher
# TODO

# Load student
# TODO


# Define training loss
# TODO

# Start the model training.
# TODO
# %% [markdown]
# #### Solution
# %%
# Distillation configuration
STUDENT_EPOCHS = 5
ALPHA = 0.1        # Weight for hard loss (try: 0.1, 0.3, 0.5)
LR = 0.01

# Load trained teacher
teacher.load_state_dict(torch.load('teacher_resnet18.pth'))
teacher.eval()

# Load student
teacher_dim = teacher.fc.in_features
student_distilled = StudentCNN(output_dim=teacher_dim).to(device)

# Define training loss
distill_criterion = DistillationLoss(alpha=ALPHA)

# Start the model training.
distilled_history = train_loop(
    model=student_distilled,
    teacher=teacher,
    criterion=distill_criterion,
    train_data_loader=trainloader,
    test_data_loader=testloader,
    num_epochs=STUDENT_EPOCHS,
    learning_rate=LR,
    checkpoint_name="student_distilled_v1",
    model_name="Distilled Student",
    experiment_name="Training Student with Knowledge Distillation",
    overwrite=False,
    # overwrite=True,
)
# %% [markdown]
# ## 10. Results Comparison
# %%
# Final comparison
teacher_acc = evaluate(teacher, testloader, device)['test_acc']
distilled_acc = evaluate(student_distilled, testloader, device)['test_acc']
baseline_acc = evaluate(student_baseline, testloader, device)['test_acc']

print("\n" + "="*60)
print("📊 FINAL RESULTS")
print("="*60)
print(f"{'Model':<25} {'Parameters':<15} {'Test Accuracy':<15}")
print("-"*60)
print(f"{'Teacher (ResNet-18)':<25} {teacher_params:>12,} {teacher_acc:>12.2f}%")
print(f"{'Student (Baseline)':<25} {student_params:>12,} {baseline_acc:>12.2f}%")
print(f"{'Student (Distilled)':<25} {student_params:>12,} {distilled_acc:>12.2f}%")
print("-"*60)
print(f"\n📈 Distillation improvement: +{distilled_acc - baseline_acc:.2f}%")
print(f"🗜️  Compression ratio: {teacher_params / student_params:.1f}x fewer parameters")
# %% Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

xs = np.arange(1, STUDENT_EPOCHS + 1)
# Test accuracy comparison
axes[0].plot(xs, baseline_history['test_acc'], label='Student (Baseline)', linestyle='--')
axes[0].plot(xs, distilled_history['test_acc'], label='Student (Distilled)', linewidth=2)
axes[0].axhline(y=teacher_acc, color='r', linestyle=':', label=f'Teacher ({teacher_acc:.1f}%)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Test Accuracy (%)')
axes[0].set_title('Test Accuracy Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss breakdown for distilled student
axes[1].plot(xs, distilled_history['hard_loss'], label='Hard Loss (CE)', alpha=0.8)
axes[1].plot(xs, distilled_history['soft_loss'], label='Soft Loss (KL)', alpha=0.8)
axes[1].plot(xs, distilled_history['total_loss'], label='Total Loss', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Distillation Loss Components')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# %% [markdown]
# ## 11. Inference Speed Comparison
# %%
def benchmark_inference(
    model: nn.Module,
    input_size: tuple = (1, 3, 32, 32),
    num_iterations: int = 1000
) -> float:
    """Benchmark inference speed.

    Args:
        model (nn.Module): Model to benchmark.
        input_size (tuple): Input tensor shape.
        num_iterations (int): Number of iterations to average.

    Returns:
        float: Average time per inference in milliseconds.
    """
    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    # Warmup
    for _ in range(100):
        with torch.no_grad():
            _ = model(dummy_input)

    # Benchmark
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start
    return elapsed / num_iterations * 1000  # ms per inference

teacher_time = benchmark_inference(teacher)
student_time = benchmark_inference(student_distilled)

print("\n⚡ Inference Speed (single sample)")
print("-" * 40)
print(f"Teacher (ResNet-18): {teacher_time:.3f} ms")
print(f"Student (Distilled): {student_time:.3f} ms")
print(f"Speedup: {teacher_time / student_time:.2f}x faster")
# %% [markdown]
# ## 12. Temperature Scaling
# We "soften" the probability distributions using temperature $T$:
# $$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
# - $T = 1$: Normal softmax
# - $T > 1$: Softer distribution (reveals more information about class relationships)
# - Typically $T \in [3, 20]$
#
# Note that the temperature is applied to both the teacher and the student outputs.
# As the temperature T increases, the gradient of the loss gets rescaled by a factor of (1/T)^2. This has been derived in the original Hinton paper. As decreasing gradient value can affect optimization dynamics, we need to upscale the gradients by a factor of T^2. The easiest way to do it is just to multiply the loss by T^2.
# %% [markdown]
# ### 🔍 Let's Visualize Temperature Effects
#
# Understanding how temperature affects the probability distribution is crucial.
# %%
# Demonstrate temperature effect
def visualize_temperature_effect(logits: torch.Tensor, temperatures: list = [1, 2, 4, 8, 16]) -> None:
    """Visualize how temperature affects the softmax distribution.

    Args:
        logits (torch.Tensor): Output logits from a model (1D tensor).
        temperatures (list): List of temperatures to visualize.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, len(temperatures), figsize=(15, 3))

    for ax, T in zip(axes, temperatures):
        probs = F.softmax(logits / T, dim=0).numpy()
        ax.bar(range(len(probs)), probs, color='steelblue')
        ax.set_title(f'T = {T}')
        ax.set_xlabel('Class')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)
        ax.set_xticks(range(10))

    plt.suptitle('Effect of Temperature on Softmax Distribution', fontsize=14)
    plt.tight_layout()
    plt.show()

# Example logits (model is quite confident about class 3)
example_logits = torch.tensor([1.0, 2.0, 1.5, 8.0, 0.5, 2.5, 1.0, 0.8, 1.2, 1.8])
visualize_temperature_effect(example_logits)

print("\n📊 Observation:")
print("- T=1: Sharp distribution (almost one-hot)")
print("- T>1: Softer distribution revealing class relationships")
print("- Higher T = more 'dark knowledge' transferred")
# %% [markdown]
# ### Activity 4. Implement the `TemperedDistillationLoss` class.
# It should be similar to `DistillationLoss` but with the temperature scaling applied to the teacher and student probabilities, and rescaled loss.
#

# %%
class TemperedDistillationLoss(nn.Module):
    """Knowledge Distillation Loss with Temperature Scaling.

    Combines:
    1. Hard label loss: CrossEntropy(student_output, true_labels)
    2. Soft label loss: KLDiv(student_soft, teacher_soft) * T^2

    The T^2 factor compensates for the gradient magnitude reduction when using
    temperature.
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.3) -> None:
        """Initialize TemperedDistillationLoss.

        Args:
            temperature (float): Temperature for softening the distribution.
            alpha (float): Weight for hard loss (1-alpha for soft loss).
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the tempered distillation loss.

        Args:
            student_logits: Logits from student model.
            teacher_logits: Logits from teacher model.
            labels: True labels.

        Returns:
            tuple: (total_loss, hard_loss, soft_loss)
        """
        # Hard label loss (standard cross-entropy)
        hard_loss = self.ce_loss(student_logits, labels)

        # TODO: student_soft and teacher_soft with temperature scaling.
        student_soft = None
        teacher_soft = None

        # KL divergence * T^2 (to match gradient magnitude)
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)

        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return total_loss, hard_loss, soft_loss
# %% [markdown]
# #### Solution
# %%
class TemperedDistillationLoss(nn.Module):
    """Knowledge Distillation Loss with Temperature Scaling.

    Combines:
    1. Hard label loss: CrossEntropy(student_output, true_labels)
    2. Soft label loss: KLDiv(student_soft, teacher_soft) * T^2

    The T^2 factor compensates for the gradient magnitude reduction when using
    temperature.
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.3) -> None:
        """Initialize TemperedDistillationLoss.

        Args:
            temperature (float): Temperature for softening the distribution.
            alpha (float): Weight for hard loss (1-alpha for soft loss).
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the tempered distillation loss.

        Args:
            student_logits: Logits from student model.
            teacher_logits: Logits from teacher model.
            labels: True labels.

        Returns:
            tuple: (total_loss, hard_loss, soft_loss)
        """
        # Hard label loss (standard cross-entropy)
        hard_loss = self.ce_loss(student_logits, labels)

        # Soft label loss (KL divergence with temperature)
        # Student: log-softmax with temperature
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        # Teacher: softmax with temperature (target distribution)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        # KL divergence * T^2 (to match gradient magnitude)
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)

        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return total_loss, hard_loss, soft_loss
# %% [markdown]
# ### Training with temperature-scaled distillation (Using big network as teacher)
# %%
# Distillation configuration
ALPHA = 0.1        # Weight for hard loss (try: 0.1, 0.3, 0.5)
TEMPERATURE = 4.0  # Try values in [1, 20]
LR = 0.01


# Load trained teacher
teacher.load_state_dict(torch.load('teacher_resnet18.pth'))
teacher.eval()

teacher_dim = teacher.fc.in_features
student_distilled = StudentCNN(output_dim=teacher_dim).to(device)

# Setup training
tempered_distill_criterion = TemperedDistillationLoss(
    temperature=TEMPERATURE,
    alpha=ALPHA
)

tempered_distillation_history = train_loop(
    model=student,
    teacher=teacher,
    criterion=tempered_distill_criterion,
    train_data_loader=trainloader,
    test_data_loader=testloader,
    num_epochs=STUDENT_EPOCHS,
    learning_rate=LR,
    checkpoint_name="student_distilled_tempered_v1",
    model_name="Tempered-distillation Student",
    experiment_name="Training Student with Tempered Knowledge Distillation",
    overwrite=False,
)
# %% Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Test accuracy comparison (axes[0] remains as is, it already has distinct labels and lines)
xs = np.arange(1, len(baseline_history['test_acc']) + 1)
axes[0].plot(xs, baseline_history['test_acc'], label='Student (Baseline)', linestyle='--')
axes[0].plot(xs, distilled_history['test_acc'], label='Student (Distilled)', linewidth=2)
axes[0].plot(xs, tempered_distillation_history['test_acc'], label='Student (Tempered Distillation)', linewidth=2)
axes[0].axhline(y=teacher_acc, color='r', linestyle=':', label=f'Teacher ({teacher_acc:.1f}%)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Test Accuracy (%)')
axes[0].set_title('Test Accuracy Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss breakdown for distilled student (axes[1])
# Define consistent colors for the loss components
hard_loss_color = 'tab:blue'
soft_loss_color = 'tab:orange'
total_loss_color = 'tab:green'

# Plot original distillation losses with solid lines
axes[1].plot(xs, distilled_history['hard_loss'], label='Hard Loss (Distilled)', alpha=0.8, color=hard_loss_color)
axes[1].plot(xs, distilled_history['soft_loss'], label='Soft Loss (Distilled)', alpha=0.8, color=soft_loss_color)
axes[1].plot(xs, distilled_history['total_loss'], label='Total Loss (Distilled)', linewidth=2, color=total_loss_color)

# Plot tempered distillation losses with dashed lines and the same colors
axes[1].plot(xs, tempered_distillation_history['hard_loss'], alpha=0.8, color=hard_loss_color, linestyle='--')
axes[1].plot(xs, tempered_distillation_history['soft_loss'], alpha=0.8, color=soft_loss_color, linestyle='--')
axes[1].plot(xs, tempered_distillation_history['total_loss'], linewidth=2, color=total_loss_color, linestyle='--')
axes[1].plot([], [], linewidth=2, linestyle='--', c='k', label='Tempered')

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Distillation Loss Components')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# %% [markdown]
# ## 13. Distillation Aids Generalization
# (particularly in a low data context)
# %%
# Grayscale + Noise (old camera simulation)
class AddGaussianNoise:
    """Add Gaussian noise to a tensor."""
    def __init__(self, mean: float = 0., std: float = 0.1) -> None:
        """Initialize the transform.

        Args:
            mean (float): Mean of the noise.
            std (float): Standard deviation of the noise.
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply noise to the tensor.

        Args:
            tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Noisy tensor.
        """
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)

transform_train_grayscale_noisy = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    AddGaussianNoise(mean=0., std=0.02),
    transforms.Normalize(cifar_mean, cifar_std),
])

transform_test_grayscale_noisy = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    AddGaussianNoise(mean=0., std=0.02),
    transforms.Normalize(cifar_mean, cifar_std),
])

# Download and load datasets
augmented_cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_grayscale_noisy)
augmented_cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test_grayscale_noisy)

augmented_cifar10_trainloader = DataLoader(augmented_cifar10_train, batch_size=128, shuffle=True, num_workers=0)
augmented_cifar10_testloader = DataLoader(augmented_cifar10_test, batch_size=128, shuffle=False, num_workers=0)
# %% [markdown]
# ### Activity 5. Plot some instances
# Similar to the original CIFAR-10 dataset, plot a few sample instances to assess the impact of the applied transformation.
# %%
# Get some random training images
dataiter_v2 = iter(augmented_cifar10_trainloader)
images, labels = next(dataiter_v2)

# Show images
# TODO
# %% [markdown]
# #### Solution
# %%
# Get some random training images
dataiter_v2 = iter(augmented_cifar10_trainloader)
images, labels = next(dataiter_v2)

# Show images
fig, axes = plt.subplots(1, 8, figsize=(6, 1))
for i in range(8):
    imshow(axes[i], images[i])
    axes[i].set_title(classes[labels[i]])
    axes[i].axis('off')
plt.tight_layout()
plt.show()
# %% [markdown]
# ### Activity 6. Train the student without distillation on the new dataset.
# %%
# TODO
# %% [markdown]
# #### Solution
# %%
# Train student from scratch (no teacher)
STUDENT_EPOCHS = 5
LR = 0.01

teacher_dim = teacher.fc.in_features
aug_cifar10_student_baseline = StudentCNN(output_dim=teacher_dim).to(device)

aug_cifar10_student_baseline_history = train_loop(
    model=aug_cifar10_student_baseline,
    teacher=None,
    criterion=nn.CrossEntropyLoss(),
    train_data_loader=augmented_cifar10_trainloader,
    test_data_loader=augmented_cifar10_testloader,
    num_epochs=STUDENT_EPOCHS,
    learning_rate=LR,
    checkpoint_name="augmented_cifar_student_no_distillation",
    model_name="Augmented Cifar-10 Baseline",
    experiment_name="Training Student Baseline (NO distillation)",
    overwrite=False,
)
# %% [markdown]
# ### Activity 7. Train the student network on the new dataset using the TemperedDistillationLoss class, and compare the results against those obtained from training the student model independently.
# %%
# TODO
# %% [markdown]
# #### Solution
# %%
# Distillation configuration
STUDENT_EPOCHS = 5
TEMPERATURE = 4.0  # Try: 2, 4, 8, 20
ALPHA = 0.1        # Weight for hard loss (try: 0.1, 0.3, 0.5)
LR = 0.01

# Load trained teacher
teacher.load_state_dict(torch.load('teacher_resnet18.pth'))
teacher.eval()

teacher_dim = teacher.fc.in_features
aug_cifar10_student_distilled = StudentCNN(output_dim=teacher_dim).to(device)
distill_criterion = TemperedDistillationLoss(temperature=TEMPERATURE, alpha=ALPHA)

aug_cifar10_student_distilled_history = train_loop(
    model=aug_cifar10_student_distilled,
    teacher=teacher,
    criterion=distill_criterion,
    train_data_loader=augmented_cifar10_trainloader,
    test_data_loader=augmented_cifar10_testloader,
    num_epochs=STUDENT_EPOCHS,
    learning_rate=LR,
    checkpoint_name="augmented_cifar_distilled_student",
    model_name="Augmented Cifar-10 Distilled Student",
    experiment_name="Training Distilled student on augmented Cifar-10",
    overwrite=False,
)
# %% Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

xs = np.arange(1, STUDENT_EPOCHS + 1)
# Test accuracy comparison
axes[0].plot(xs, aug_cifar10_student_baseline_history['test_acc'], label='Student (Baseline)', linestyle='--')
axes[0].plot(xs, aug_cifar10_student_distilled_history['test_acc'], label='Student (Distilled)', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Test Accuracy (%)')
axes[0].set_title('Test Accuracy Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

hard_loss_col = "tab:blue"
# Loss breakdown for distilled student
axes[1].plot(xs, aug_cifar10_student_distilled_history['hard_loss'], label='Hard Loss (CE)', alpha=0.8)
axes[1].plot(xs, aug_cifar10_student_distilled_history['soft_loss'], label='Soft Loss (KL)', alpha=0.8)
axes[1].plot(xs, aug_cifar10_student_distilled_history['total_loss'], label='Total Loss', linewidth=2)
axes[1].plot(
    xs, aug_cifar10_student_baseline_history['total_loss'],
    label='Hard Loss (Baseline)', alpha=0.8, c=hard_loss_col, linestyle='--'
)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Distillation Loss Components')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# %% [markdown]
# ## 13. Born Again Networks (Use the previous trained model (Student) as a Teacher).
# %% [markdown]
# You train a model once (the Teacher). Then, you take a fresh version of the exact same architecture (the Student) and train it to not only hit the correct labels but also to mimic the "confidence" of the Teacher. Surprisingly, the Student almost always ends up smarter than the Teacher. This concept was introduced by [Furlanello et al](https://arxiv.org/pdf/1805.04770).
# %%
# Distillation configuration
STUDENT_EPOCHS = 30
TEMPERATURE = 4.0  # Try: 2, 4, 8, 20
ALPHA = 0.1        # Weight for hard loss (try: 0.1, 0.3, 0.5)
LR = 0.01

# Load trained teacher
teacher_dim = teacher.fc.in_features
teacher_st_baseline = StudentCNN(output_dim=teacher_dim).to(device)
teacher_st_baseline.load_state_dict(torch.load('student_baseline.pth'))
teacher_st_baseline.eval()

student_distilled = StudentCNN(output_dim=teacher_dim).to(device)

# Setup training
distill_criterion = TemperedDistillationLoss(temperature=TEMPERATURE, alpha=ALPHA)

ban_history = train_loop(
    model=student,
    teacher=teacher,
    criterion=tempered_distill_criterion,
    num_epochs=STUDENT_EPOCHS,
    learning_rate=LR,
    checkpoint_name="ban_v1",
    model_name="Born-Again Network",
    experiment_name="Training Student Born-Again Network",
)
# %% [markdown]
# ## 14. 🧪 Experiment: Hyperparameter Sensitivity
#
# Try different values of **Temperature** and **Alpha** to see their effects!
# %%
def quick_distillation_experiment(temperature, alpha, epochs=15):
    """Quick experiment with different hyperparameters"""
    student = StudentCNN().to(device)
    criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    optimizer = optim.Adam(student.parameters(), lr=0.001)

    for epoch in range(epochs):
        train_with_distillation(student, teacher, trainloader, criterion, optimizer, device)

    return evaluate(student, testloader, device)

# Uncomment to run experiments (takes a few minutes)
# print("Running hyperparameter experiments...")

# temperatures = [1, 2, 4, 8, 16]
# alphas = [0.1, 0.3, 0.5, 0.7]

# results = {}
# for T in temperatures:
#     for a in alphas:
#         acc = quick_distillation_experiment(T, a)
#         results[(T, a)] = acc
#         print(f"T={T}, α={a}: {acc:.2f}%")
# %% [markdown]
# ## 15. Types of Knowledge Distillation
#
# What we implemented is **Response-based Distillation**. There are other types:
#
# | Type | What's Transferred | Example |
# |------|-------------------|--------|
# | **Response-based** | Final output logits | What we did! |
# | **Feature-based** | Intermediate representations | FitNets, Attention Transfer |
# | **Relation-based** | Relationships between samples | Contrastive distillation |
#
# ### Bonus: Feature-based Distillation (FitNets)
#
# The idea is to also match intermediate feature maps, not just outputs.
# %%
class FeatureDistillationLoss(nn.Module):
    """
    Feature-based distillation: match intermediate representations.
    Requires a projection layer if dimensions don't match.
    """
    def __init__(self, device, student_dim, teacher_dim, temperature=4.0, alpha=0.3, beta=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta  # Weight for feature loss

        # Projection layer if dimensions don't match
        self.projector = nn.Linear(student_dim, teacher_dim) if student_dim != teacher_dim else nn.Identity()
        self.projector = self.projector.to(device)

        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()

    def forward(self, student_logits, teacher_logits, student_features, teacher_features, labels):
        # Standard distillation losses
        hard_loss = self.ce_loss(student_logits, labels)
        soft_loss = self.kl_loss(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)

        # Feature matching loss
        student_proj = self.projector(student_features)
        feature_loss = 1 - F.cosine_similarity(student_proj, teacher_features).mean()

        # Combined loss
        total = self.alpha * hard_loss + (1 - self.alpha - self.beta) * soft_loss + self.beta * 10 * feature_loss

        return total, hard_loss, soft_loss, feature_loss
# %%
def train_with_feature_distillation(student, teacher, trainloader, criterion, optimizer, device):
    """
    Train student using knowledge distillation from teacher.
    Teacher is frozen (no gradients).
    """
    student.train()
    teacher.eval()  # Teacher is always in eval mode

    running_loss = 0.0
    running_hard_loss = 0.0
    running_soft_loss = 0.0
    running_feature_loss = 0.0
    correct = 0
    total = 0

    tqdm_bar = tqdm.auto.tqdm
    for inputs, labels in tqdm_bar(trainloader, desc="Distilling", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        # Get teacher predictions (no gradient needed)
        with torch.no_grad():
            teacher_logits, teacher_features = teacher(inputs, output_features=True)

        # Forward pass for student
        optimizer.zero_grad()
        student_logits, student_features = student(inputs, output_features=True)

        # Compute distillation loss
        loss, hard_loss, soft_loss, feature_loss = criterion(student_logits, teacher_logits, student_features, teacher_features, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        running_hard_loss += hard_loss.item()
        running_soft_loss += soft_loss.item()
        running_feature_loss += feature_loss.item()
        _, predicted = student_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    n = len(trainloader)
    return running_loss/n, running_hard_loss/n, running_soft_loss/n, running_feature_loss/n, 100.*correct/total
# %%
# Distillation configuration
STUDENT_EPOCHS = 30
TEMPERATURE = 4.0  # Try: 2, 4, 8, 20
ALPHA = 0.1        # Weight for hard loss (try: 0.1, 0.3, 0.5)
LR = 0.01
BETA = 0.5

# Load trained teacher
teacher.load_state_dict(torch.load('teacher_resnet18.pth'))
teacher.eval()

# Create fresh student
teacher_dim = teacher.fc.in_features
student_distilled = StudentCNN(output_dim=teacher_dim).to(device)

# Setup training
student_dim = student_distilled.fc2[1].in_features
teacher_dim = teacher.fc.in_features
distill_criterion = FeatureDistillationLoss(device, student_dim, teacher_dim, temperature=TEMPERATURE, alpha=ALPHA, beta=BETA)
parameters = list(student_distilled.parameters()) + list(distill_criterion.projector.parameters())
optimizer = optim.Adam(parameters, lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STUDENT_EPOCHS, eta_min=0.001)

print(f"Training Student with Knowledge Distillation")
print(f"Temperature: {TEMPERATURE}, Alpha: {ALPHA}, Beta: {BETA}")
print("=" * 60)

distilled_history = {'total_loss': [], 'hard_loss': [], 'soft_loss': [], 'feature_loss': [],'train_acc': [], 'test_acc': []}

for epoch in range(STUDENT_EPOCHS):
    total_loss, hard_loss, soft_loss, feature_loss, train_acc = train_with_feature_distillation(
        student_distilled, teacher, trainloader, distill_criterion, optimizer, device
    )
    test_acc = evaluate(student_distilled, testloader, device)
    scheduler.step()

    distilled_history['total_loss'].append(total_loss)
    distilled_history['hard_loss'].append(hard_loss)
    distilled_history['soft_loss'].append(soft_loss)
    distilled_history['feature_loss'].append(feature_loss)
    distilled_history['train_acc'].append(train_acc)
    distilled_history['test_acc'].append(test_acc)

    print(f"Epoch {epoch+1:2d}/{STUDENT_EPOCHS} | Loss: {total_loss:.4f} (H:{hard_loss:.3f} S:{soft_loss:.3f} F:{feature_loss:.3f}) | "
          f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

print(f"\n✅ Distilled Student final test accuracy: {test_acc:.2f}%")
# %% [markdown]
# ## 16. Key Takeaways
#
# ### ✅ What We Learned
#
# 1. **Knowledge Distillation** transfers "dark knowledge" from a large teacher to a small student
#
# 2. **Soft labels** contain richer information than hard labels (class relationships)
#
# 3. **Temperature** controls how soft the probability distribution is:
#    - Higher T = more information transfer, but potentially noisier
#    - Typical values: 2-20
#
# 4. **Alpha** balances hard and soft losses:
#    - Lower α = more emphasis on mimicking teacher
#    - Higher α = more emphasis on ground truth
#
# 5. **Results**: Distilled students typically outperform students trained from scratch by 1-5%
#
# ### 🚀 Extensions to Explore
#
# - **Self-distillation**: Use the same architecture for teacher and student
# - **Online distillation**: Train teacher and student simultaneously
# - **Multi-teacher distillation**: Ensemble of teachers
# - **Task-specific distillation**: For NLP, use DistilBERT approach
#
# ### 📚 References
#
# 1. Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
# 2. Romero et al., "FitNets: Hints for Thin Deep Nets" (2015)
# 3. Gou et al., "Knowledge Distillation: A Survey" (2021)
# %% [markdown]
# ## 15. 💪 Exercise for You!
#
# Try these modifications and see what happens:
#
# 1. **Change the student architecture**: Make it deeper or wider
# 2. **Try different temperatures**: Plot accuracy vs temperature
# %%
# Your experiments here!
#
# Example: Try a deeper student
# class DeeperStudent(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Your architecture here
#         pass
# %% [markdown]
# Implement computing gradient norms and compare the student network with and without distillation, and with different temperatures during distillation
# %%
grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in model.parameters()]))
# %%