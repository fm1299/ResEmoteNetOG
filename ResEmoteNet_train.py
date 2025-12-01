import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)  # [web:34][web:35][web:40][web:41]

from ResEmoteNet import ResEmoteNet
from get_dataset import Four4All


def compute_class_weights(train_loader, num_classes=None, class_names=None):
    """Compute class weights from the training loader.

    Returns a torch.FloatTensor of size (num_classes,) with higher weights for
    under-represented classes. Works when `labels` returned by the loader are
    integers/tensors/numpy arrays.
    """
    print("\nComputing class weights from training data...")

    # Infer num_classes if not provided
    if num_classes is None:
        if class_names is not None:
            num_classes = len(class_names)
        else:
            # try to infer from dataset labels (best-effort)
            labels_attr = getattr(train_loader.dataset, "labels", None)
            if isinstance(labels_attr, pd.DataFrame):
                num_classes = int(labels_attr.iloc[:, 1].nunique())
            elif isinstance(labels_attr, (pd.Series, list, tuple, np.ndarray)):
                num_classes = int(pd.Series(labels_attr).nunique())
            else:
                raise ValueError("Unable to infer num_classes; please pass it explicitly.")

    class_counts = torch.zeros(int(num_classes), dtype=torch.float64)

    for _, labels in tqdm(train_loader, desc="Counting classes"):
        # labels may be a tensor, numpy array, list, or pandas Series
        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        elif isinstance(labels, (pd.Series, list, tuple, np.ndarray)):
            labels_np = np.array(labels)
        else:
            labels_np = np.array(labels)

        for l in labels_np:
            idx = int(l)
            if 0 <= idx < num_classes:
                class_counts[idx] += 1

    total_samples = class_counts.sum()
    # avoid division by zero
    eps = 1e-9
    class_weights = total_samples / (num_classes * (class_counts + eps))
    # normalize so that sum(weights) == num_classes (keeps similar scale)
    class_weights = class_weights / class_weights.sum() * num_classes

    print("\nClass Distribution in Training Set:")
    print("-" * 60)
    names = class_names if class_names is not None else [str(i) for i in range(num_classes)]
    for i, (name, count, weight) in enumerate(zip(names, class_counts, class_weights)):
        pct = (count / total_samples * 100) if total_samples > 0 else 0.0
        print(f"{name:10s}: {int(count):6d} samples ({pct:5.2f}%) | Weight: {float(weight):.4f}")
    print("-" * 60)

    return class_weights.float()

# ==================== Device ====================
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


# ==================== Transforms ====================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==================== Data ====================
train_dataset = Four4All(csv_file='rafdb/train_labels.csv',
                         img_dir='rafdb/train', transform=transform)
val_dataset = Four4All(csv_file='rafdb/valid_labels.csv',
                       img_dir='rafdb/valid', transform=transform)
test_dataset = Four4All(csv_file='rafdb/test_labels.csv',
                        img_dir='rafdb/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Determine class names robustly from the dataset labels.
# `Four4All.labels` is a pandas DataFrame (see `get_dataset.py`),
# so avoid trying to convert the whole DataFrame to a torch tensor.
if hasattr(train_dataset, "class_names"):
    class_names = train_dataset.class_names
else:
    labels_attr = train_dataset.labels
    # If labels_attr is a DataFrame, assume the label column is the second column
    if isinstance(labels_attr, pd.DataFrame):
        labels_array = labels_attr.iloc[:, 1].to_numpy()
    elif isinstance(labels_attr, pd.Series):
        labels_array = labels_attr.to_numpy()
    else:
        labels_array = np.array(labels_attr)

    # Try to coerce to integers when possible (common for numeric labels)
    try:
        labels_array = labels_array.astype(int)
    except Exception:
        pass

    unique_labels = np.unique(labels_array)
    unique_sorted = np.sort(unique_labels)
    class_names = [str(u) for u in unique_sorted]

train_image, train_label = next(iter(train_loader))
val_image, val_label = next(iter(val_loader))
test_image, test_label = next(iter(test_loader))

print(f"Train batch: Image shape {train_image.shape}, Label shape {train_label.shape}")
print(f"Validation batch: Image shape {val_image.shape}, Label shape {val_label.shape}")
print(f"Test batch: Image shape {test_image.shape}, Label shape {test_label.shape}")


# ==================== Model, Loss, Optimizer ====================
model = ResEmoteNet().to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')

# Compute class weights from the training loader and create weighted loss
num_classes = len(class_names)
class_weights = compute_class_weights(train_loader, num_classes=num_classes, class_names=class_names)
class_weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

patience = 15
num_epochs = 80


# ==================== Utilities ====================
def evaluate_loader(model, loader, criterion):
    """Return average loss, accuracy, preds, labels for a loader."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc, all_preds, all_labels


def plot_training_curves(train_losses, val_losses, test_losses,
                         train_accs, val_accs, test_accs, save_path="rafdb/training_curves.png"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Val")
    plt.plot(epochs, test_losses, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train")
    plt.plot(epochs, val_accs, label="Val")
    plt.plot(epochs, test_accs, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix_percent(y_true, y_pred, class_names, save_path="rafdb/confusion_matrix_percent.png"):

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    plt.title("Test Confusion Matrix (%)")
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Confusion matrix saved to {save_path}")


    """Plot confusion matrix in percentages per row (true class)."""
    cm = confusion_matrix(y_true, y_pred)  # [web:35][web:41]
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_percent = np.nan_to_num(cm_percent)  # handle division by zero if any

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix (row-normalized %)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix (percent) saved to {save_path}")


    



def classification_report_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, F1 (macro) for multi-class FER."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)  # [web:34][web:40]
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return acc, prec, rec, f1


# ==================== Training Loop ====================
train_losses, val_losses, test_losses = [], [], []
train_accuracies, val_accuracies, test_accuracies = [], [], []

best_val_acc = 0.0
patience_counter = 0
epoch_counter = 0
best_model_path = "best_model.pth"

print("\n" + "=" * 70)
print("Starting Training")
print("=" * 70)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Test set evaluation (per epoch, as in original)
    test_loss, test_acc, _, _ = evaluate_loader(model, test_loader, criterion)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    # Validation evaluation
    val_loss, val_acc, _, _ = evaluate_loader(model, val_loader, criterion)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    epoch_counter += 1
    print(f"Epoch {epoch + 1:3d} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    # Early stopping on val_acc
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        print(f"No improvement in validation accuracy for {patience_counter} epochs.")

    if patience_counter > patience:
        print("Early stopping triggered.")
        break

# Save metrics CSV
df = pd.DataFrame({
    'Epoch': range(1, epoch_counter + 1),
    'Train Loss': train_losses,
    'Test Loss': test_losses,
    'Validation Loss': val_losses,
    'Train Accuracy': train_accuracies,
    'Test Accuracy': test_accuracies,
    'Validation Accuracy': val_accuracies
})
df.to_csv('result_four4all.csv', index=False)
print("Epoch metrics saved to result_four4all.csv")

# Plot training curves
plot_training_curves(
    train_losses, val_losses, test_losses,
    train_accuracies, val_accuracies, test_accuracies,
    save_path="rafdb/training_curves_rafdb.png"
)


# ==================== Final Test Evaluation Function ====================
def final_test_evaluation(model, test_loader, criterion, class_names):
    """Load best model and compute metrics, confusion matrix, etc."""
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION (best model)")
    print("=" * 70)

    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    test_loss = running_loss / len(test_loader)

    acc, prec, rec, f1 = classification_report_metrics(all_labels, all_preds)

    print(f"Test loss:   {test_loss:.4f}")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"F1-score:    {f1:.4f}")

    plot_confusion_matrix_percent(
        all_labels, all_preds, class_names,
        save_path="rafdb/confusion_matrix_rafdb.png"
    )


# ==================== Load Best Model and Run Final Evaluation ====================
best_model = ResEmoteNet().to(device)
best_model.load_state_dict(torch.load(best_model_path, map_location=device))

final_test_evaluation(best_model, test_loader, criterion, class_names)
