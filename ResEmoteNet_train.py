import os
import random
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

from ResEmoteNet import ResEmoteNet
from get_dataset import Four4All

# ------------------ 1. Reproducibility Block --------------------------
def set_seed(seed=10):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(10)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':
    # ------------------ 2. Device Selection --------------------------
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 
                          'cpu')
    print(f"Using {device} device")

    # ------------------ 3. Transforms and DataLoaders ---------------------
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

    train_dataset = Four4All(csv_file='rafdb/train_labels.csv',
                             img_dir='rafdb/train', transform=transform)
    val_dataset = Four4All(csv_file='rafdb/valid_labels.csv', 
                           img_dir='rafdb/valid/', transform=transform)
    test_dataset = Four4All(csv_file='rafdb/test_labels.csv', 
                            img_dir='rafdb/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                             num_workers=2, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True,
                           num_workers=2, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                            num_workers=2, worker_init_fn=seed_worker)

    train_image, train_label = next(iter(train_loader))
    val_image, val_label = next(iter(val_loader))
    test_image, test_label = next(iter(test_loader))

    print(f"Train batch: Image shape {train_image.shape}, Label shape {train_label.shape}")
    print(f"Validation batch: Image shape {val_image.shape}, Label shape {val_label.shape}")
    print(f"Test batch: Image shape {test_image.shape}, Label shape {test_label.shape}")

    # ------------------ 4. Model and Optimizer ---------------------------
    model = ResEmoteNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')

    # ----------- Compute class weights (as in the paper!) ------------------
    train_labels_for_weights = [label for _, label in train_dataset]
    class_counts = Counter(train_labels_for_weights)
    num_classes = len(class_counts)
    total = sum(class_counts.values())
    class_weights = [total / (num_classes * class_counts[i]) for i in range(num_classes)]
    class_weights = torch.FloatTensor(class_weights).to(device)
    print("Class weights:", class_weights)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6
    )

    patience = 15
    best_val_acc = 0
    patience_counter = 0
    epoch_counter = 0
    num_epochs = 80

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    test_losses, test_accuracies = [], []

    # ------------------ 5. Training Loop -----------------------------
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = data[0].to(device), data[1].to(device)
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

        model.eval()
        # ---- Test evaluation ----
        test_running_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_loss = test_running_loss / len(test_loader)
        test_acc = test_correct / test_total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # ---- Validation evaluation ----
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        epoch_counter += 1

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0 
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            print(f"No improvement in validation accuracy for {patience_counter} epochs.")

        if patience_counter > patience:
            print("Stopping early due to lack of improvement in validation accuracy.")
            break

    df = pd.DataFrame({
        'Epoch': range(1, epoch_counter+1),
        'Train Loss': train_losses,
        'Test Loss': test_losses,
        'Validation Loss': val_losses,
        'Train Accuracy': train_accuracies,
        'Test Accuracy': test_accuracies,
        'Validation Accuracy': val_accuracies
    })
    df.to_csv('result_four4all.csv', index=False)

    # ================ 6. Evaluation: Metrics & Confusion Matrix ==================

    def evaluate_model(model, data_loader, split_name="test", class_names=None, save_dir="."):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(f"\nResults for {split_name}:")
        print(f"  Accuracy:  {acc*100:.2f}%")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall:    {rec:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        print(f"  Classification Report:\n{report}")

        # Save confusion matrix
        plt.figure(figsize=(8,6))
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{split_name.capitalize()} Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{split_name}_confusion_matrix.png"), dpi=200)
        plt.close()

        # Save classification report to file
        with open(os.path.join(save_dir, f"{split_name}_classification_report.txt"), "w") as f:
            f.write(report)

        return acc, prec, rec, f1, cm

    # Replace with your emotion class ordering
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # Evaluate final/best model on test set
    print("\n--- Evaluating on TEST SET (best model) ---")
    model.load_state_dict(torch.load('best_model.pth'))
    test_metrics = evaluate_model(model, test_loader, split_name="test", class_names=class_names, save_dir=".")

    print("\n--- Evaluating on VALIDATION SET ---")
    val_metrics = evaluate_model(model, val_loader, split_name="val", class_names=class_names, save_dir=".")

    # ================ 7. Plotting Learning Curves ==================
    epochs_ran = range(1, epoch_counter+1)

    plt.figure(figsize=(16,5))
    plt.subplot(1,3,1)
    plt.plot(epochs_ran, train_losses, label="Train")
    plt.plot(epochs_ran, val_losses, label="Validation")
    plt.plot(epochs_ran, test_losses, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(epochs_ran, train_accuracies, label="Train")
    plt.plot(epochs_ran, val_accuracies, label="Validation")
    plt.plot(epochs_ran, test_accuracies, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    # Optionally plot F1 (add your own F1 tracking during loop for further plots)

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()
