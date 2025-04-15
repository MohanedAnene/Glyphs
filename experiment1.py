import torch
import torch.nn as nn
import torch.nn.functional as F
import src.machine_learning as ML

class GlyphClassifier(nn.Module):
    def __init__(self, NUM_classes, resolution):
        super(GlyphClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * resolution[0]//4 * resolution[1]//4, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_classes)
        )
 
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.classifier(x)
        return x
    
def train_model(model, train_loader, num_epochs=100, lr=0.001):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    return model, losses

import pandas as pd


results = []

dataset_file = 'data/random-stars-L.zip'

for nclasses in [10, 20, 50, 80, 100, 150]:
    for res in [32, 48, 64, 128, 256]:
        print(f"\n=== Training for {nclasses} classes at {res}x{res} resolution ===")

        train_dataset = ML.GlyphDataset(dataset_file, split='train', resize=(res, res), num_classes=nclasses)
        test_dataset = ML.GlyphDataset(dataset_file, split='test', resize=(res, res), num_classes=nclasses)

        train_loader = ML.create_loader(train_dataset, batch_size=64, shuffle=True)
        test_loader = ML.create_loader(test_dataset, batch_size=64, shuffle=False)

        model = GlyphClassifier(nclasses, resolution=(res, res))
        trained_model, loss_history = train_model(model, train_loader, num_epochs=100, lr=0.001)

        # === Evaluation ===
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trained_model.eval()
        all_true_labels = []
        all_predicted_labels = []
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = trained_model(images)
                _, predicted_labels = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted_labels == labels).sum().item()
                all_true_labels.extend(labels.cpu().numpy())
                all_predicted_labels.extend(predicted_labels.cpu().numpy())

        accuracy = 100 * correct_predictions / total_samples
        print(f"\n--- Evaluation Results for {nclasses} classes at {res}x{res} ---")
        print(f"Total Test Samples: {total_samples}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Accuracy on the test set: {accuracy:.2f}%")
        print(f"-------------------------------------------------------------")

        # Store in results list
        results.append({
            'num_classes': nclasses,
            'resolution': (res, res),
            'test_samples': total_samples,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'true_labels': all_true_labels,
            'predicted_labels': all_predicted_labels
        })

# Create dataframe
results_df = pd.DataFrame(results)

# Save to CSV (without labels lists) for quick overview
results_df.drop(['true_labels', 'predicted_labels'], axis=1).to_csv('experiment_summary.csv', index=False)

# Save the full object with labels for deeper analysis
results_df.to_pickle('experiment_results.pkl')
