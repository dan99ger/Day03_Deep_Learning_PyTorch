import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# 1. تحديد الجهاز (GPU/CUDA إن وجد، وإلا CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"جهاز التدريب المستخدم: {device}")

# 2. إنشاء فئة Dataset مخصصة
class CustomMLDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # تحويل إلى [N, 1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. بناء الشبكة العصبية الاصطناعية (ANN Architecture)
class CustomClassifier(nn.Module):
    def __init__(self, input_dim):
        super(CustomClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1) # مخرج واحد للتصنيف الثنائي Binary Classification
        )

    def forward(self, x):
        return self.net(x)

def main():
    # تحميل البيانات المعالجة من اليوم الأول
    data_path = '../Day01_Data_Preprocessing/processed_data.csv'
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['Target']).values
    y = df['Target'].values

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # تجهيز Datasets & DataLoaders
    train_dataset = CustomMLDataset(X_train, y_train)
    test_dataset = CustomMLDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # تهيئة النموذج، دالة الخسارة، والمحسن (Optimizer)
    input_dim = X_train.shape[1]
    model = CustomClassifier(input_dim).to(device)
    
    criterion = nn.BCEWithLogitsLoss()  # دالة خسارة ممتازة للتصنيف الثنائي المستقر عديًا
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

    # 4. حلقة التدريب والاختبار (Training Loop)
    epochs = 30
    train_losses, val_losses = [], []

    print("\n--- بدء حلقة التدريب (Training Loop) ---")
    for epoch in range(1, epochs + 1):
        # وضع التدريب
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # صفر المحاذيات (Zero Gradients)
            optimizer.zero_grad()
            
            # التمرير الأمامي (Forward Pass)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # التمرير الخلفي (Backward Pass)
            loss.backward()
            
            # تحديث الأوزان
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # وضع التقييم (Validation)
        model.eval()
        val_loss = 0.0
        with torch.no_grad(): # إيقاف حساب التدرجات لتوفير الذاكرة والسرعة
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        epoch_val_loss = val_loss / len(test_loader.dataset)
        val_losses.append(epoch_val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    # 5. التقييم النهائي
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    print(f"\n--- نتائج التقييم النهائي ---")
    print(f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

    # 6. حفظ أوزان النموذج رسم منحنيات التعلم
    os.makedirs('artifacts', exist_ok=True)
    torch.save(model.state_dict(), 'artifacts/pytorch_ann_model.pth')
    print("تم حفظ أوزان النموذج في artifacts/pytorch_ann_model.pth")

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('PyTorch Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('artifacts/loss_curve.png')
    print("تم حفظ منحنى الخسارة في artifacts/loss_curve.png")

if __name__ == '__main__':
    main()