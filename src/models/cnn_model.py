import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import os
import sys
from PIL import Image
import optuna
import matplotlib.pyplot as plt


path_proyecto = './'
path_imagenes = './imagenes/'
path_dataset = os.path.join(path_proyecto, 'dataset2') 
images_dir = os.path.join(path_dataset, 'images')
labels_dir = os.path.join(path_dataset, 'labels')

training_images = os.path.join(images_dir, 'train')
validation_images = os.path.join(images_dir, 'val')
training_labels = os.path.join(labels_dir, 'train')
validation_labels = os.path.join(labels_dir, 'val')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def check_gpu():
    print("-" * 50)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("ERROR: No se ha encontrado GPU, estamos en CPU.")
    print("-" * 50)



class MPIIPoseDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(self.label_dir, label_name)

        try:
            if os.path.exists(label_path):
                labels = np.loadtxt(label_path, dtype=np.float32)
                if labels.ndim == 1: labels = labels.reshape(1, -1)
                keypoints = labels[:, 1:3].flatten()
                
                if len(keypoints) != 6:
                    temp = np.zeros(6, dtype=np.float32)
                    limit = min(len(keypoints), 6)
                    temp[:limit] = keypoints[:limit]
                    keypoints = temp
            else:
                keypoints = np.zeros(6, dtype=np.float32)
        except:
            keypoints = np.zeros(6, dtype=np.float32)

        if self.transform:
            img = self.transform(img)

        return img, torch.from_numpy(keypoints)

class PoseResNet(nn.Module):
    def __init__(self, num_keypoints=3):
        super(PoseResNet, self).__init__()
        self.backbone = models.resnet50(weights='DEFAULT') 
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_keypoints * 2) 

    def forward(self, x):
        return self.backbone(x)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def objective(trial):
    train_ds = MPIIPoseDataset(training_images, training_labels, transform)
    val_ds = MPIIPoseDataset(validation_images, validation_labels, transform)

    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = PoseResNet(num_keypoints=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    epochs_optuna = 4 

    for epoch in range(epochs_optuna):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            labels = labels.view(images.size(0), -1) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                labels = labels.view(images.size(0), -1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

##Training con early stopping
def training(study):
    print("\n>>> Iniciando el entrenamiento final...")
    
    full_train_ds = MPIIPoseDataset(training_images, training_labels, transform)
    full_val_ds = MPIIPoseDataset(validation_images, validation_labels, transform)

    best_params = study.best_params
    
    final_train_loader = DataLoader(full_train_ds, batch_size=best_params['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    final_val_loader = DataLoader(full_val_ds, batch_size=best_params['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    final_model = PoseResNet(num_keypoints=3).to(device)
    final_optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss()

    patience = 7               
    early_stopping_counter = 0 
    best_val_loss = float('inf') 
    path_to_save_model = os.path.join(path_proyecto, 'best_resnet_pose.pth')
    
    epochs = 50
    
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    for epoch in range(epochs):
        ##Train
        final_model.train()
        running_loss = 0.0
        
        
        for images, labels in final_train_loader:
            images, labels = images.to(device), labels.to(device)
            final_optimizer.zero_grad()
            outputs = final_model(images)
            labels = labels.view(images.size(0), -1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            final_optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        
        epoch_loss = running_loss / len(final_train_loader.dataset)
        
        # #Val
        final_model.eval()
        val_running_loss = 0.0
        
        
        with torch.no_grad():
            for images, labels in final_val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = final_model(images)
                labels = labels.view(images.size(0), -1)
                
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                
                
        epoch_val_loss = val_running_loss / len(final_val_loader.dataset)
        
        
        # Guardar historial
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(epoch_val_loss)
        
        
        scheduler.step(epoch_val_loss)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {epoch_loss:.4f}")
        print(f"  Val Loss:   {epoch_val_loss:.4f}")

        ##Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            early_stopping_counter = 0
            torch.save(final_model.state_dict(), path_to_save_model)
            print(f"  --> ¡Mejora! Modelo guardado en {path_to_save_model}")
        else:
            early_stopping_counter += 1
            print(f"  --> No mejora. Contador Early Stopping: {early_stopping_counter}/{patience}")
            
            if early_stopping_counter >= patience:
                print("\n>>> Early Stopping activado. Deteniendo entrenamiento.")
                break
        

    ##Gráficos
    print("\n>>> Generando y guardando gráficas...")
    plt.figure(figsize=(12, 5))

    # Loss
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss (MSE)')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)


    plt.tight_layout()
    save_plot_path = os.path.join(path_imagenes, 'graficas_entrenamiento.png')
    plt.savefig(save_plot_path)
    print(f"Gráficas guardadas en: {save_plot_path}")
    print("¡FIN!")


if __name__ == '__main__':
    check_gpu()
    
    if not os.path.exists(training_images):
        print(f"ERROR: No encuentro la ruta {training_images}")
        sys.exit()

    print("\n>>> Iniciando OPTUNA...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10) 

    print("\nMEJORES HIPERPARÁMETROS:", study.best_params)
    training(study=study)