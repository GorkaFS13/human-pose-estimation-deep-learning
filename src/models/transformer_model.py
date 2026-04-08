import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN ---
path_proyecto = './'
path_imagenes = './imagenes_pro/' 
if not os.path.exists(path_imagenes): os.makedirs(path_imagenes)

path_dataset = os.path.join(path_proyecto, 'dataset2') 
images_dir = os.path.join(path_dataset, 'images')
labels_dir = os.path.join(path_dataset, 'labels')
scales_dir = os.path.join(path_dataset, 'labels2')

training_images = os.path.join(images_dir, 'train')
validation_images = os.path.join(images_dir, 'val')
training_labels = os.path.join(labels_dir, 'train')
validation_labels = os.path.join(labels_dir, 'val')
training_scales = os.path.join(scales_dir, 'train')
validation_scales = os.path.join(scales_dir, 'val')

path_pretrained_cnn = os.path.join(path_proyecto, 'best_resnet_pose_pck2.pth')
path_save_transformer = os.path.join(path_proyecto, 'best_tokenpose_pro.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. MÉTRICA PCKh@0.3 ---
def calcular_pckh_manual(preds, gts, head_sizes, masks, threshold=0.3):
    distances = np.linalg.norm(preds - gts, axis=-1)
    error_thresholds = threshold * head_sizes
    correct_keypoints = distances <= error_thresholds
    valid_correct = correct_keypoints & masks
    visible_points = np.sum(masks)
    if visible_points == 0: return 0.0
    return np.sum(valid_correct) / visible_points

# --- 3. DATASET HIGH RES (112x112) ---
class MPIIPoseDataset(Dataset):
    # CAMBIO 1: Sigma subido a 3 para que el punto sea más visible en 112x112
    def __init__(self, image_dir, label_dir, scale_dir, target_size=(224, 224), heatmap_size=(112, 112), sigma=3, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.scale_dir = scale_dir
        self.transform = transform 
        self.target_size = target_size
        self.heatmap_size = heatmap_size 
        self.sigma = sigma
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        w_orig, h_orig = img.size

        txt_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(self.label_dir, txt_name)
        scale_path = os.path.join(self.scale_dir, txt_name)
        
        keypoints = np.zeros((3, 2), dtype=np.float32)
        visible_mask = np.zeros(3, dtype=bool) 
        head_size_original = h_orig * 0.5 
        
        try:
            if os.path.exists(label_path):
                labels = np.loadtxt(label_path, dtype=np.float32)
                if labels.ndim == 1: labels = labels.reshape(1, -1)
                raw_kps = labels[:, 1:3] 
                limit = min(len(raw_kps), 3)
                is_normalized = np.max(raw_kps) <= 1.0
                
                if is_normalized:
                    keypoints[:limit, 0] = raw_kps[:limit, 0] * self.target_size[0]
                    keypoints[:limit, 1] = raw_kps[:limit, 1] * self.target_size[1]
                else:
                    scale_x = self.target_size[0] / w_orig
                    scale_y = self.target_size[1] / h_orig
                    keypoints[:limit, 0] = raw_kps[:limit, 0] * scale_x
                    keypoints[:limit, 1] = raw_kps[:limit, 1] * scale_y
                visible_mask[:limit] = True
        except: pass

        try:
            if os.path.exists(scale_path):
                with open(scale_path, 'r') as f:
                    content = f.read().strip()
                scale_val = float(content)
                head_size_original = scale_val * 200.0
        except: pass

        img = img.resize(self.target_size)
        scale_y_global = self.target_size[1] / h_orig
        head_size_final = head_size_original * scale_y_global
        
        heatmaps_gt = self.generate_heatmaps(keypoints, visible_mask)

        if self.transform:
            img = self.transform(img)

        return img, torch.from_numpy(heatmaps_gt), torch.from_numpy(keypoints), torch.from_numpy(visible_mask), head_size_final

    def generate_heatmaps(self, keypoints, visible_mask):
        heatmaps = np.zeros((3, *self.heatmap_size), dtype=np.float32)
        for i in range(3):
            if visible_mask[i]:
                x, y = keypoints[i]
                x = int(x * (self.heatmap_size[1] / self.target_size[0]))
                y = int(y * (self.heatmap_size[0] / self.target_size[1]))
                heatmaps[i] = self.gaussian_heatmap(x, y)
        return heatmaps

    def gaussian_heatmap(self, cx, cy):
        h, w = self.heatmap_size
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        if cx < 0 or cx >= w or cy < 0 or cy >= h: return np.zeros((h, w), dtype=np.float32)
        heatmap = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * self.sigma**2))
        return heatmap

transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 4. ARQUITECTURA HIGH-RES ---
class TokenPosePro(nn.Module):
    def __init__(self, num_keypoints=3, feature_dim=256, nhead=8, num_layers=4):
        super(TokenPosePro, self).__init__()
        
        resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        self.to_patch_embedding = nn.Conv2d(2048, feature_dim, kernel_size=1)
        self.keypoint_tokens = nn.Parameter(torch.zeros(1, num_keypoints, feature_dim))
        self.pos_embedding_patches = self.build_2d_sine_encoding(feature_dim, 7, 7)
        self.pos_embedding_tokens = nn.Parameter(torch.randn(1, num_keypoints, feature_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, dim_feedforward=1024, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder 112x112
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, num_keypoints, kernel_size=4, stride=2, padding=1)
        )
        self._init_weights()

    def build_2d_sine_encoding(self, dim, height, width):
        y_embed = torch.arange(height, dtype=torch.float32).view(-1, 1).repeat(1, width).unsqueeze(0)
        x_embed = torch.arange(width, dtype=torch.float32).view(1, -1).repeat(height, 1).unsqueeze(0)
        y_embed = y_embed / height
        x_embed = x_embed / width
        dim_t = torch.arange(dim // 2, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / (dim // 2))
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1).view(-1, dim).unsqueeze(0)
        return nn.Parameter(pos, requires_grad=False)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.keypoint_tokens)
        nn.init.trunc_normal_(self.pos_embedding_tokens, std=0.02)
        
    def forward(self, x):
        features = self.backbone(x)
        x_patches = self.to_patch_embedding(features).flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls_tokens = self.keypoint_tokens.expand(B, -1, -1)
        x_in = torch.cat((cls_tokens, x_patches), dim=1)
        x_in[:, :3] += self.pos_embedding_tokens
        x_in[:, 3:] += self.pos_embedding_patches
        x_out = self.transformer(x_in)
        
        visual_tokens = x_out[:, 3:, :] 
        feature_map = visual_tokens.transpose(1, 2).view(B, 256, 7, 7)
        heatmaps = self.deconv_layers(feature_map) 
        heatmaps = torch.sigmoid(heatmaps) 
        
        coords = self.heatmap_to_coord_refined(heatmaps)
        return heatmaps, coords

    def heatmap_to_coord_refined(self, heatmaps):
        B, K, H, W = heatmaps.shape
        heatmaps_flat = heatmaps.view(B, K, -1)
        idx = heatmaps_flat.argmax(dim=2)
        y = (idx // W).float()
        x = (idx % W).float()
        coords = torch.stack([x, y], dim=2)
        coords *= 2.0 
        return coords

def load_backbone_weights(model, path):
    print(f"\n>>> Cargando Backbone desde: {path}")
    if not os.path.exists(path):
        print("ERROR: No existe el modelo CNN. Verifica ruta.")
        return
    state_dict = torch.load(path)
    new_dict = {}
    for k, v in state_dict.items():
        if 'backbone' in k:
            new_key = k.replace('backbone.', '')
            new_dict[new_key] = v
    try:
        model.backbone.load_state_dict(new_dict, strict=False)
        print(">>> Pesos transferidos con éxito.")
    except: pass

def training_pro():
    print(f"\n>>> Entrenando TokenPose PRO (112x112) - FIX...")
    
    train_ds = MPIIPoseDataset(training_images, training_labels, training_scales, transform=transform)
    val_ds = MPIIPoseDataset(validation_images, validation_labels, validation_scales, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=12, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=12, shuffle=False, num_workers=2, pin_memory=True)

    model = TokenPosePro(num_keypoints=3).to(device)
    load_backbone_weights(model, path_pretrained_cnn)
    
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.transformer.parameters(), 'lr': 1e-4},
        {'params': model.deconv_layers.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    criterion = nn.MSELoss() 
    
    epochs = 100
    patience = 20 
    early_stop_cnt = 0
    best_acc = 0.0
    history = {'val_acc': [], 'train_loss': []}

    for epoch in range(epochs):
        model.train()
        run_loss = 0.0
        
        for imgs, heatmaps_gt, _, masks, _ in train_loader:
            imgs = imgs.to(device)
            heatmaps_gt = heatmaps_gt.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            heatmaps_pred, _ = model(imgs)
            
            mask_tensor = masks.float().view(-1, 3, 1, 1)
            
            # CAMBIO 2: Amplificación de la Loss (Gradient Boosting)
            # Multiplicamos por 1000 para que el error no sea 0.001
            loss = criterion(heatmaps_pred * mask_tensor, heatmaps_gt * mask_tensor) * 1000.0
            
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
        
        epoch_loss = run_loss / len(train_loader)

        model.eval()
        all_preds = []
        all_gts = []
        all_masks = []
        all_head_sizes = []
        
        with torch.no_grad():
            for imgs, _, kps_real, masks, head_sizes in val_loader:
                imgs = imgs.to(device)
                _, coords_pred = model(imgs)
                all_preds.append(coords_pred.cpu().numpy())
                all_gts.append(kps_real.numpy())
                all_masks.append(masks.numpy())
                all_head_sizes.append(head_sizes.numpy().reshape(-1, 1))

        current_acc = calcular_pckh_manual(
            preds=np.concatenate(all_preds),
            gts=np.concatenate(all_gts),
            head_sizes=np.concatenate(all_head_sizes),
            masks=np.concatenate(all_masks),
            threshold=0.3
        ) * 100

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f} | Val PCKh@0.3: {current_acc:.2f}%")
        
        history['val_acc'].append(current_acc)
        history['train_loss'].append(epoch_loss)

        if current_acc > best_acc:
            best_acc = current_acc
            early_stop_cnt = 0
            torch.save(model.state_dict(), path_save_transformer)
            print(f"  --> ¡Récord! {current_acc:.2f}%")
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= patience:
                print("Early stopping activado.")
                break

    plt.figure()
    plt.plot(history['val_acc'])
    plt.title('TokenPose PRO Accuracy')
    plt.savefig(os.path.join(path_imagenes, 'tokenpose_pro_fix.png'))
    print("¡Fin!")

if __name__ == '__main__':
    training_pro()