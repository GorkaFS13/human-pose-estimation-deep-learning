import torch
import numpy as np
import os
import cv2
import random
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# --- IMPORTACIONES DE TUS MODELOS ---
try:
    from modelo_cnn import PoseResNet
    from modelo_transformer import TokenPosePro
except ImportError:
    print("Aviso: No se encontraron 'modelo_cnn.py' o 'modelo_transformer.py'.")

# --- CONFIGURACIÓN GLOBAL ---
PATH_DATASET = './dataset2' 
IMAGES_VAL = os.path.join(PATH_DATASET, 'images', 'val') 
LABELS_VAL = os.path.join(PATH_DATASET, 'labels', 'val')

PATH_CNN = './best_resnet_pose_pck2.pth'
PATH_TRANS = './best_tokenpose_pro.pth'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clases para cuando pintamos el dataset crudo
CLASSES = {
    0: {'name': 'Cabeza',  'color': (255, 0, 0)},   # Rojo en RGB
    1: {'name': 'M.Dcha',  'color': (0, 255, 0)},   # Verde
    2: {'name': 'M.Izq',   'color': (0, 0, 255)}    # Azul
}

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==========================================
# FUNCIONES AUXILIARES COMPARTIDAS
# ==========================================
def obtener_ground_truth(img_name, w_orig, h_orig):
    """Extrae las coordenadas (x,y) reales del .txt"""
    txt_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
    label_path = os.path.join(LABELS_VAL, txt_name)
    gt_points = [] 
    if os.path.exists(label_path):
        try:
            labels = np.loadtxt(label_path, dtype=np.float32)
            if labels.ndim == 1: labels = labels.reshape(1, -1)
            raw_kps = labels[:, 1:3]
            is_normalized = np.max(raw_kps) <= 1.0
            
            for i in range(min(len(raw_kps), 3)):
                x_raw, y_raw = raw_kps[i]
                if is_normalized:
                    gt_points.append((int(x_raw * w_orig), int(y_raw * h_orig)))
                else:
                    gt_points.append((int(x_raw), int(y_raw)))
        except: pass
    return gt_points

def unnormalize_bbox(yolo_line, img_w, img_h):
    """Convierte bounding box YOLO a pixeles reales"""
    parts = yolo_line.strip().split()
    class_id = int(parts[0])
    ncx, ncy, nw, nh = map(float, parts[1:])
    w, h = nw * img_w, nh * img_h
    cx, cy = ncx * img_w, ncy * img_h
    return class_id, int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)

def cargar_modelos():
    """Carga ambos modelos en memoria"""
    cnn = PoseResNet(num_keypoints=3).to(DEVICE)
    trans = TokenPosePro(num_keypoints=3).to(DEVICE)
    try:
        cnn.load_state_dict(torch.load(PATH_CNN, map_location=DEVICE))
        trans.load_state_dict(torch.load(PATH_TRANS, map_location=DEVICE))
        cnn.eval()
        trans.eval()
        return cnn, trans
    except Exception as e:
        print(f"Error cargando pesos: {e}")
        return None, None

# ==========================================
# FUNCIÓN 1: VISUALIZAR DATASET CRUDO 
# (Sustituye a visualize_keypoints.py)
# ==========================================
def visualizar_dataset_crudo(num_images=3):
    print(f"--- Verificando Anotaciones del Dataset ({num_images} imgs) ---")
    all_images = [f for f in os.listdir(IMAGES_VAL) if f.endswith(('.jpg', '.png'))]
    samples = random.sample(all_images, min(len(all_images), num_images))
    
    plt.figure(figsize=(15, 5))
    for i, img_name in enumerate(samples):
        img_path = os.path.join(IMAGES_VAL, img_name)
        txt_path = os.path.join(LABELS_VAL, img_name.replace('.jpg', '.txt'))
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_img, w_img = img.shape[:2]
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    cls_id, x1, y1, x2, y2 = unnormalize_bbox(line, w_img, h_img)
                    color = CLASSES.get(cls_id, {'color': (255, 255, 255)})['color']
                    name = CLASSES.get(cls_id, {'name': '?'})['name']
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(f"Dataset GT: {img_name}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==========================================
# FUNCIÓN 2: VISUALIZAR SOLO CNN
# (Sustituye a vis_def.py y visualize_modelpredict.py)
# ==========================================
def visualizar_cnn(num_images=3):
    print("--- Resultados CNN (Regresión) ---")
    cnn, _ = cargar_modelos()
    if not cnn: return

    all_images = [f for f in os.listdir(IMAGES_VAL) if f.endswith(('.jpg', '.png'))]
    samples = random.sample(all_images, min(len(all_images), num_images))
    
    plt.figure(figsize=(12, 4 * num_images))
    with torch.no_grad():
        for i, img_name in enumerate(samples):
            full_path = os.path.join(IMAGES_VAL, img_name)
            img_cv = cv2.cvtColor(cv2.imread(full_path), cv2.COLOR_BGR2RGB)
            h_orig, w_orig = img_cv.shape[:2]
            
            img_pil = Image.open(full_path).convert("RGB")
            input_tensor = val_transform(img_pil).unsqueeze(0).to(DEVICE)
            
            # Inferencia
            out = cnn(input_tensor).cpu().numpy().reshape(-1, 2)
            preds = [(int(x * (w_orig/224)), int(y * (h_orig/224))) for x, y in out]
            gt_points = obtener_ground_truth(img_name, w_orig, h_orig)
            
            for k in range(3):
                cv2.drawMarker(img_cv, preds[k], (255, 0, 0), cv2.MARKER_CROSS, 20, 3) # Rojo = Pred
                if k < len(gt_points):
                    cv2.circle(img_cv, gt_points[k], 6, (0, 255, 0), -1) # Verde = GT
                    cv2.line(img_cv, preds[k], gt_points[k], (255, 255, 0), 1) # Amarillo = Error
            
            plt.subplot(num_images, 1, i + 1)
            plt.imshow(img_cv)
            plt.title(f"{img_name} | Verde: Real - Rojo: Predicción CNN")
            plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==========================================
# FUNCIÓN 3: VISUALIZAR SOLO TRANSFORMER
# (Sustituye a vis_trans.py y vis_transFinal.py)
# ==========================================
def visualizar_transformer(num_images=3):
    print("--- Resultados Transformer (Heatmaps) ---")
    _, trans = cargar_modelos()
    if not trans: return

    all_images = [f for f in os.listdir(IMAGES_VAL) if f.endswith(('.jpg', '.png'))]
    samples = random.sample(all_images, min(len(all_images), num_images))
    
    plt.figure(figsize=(10, 4 * num_images))
    with torch.no_grad():
        for i, img_name in enumerate(samples):
            full_path = os.path.join(IMAGES_VAL, img_name)
            img_cv = cv2.cvtColor(cv2.imread(full_path), cv2.COLOR_BGR2RGB)
            h_orig, w_orig = img_cv.shape[:2]
            
            img_pil = Image.open(full_path).convert("RGB").resize((224, 224))
            input_tensor = val_transform(img_pil).unsqueeze(0).to(DEVICE)
            
            # Inferencia
            heatmaps, coords = trans(input_tensor)
            coords = coords.cpu().numpy()[0]
            preds = [(int(x * (w_orig/224)), int(y * (h_orig/224))) for x, y in coords]
            gt_points = obtener_ground_truth(img_name, w_orig, h_orig)
            
            # Heatmap
            hm_head = heatmaps[0, 0].cpu().numpy()
            hm_head = cv2.applyColorMap(np.uint8(255 * cv2.resize(hm_head, (224, 224))), cv2.COLORMAP_JET)
            
            for k in range(3):
                cv2.drawMarker(img_cv, preds[k], (255, 0, 0), cv2.MARKER_CROSS, 20, 3)
                if k < len(gt_points):
                    cv2.circle(img_cv, gt_points[k], 8, (0, 255, 0), -1)

            plt.subplot(num_images, 2, i*2 + 1)
            plt.imshow(img_cv)
            plt.title(f"{img_name} | Pred Transformer")
            plt.axis('off')
            
            plt.subplot(num_images, 2, i*2 + 2)
            plt.imshow(hm_head)
            plt.title("Heatmap (Cabeza)")
            plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==========================================
# FUNCIÓN 4: MOSAICO COMPARATIVO FINAL
# ==========================================
def visualizar_comparativa(num_images=3):
    print("--- Mosaico Comparativo: CNN vs Transformer ---")
    cnn, trans = cargar_modelos()
    if not cnn or not trans: return

    all_images = [f for f in os.listdir(IMAGES_VAL) if f.endswith(('.jpg', '.png'))]
    samples = random.sample(all_images, min(len(all_images), num_images))

    plt.figure(figsize=(15, 5 * num_images))
    with torch.no_grad():
        for i, img_name in enumerate(samples):
            full_path = os.path.join(IMAGES_VAL, img_name)
            img_cv = cv2.cvtColor(cv2.imread(full_path), cv2.COLOR_BGR2RGB)
            h_orig, w_orig = img_cv.shape[:2]
            
            img_pil = Image.open(full_path).convert("RGB").resize((224, 224))
            input_tensor = val_transform(img_pil).unsqueeze(0).to(DEVICE)
            
            out_cnn = cnn(input_tensor).cpu().numpy().reshape(-1, 2)
            preds_cnn = [(int(x * (w_orig/224)), int(y * (h_orig/224))) for x, y in out_cnn]
            
            heatmaps_trans, out_trans = trans(input_tensor)
            preds_trans = [(int(x * (w_orig/224)), int(y * (h_orig/224))) for x, y in out_trans.cpu().numpy()[0]]
            
            hm_head = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmaps_trans[0, 0].cpu().numpy(), (224, 224))), cv2.COLORMAP_JET)
            gt_points = obtener_ground_truth(img_name, w_orig, h_orig)
            
            img_cnn = img_cv.copy()
            img_trans = img_cv.copy()
            for k in range(3):
                cv2.drawMarker(img_cnn, preds_cnn[k], (255, 0, 0), cv2.MARKER_CROSS, 20, 3)
                cv2.drawMarker(img_trans, preds_trans[k], (255, 0, 0), cv2.MARKER_CROSS, 20, 3)
                if k < len(gt_points):
                    cv2.circle(img_cnn, gt_points[k], 8, (0, 255, 0), -1)
                    cv2.circle(img_trans, gt_points[k], 8, (0, 255, 0), -1)

            plt.subplot(num_images, 3, i*3 + 1)
            plt.imshow(img_cnn)
            plt.title("CNN (Regresión)")
            plt.axis('off')
            
            plt.subplot(num_images, 3, i*3 + 2)
            plt.imshow(img_trans)
            plt.title("Transformer (Heatmaps)")
            plt.axis('off')

            plt.subplot(num_images, 3, i*3 + 3)
            plt.imshow(hm_head)
            plt.title("Heatmap Transformer")
            plt.axis('off')

    plt.tight_layout()
    plt.show()

# Si ejecutas el script directamente, lanza la comparativa por defecto
if __name__ == '__main__':
    visualizar_comparativa(3)