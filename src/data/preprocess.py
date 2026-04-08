import scipy.io
import os
import shutil
import cv2
import numpy as np
import random
import yaml
from tqdm import tqdm

# --- CONFIGURACIÓN ---
# Rutas (con r'' para Windows)
MAT_FILE = r'C:\Users\Gorka\Desktop\IA\APUNTES\3IA\AARN\PoseEstimation\data_raw\annotations\mpii_human_pose_v1_u12_1.mat'
SOURCE_IMG_DIR = r'C:\Users\Gorka\Desktop\IA\APUNTES\3IA\AARN\PoseEstimation\data_raw\images'

DATASET_DIR = 'dataset2' # Nombre actualizado

# --- PARÁMETROS DE DIVISIÓN ---
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1
# Nota: La suma debe ser 1.0

# --- PARÁMETROS DE EXTRACCIÓN ---
SINGLE_PERSON_ONLY = True  # True: Solo fotos con 1 persona
WRIST_BOX_SCALE_FACTOR = 35 
# -----------------------------

def setup_directories():
    """Crea la estructura de carpetas train/val/test."""
    if os.path.exists(DATASET_DIR):
        print(f"Aviso: '{DATASET_DIR}' ya existe. Se combinará/sobreescribirá...")
    
    # Ahora incluimos 'test'
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(DATASET_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, 'labels', split), exist_ok=True)

def normalize_yolo(cx, cy, w, h, img_w, img_h):
    norm_cx = max(0.0, min(1.0, cx / img_w))
    norm_cy = max(0.0, min(1.0, cy / img_h))
    norm_w = max(0.0, min(1.0, w / img_w))
    norm_h = max(0.0, min(1.0, h / img_h))
    return norm_cx, norm_cy, norm_w, norm_h

def get_bboxes(rect, img_width, img_height):
    yolo_lines = []

    # 1. HEAD (Clase 0)
    if (hasattr(rect, 'x1') and hasattr(rect, 'y1') and 
        hasattr(rect, 'x2') and hasattr(rect, 'y2')):
        try:
            x1 = float(rect.x1) if np.ndim(rect.x1) == 0 else float(rect.x1.item())
            y1 = float(rect.y1) if np.ndim(rect.y1) == 0 else float(rect.y1.item())
            x2 = float(rect.x2) if np.ndim(rect.x2) == 0 else float(rect.x2.item())
            y2 = float(rect.y2) if np.ndim(rect.y2) == 0 else float(rect.y2.item())
            
            w = x2 - x1
            h = y2 - y1
            cx = x1 + (w / 2)
            cy = y1 + (h / 2)
            
            ncx, ncy, nw, nh = normalize_yolo(cx, cy, w, h, img_width, img_height)
            yolo_lines.append(f"0 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
        except:
            pass 

    # 2. WRISTS (Clase 1 y 2)
    if not hasattr(rect, 'annopoints') or np.all(rect.annopoints == []):
        return yolo_lines 

    try:
        points = rect.annopoints.point
        if not isinstance(points, np.ndarray) and not isinstance(points, list):
            points = [points]
        elif isinstance(points, np.ndarray):
            if points.size == 1: points = [points.item()] 
            
        person_scale = float(rect.scale) if (hasattr(rect, 'scale') and np.ndim(rect.scale)==0) else 4.0
        box_size = person_scale * WRIST_BOX_SCALE_FACTOR

        for p in points:
            try:
                p_id = int(p.id)
                p_x = float(p.x)
                p_y = float(p.y)
                
                cls_id = -1
                if p_id == 10: cls_id = 1  # Right
                elif p_id == 15: cls_id = 2 # Left
                
                if cls_id != -1:
                    ncx, ncy, nw, nh = normalize_yolo(p_x, p_y, box_size, box_size, img_width, img_height)
                    yolo_lines.append(f"{cls_id} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")

            except AttributeError:
                continue 
    except Exception as e:
        pass 

    return yolo_lines

def create_yaml_config():
    """Genera el YAML incluyendo la ruta de test."""
    config = {
        'path': os.path.abspath(DATASET_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',  # Añadido test
        'nc': 3,
        'names': {0: 'head', 1: 'wrist_right', 2: 'wrist_left'}
    }
    
    yaml_path = os.path.join(DATASET_DIR, 'mpii_single_person.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    return yaml_path

def process_set(dataset, split_name):
    """Función auxiliar para copiar imágenes y etiquetas."""
    if not dataset:
        return
    
    for sample in tqdm(dataset, desc=f"Writing {split_name}"):
        # Copiar imagen
        dst_img_path = os.path.join(DATASET_DIR, 'images', split_name, sample['name'])
        shutil.copy2(sample['path'], dst_img_path)
        
        # Crear txt
        txt_name = sample['name'].replace('.jpg', '.txt').replace('.png', '.txt')
        dst_label_path = os.path.join(DATASET_DIR, 'labels', split_name, txt_name)
        with open(dst_label_path, 'w') as f:
            f.write('\n'.join(sample['labels']))

def main():
    setup_directories()
    
    print(f"Cargando {MAT_FILE}...")
    try:
        mat = scipy.io.loadmat(MAT_FILE, struct_as_record=False, squeeze_me=True)
        data = mat['RELEASE']
        annolist = data.annolist
        img_train_flags = data.img_train
    except Exception as e:
        print(f"Error cargando .mat: {e}")
        return

    valid_samples = []
    skipped_multigente = 0
    skipped_incomplete = 0 # Contador para fotos a las que les falta una mano/cabeza
    
    print("Procesando anotaciones...")
    for idx, annotation in enumerate(tqdm(annolist)):
        if img_train_flags[idx] == 0: continue 
            
        try:
            rects = annotation.annorect
            if not isinstance(rects, np.ndarray) and not isinstance(rects, list):
                rects = [rects]
            elif isinstance(rects, np.ndarray) and rects.size == 0:
                rects = [] 

            # --- 1. FILTRO SINGLE PERSON ---
            if SINGLE_PERSON_ONLY and len(rects) != 1:
                skipped_multigente += 1
                continue
            
            image_name = annotation.image.name
            full_img_path = os.path.join(SOURCE_IMG_DIR, image_name)
            
            if not os.path.exists(full_img_path): continue

            # Leer imagen
            img = cv2.imread(full_img_path)
            if img is None: continue
            h_img, w_img = img.shape[:2]

            image_labels = []
            for rect in rects:
                bboxes = get_bboxes(rect, w_img, h_img)
                image_labels.extend(bboxes)
            
            # --- 2. FILTRO "TENER LAS 3 PARTES" ---
            # Analizamos qué clases hemos encontrado en esta imagen
            found_classes = set()
            for line in image_labels:
                # La linea es "0 0.5 0.5 ...", cogemos el primer caracter (la clase)
                class_id = int(line.split()[0])
                found_classes.add(class_id)
            
            # Verificamos si tenemos la clase 0 (Head), 1 (R_Wrist) y 2 (L_Wrist)
            # set({0, 1, 2}) significa que buscamos esos tres IDs.
            if {0, 1, 2}.issubset(found_classes):
                valid_samples.append({
                    'name': image_name,
                    'path': full_img_path,
                    'labels': image_labels
                })
            else:
                # Si falta alguna parte, la contamos como incompleta y NO la añadimos
                skipped_incomplete += 1
                
        except Exception as e:
            continue

    total_samples = len(valid_samples)
    print(f"\nResultados del Filtrado:")
    print(f"  - Imágenes descartadas (Multitud/Vacías): {skipped_multigente}")
    print(f"  - Imágenes descartadas (Faltan manos/cabeza): {skipped_incomplete}")
    print(f"  - TOTAL IMÁGENES VÁLIDAS (Completas): {total_samples}")
    
    if total_samples == 0:
        print("ERROR: No han quedado imágenes. Revisa las rutas o relaja los filtros.")
        return

    # --- DIVISIÓN 80 / 10 / 10 ---
    random.seed(42)
    random.shuffle(valid_samples)
    
    idx_train_end = int(total_samples * TRAIN_RATIO)
    idx_val_end = idx_train_end + int(total_samples * VAL_RATIO)
    
    train_set = valid_samples[:idx_train_end]
    val_set = valid_samples[idx_train_end:idx_val_end]
    test_set = valid_samples[idx_val_end:]
    
    print(f"\nDistribución Final:")
    print(f"  - Train: {len(train_set)}")
    print(f"  - Val:   {len(val_set)}")
    print(f"  - Test:  {len(test_set)}")
    print("-" * 30)

    process_set(train_set, 'train')
    process_set(val_set, 'val')
    process_set(test_set, 'test')

    yaml_path = create_yaml_config()
    print("\n¡Hecho!")
    print(f"YAML creado en: {yaml_path}")

if __name__ == "__main__":
    main()