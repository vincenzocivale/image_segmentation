import os
import numpy as np
from PIL import Image
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm  
import cv2
from torchvision import transforms
import torch

# Definisci le trasformazioni per le immagini (converti in tensor)
transform = transforms.Compose([
    transforms.ToTensor(),  # Converte l'immagine in un tensore
])

COLORS = np.array([
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
    [0, 170, 255],
    [255, 85, 255]
])


def convert_label_to_class(label_img):
    """
    Converte un'immagine label (PIL.Image) in una mappa con ID delle classi.
    """
    label_np = np.array(label_img)  # dimensione (H, W, 3)
    class_map = np.zeros((label_np.shape[0], label_np.shape[1]), dtype=np.int64)
    for class_id, color in enumerate(COLORS):
        # Crea una maschera per i pixel che corrispondono al colore
        mask = (label_np == color).all(axis=-1)
        class_map[mask] = class_id
    return class_map


def load_custom_dataset(data_dir):
    """
    Carica il dataset da una cartella contenente 'images/' e 'labels/'.
    Restituisce un oggetto HuggingFace Dataset.
    """
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    image_filenames = sorted(os.listdir(images_dir))
    label_filenames = sorted(os.listdir(labels_dir))
    
    data = {"image": [], "label": []}
    
    # Usa tqdm per visualizzare una barra di caricamento
    for img_name, lbl_name in tqdm(zip(image_filenames, label_filenames),
                               total=len(image_filenames),
                               desc="Caricamento dataset"):
        img_path = os.path.join(images_dir, img_name)
        lbl_path = os.path.join(labels_dir, lbl_name)

        # Carica l'immagine e la label con OpenCV
        image = cv2.imread(img_path).astype(np.uint8)
        label_img = cv2.imread(lbl_path).astype(np.uint8)

        # Converte la label in una mappa di classi
        label = convert_label_to_class(label_img)
        label = torch.tensor(label, dtype=torch.long)  # 🚀 Converti in tensor!

        image = transform(image)

        data["image"].append(image)
        data["label"].append(label)  # Ora anche le label sono tensori!

    
    data["image"] = data["image"][:100]
    data["label"] = data["label"][:100]
    # Crea il dataset HuggingFace
    dataset = Dataset.from_dict(data)
    return dataset


def splited_dataset(dataset):
    """
    Suddivide il dataset in train, validation e test.
    """
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # Seleziona le parti corrispondenti
    train_data = dataset.select(train_idx)
    test_data = dataset.select(test_idx)

    # Suddivide ulteriormente il training set per ottenere una validazione (es. 10% del training)
    train_indices = list(range(len(train_data)))
    train_idx, val_idx = train_test_split(train_indices, test_size=0.1, random_state=42)
    train_data_final = train_data.select(train_idx)
    val_data = train_data.select(val_idx)

    # Crea il DatasetDict
    dataset_dict = DatasetDict({
        "train": train_data_final,
        "validation": val_data,
        "test": test_data
    })

    return dataset_dict

