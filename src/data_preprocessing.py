import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision.transforms import functional as TF
import random

class SegmentationDataset(Dataset):
    def __init__(self, 
                 img_dir, 
                 mask_dir, 
                 class_rgb_values, 
                 transform=None, 
                 mask_transform=None,
                 augmentation=False):
        """
        Dataloader per dataset di segmentazione semantica
        
        Parametri:
            img_dir (str): Percorso alla directory con le immagini raw
            mask_dir (str): Percorso alla directory con le maschere
            class_rgb_values (dict): Dizionario che mappa classi a valori RGB 
                                    es. {0: [0, 0, 0], 1: [255, 0, 0], ...}
            transform (callable, optional): Trasformazioni da applicare alle immagini
            mask_transform (callable, optional): Trasformazioni da applicare alle maschere
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.class_rgb_values = class_rgb_values
        self.n_classes = len(class_rgb_values)
        self.augmentation = augmentation
        
        # Ottieni la lista di tutti i file
        self.img_names = sorted([f for f in os.listdir(img_dir) if not f.startswith('.')])
        
        # Controlla che ci sia corrispondenza tra immagini e maschere
        self.valid_indices = []
        for i, img_name in enumerate(self.img_names):
            mask_name = os.path.splitext(img_name)[0] + '.png'  # Presuppone maschere con estensione .png
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                self.valid_indices.append(i)
        
        # Filtra solo i file validi
        self.img_names = [self.img_names[i] for i in self.valid_indices]
        
        print(f"Dataset caricato: {len(self.img_names)} immagini con maschere corrispondenti")
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        # Carica l'immagine
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Carica la maschera (presuppone stesso nome con estensione .png)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = np.array(Image.open(mask_path).convert('RGB'))
        
        # Applica trasformazioni
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = np.array(mask)
            
        # Converti la maschera RGB in mappa di classi
        class_mask = self.convert_rgb_to_class(mask)
        
        return {
            'image': image,
            'mask': torch.tensor(class_mask, dtype=torch.long),
            'image_name': img_name
        }
    
    


    def convert_rgb_to_class(self, image, one_hot=False, tensor_output=True):
        """
        Crea una maschera di segmentazione da un'immagine RGB dove ogni classe ha un colore specifico.
        
        Args:
            image: numpy.ndarray in formato RGB di shape (H, W, 3)
            one_hot: se True, restituisce una maschera one-hot encoded (H, W, num_classes)
                se False, restituisce una maschera a singolo canale (H, W) con indici di classe
            tensor_output: se True, converte l'output in tensor PyTorch
        
        Returns:
            Maschera di segmentazione nel formato richiesto
        """
        height, width, _ = image.shape
        num_classes = len(self.class_rgb_values)
        
        # Inizializza la maschera con valori di sfondo/classe 0
        # Se hai una classe di sfondo, assicurati che sia mappata correttamente
        class_mask = np.zeros((height, width), dtype=np.int64)
        
        # Crea un dizionario inverso per trovare l'ID classe dato un valore RGB
        # Converti le liste in tuple per renderle hashable
        rgb_to_class = {tuple(rgb): class_id for class_id, rgb in self.class_rgb_values.items()}
        
        # Riempi la maschera con l'ID classe appropriato per ogni pixel
        for h in range(height):
            for w in range(width):
                pixel_rgb = tuple(image[h, w])
                if pixel_rgb in rgb_to_class:
                    class_mask[h, w] = rgb_to_class[pixel_rgb]
        
        # Se richiesto one-hot encoding
        if one_hot:
            # Crea una maschera one-hot encoding
            mask_one_hot = np.zeros((height, width, num_classes), dtype=np.float32)
            for class_id in self.class_rgb_values.keys():
                mask_one_hot[:, :, class_id] = (class_mask == class_id).astype(np.float32)
            
            # Converti in tensor se richiesto
            if tensor_output:
                return torch.from_numpy(mask_one_hot).permute(2, 0, 1)  # (num_classes, H, W)
            return mask_one_hot
        
        # Restituisci la maschera a singolo canale
        if tensor_output:
            return torch.from_numpy(class_mask).long()  # (H, W) con valori long
        
        return class_mask

    def get_class_weight(self):
        """
        Calcola i pesi per bilanciare le classi
        
        Ritorna:
            class_weights: torch.tensor con pesi per bilanciare le classi
        """
        class_counts = np.zeros(self.n_classes)
        
        for idx in range(len(self)):
            mask = self[idx]['mask'].numpy()
            for c in range(self.n_classes):
                class_counts[c] += np.sum(mask == c)
        
        # Gestisce il caso in cui una classe non è presente
        class_counts = np.where(class_counts == 0, 1, class_counts)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / np.sum(class_weights) * self.n_classes
        
        return torch.tensor(class_weights, dtype=torch.float)

    
    def apply_augmentation(self, image, mask):
        """
        Applica trasformazioni di data augmentation sia all'immagine che alla maschera
        
        Parametri:
            image: Immagine PIL
            mask: Maschera PIL
            
        Ritorna:
            image, mask: Immagine e maschera trasformate
        """
        # Resize con la stessa dimensione target (prima di altre trasformazioni)
        resize_size = (256, 256)
        image = TF.resize(image, resize_size)
        mask = TF.resize(mask, resize_size, interpolation=TF.InterpolationMode.NEAREST)
        
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(224, 224))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random vertical flipping
        if random.random() > 0.7:  # Meno probabile del flip orizzontale
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            
        # Random rotation
        if random.random() > 0.7:
            angle = random.randint(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        
        return image, mask


def prepare_dataset(img_dir, mask_dir, class_rgb_values, batch_size=8, test_size=0.2, val_size=0.1, num_workers=4, augmentation=True):
    """
    Prepara il dataset dividendolo in train, validation e test
    
    Parametri:
        img_dir (str): Percorso alla directory con le immagini raw
        mask_dir (str): Percorso alla directory con le maschere
        class_rgb_values (dict): Dizionario che mappa classi a valori RGB
        batch_size (int): Dimensione del batch
        test_size (float): Frazione del dataset per il test
        val_size (float): Frazione del dataset per la validazione
        num_workers (int): Numero di worker per il dataloader
        augmentation (bool): Se applicare data augmentation al training set
        
    Ritorna:
        train_loader, val_loader, test_loader: Dataloader per train, validation e test
    """
    # Definisci le trasformazioni base (normalizzazione)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Non usiamo più trasformazioni specifiche per le maschere perché 
    # ora sono gestite all'interno del metodo apply_augmentation
    
    # Carica tutti i dati
    full_dataset = SegmentationDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        class_rgb_values=class_rgb_values,
        transform=transform,
        mask_transform=None,
        augmentation=False  # No augmentation per il dataset completo
    )
    
    # Ottieni tutti gli indici
    indices = list(range(len(full_dataset)))
    
    # Dividi in train, validation e test
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size/(1-test_size), random_state=42)
    
    print(f"Divisione dataset: {len(train_indices)} train, {len(val_indices)} validation, {len(test_indices)} test")
    
    # Crea dataset specializzati
    train_dataset = SegmentationDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        class_rgb_values=class_rgb_values,
        transform=transform,
        mask_transform=None,
        augmentation=augmentation  # Applica augmentation solo al training set
    )
    
    # Crea subset
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)  # Usa il dataset originale senza augmentation
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)  # Usa il dataset originale senza augmentation
    
    # Crea dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, full_dataset


def visualize_sample(dataset, idx=0):
    """
    Visualizza un campione dal dataset
    
    Parametri:
        dataset: Dataset di segmentazione
        idx (int): Indice del campione da visualizzare
    """
    sample = dataset[idx]
    image = sample['image']
    mask = sample['mask']
    
    # De-normalizza l'immagine
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = image.permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)
    
    # Crea mappa colori per la visualizzazione della maschera
    cmap = np.zeros((dataset.n_classes, 3), dtype=np.uint8)
    for class_idx, rgb in dataset.class_rgb_values.items():
        cmap[class_idx] = rgb
    
    # Applica la mappa colori
    colored_mask = cmap[mask]
    
    # Visualizza
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Immagine originale')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(colored_mask)
    plt.title('Maschera segmentazione')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()