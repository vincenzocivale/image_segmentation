import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class SegmentationDataset(Dataset):
    def __init__(self, 
                 img_dir, 
                 mask_dir, 
                 class_rgb_values, 
                 transform=None, 
                 mask_transform=None):
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
        mask = Image.open(mask_path).convert('RGB')
        
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
    
    
    def convert_rgb_to_class(self, rgb_mask):
        """
        Converte una maschera RGB in una maschera di classi.

        Parametri:
            rgb_mask: numpy array o tensore di PyTorch, di forma (H, W, 3) o (3, H, W)

        Ritorna:
            class_mask: numpy array di forma (H, W) con indici di classe
        """
        # Se rgb_mask è un tensore PyTorch, convertirlo in NumPy
        if isinstance(rgb_mask, torch.Tensor):
            rgb_mask = rgb_mask.numpy()

        # Se rgb_mask ha la forma (3, H, W), trasporlo in (H, W, 3)
        if rgb_mask.shape[0] == 3 and len(rgb_mask.shape) == 3:
            rgb_mask = np.transpose(rgb_mask, (1, 2, 0))

        # Creazione della maschera delle classi
        class_mask = np.zeros((rgb_mask.shape[0], rgb_mask.shape[1]), dtype=np.uint8)

        # Itera sulle classi e assegna il valore corretto alla maschera
        for class_idx, rgb in self.class_rgb_values.items():
            rgb_array = np.array(rgb, dtype=np.uint8)  # Converte la tupla/lista in array

            # Confronta pixel per pixel
            mask = np.all(rgb_mask == rgb_array, axis=-1)

            # Assegna il valore della classe
            class_mask[mask] = class_idx

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


def prepare_dataset(img_dir, mask_dir, class_rgb_values, batch_size=8, test_size=0.2, val_size=0.1, num_workers=4):
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
        
    Ritorna:
        train_loader, val_loader, test_loader: Dataloader per train, validation e test
    """
    # Definisci le trasformazioni
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    
    # Carica tutti i dati
    full_dataset = SegmentationDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        class_rgb_values=class_rgb_values,
        transform=transform,
        mask_transform=mask_transform
    )
    
    # Ottieni tutti gli indici
    indices = list(range(len(full_dataset)))
    
    # Dividi in train, validation e test
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size/(1-test_size), random_state=42)
    
    print(f"Divisione dataset: {len(train_indices)} train, {len(val_indices)} validation, {len(test_indices)} test")
    
    # Crea subset
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
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