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
                 augmentation=True,
                 img_size=(224, 224)):  # Aggiungo dimensione target fissa
        """
        Dataloader per dataset di segmentazione semantica
        
        Parametri:
            img_dir (str): Percorso alla directory con le immagini raw
            mask_dir (str): Percorso alla directory con le maschere
            class_rgb_values (dict): Dizionario che mappa classi a valori RGB 
            transform (callable, optional): Trasformazioni da applicare alle immagini
            mask_transform (callable, optional): Trasformazioni da applicare alle maschere
            augmentation (bool, optional): Se applicare data augmentation
            img_size (tuple): Dimensione target per le immagini finali (assicura batch consistenti)
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.class_rgb_values = class_rgb_values
        self.n_classes = len(class_rgb_values)
        self.augmentation = augmentation
        self.img_size = img_size
        
        # Ottieni la lista di tutti i file
        self.img_names = sorted([f for f in os.listdir(img_dir) if not f.startswith('.')])
        print(f"Caricate {len(self.img_names)} immagini")
        
        print(f"Dataset caricato: {len(self.img_names)} immagini con maschere corrispondenti")
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        # Carica l'immagine
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
            
        for class_idx in self.class_rgb_values.keys():
            mask_name = os.path.splitext(img_name)[0] + f'_class{class_idx}.png'
            mask_path = os.path.join(self.mask_dir, f"class_{class_idx}", mask_name)
            if os.path.exists(mask_path):
                class_mask = Image.open(mask_path).convert('L')
                class_mask = np.array(class_mask, dtype=np.uint8)
                mask[class_mask > 0] = class_idx

        if np.all(mask == 0):
            raise ValueError(f"Maschera vuota per l'immagine {img_name}")

        # Converti la maschera in immagine PIL per le trasformazioni
        mask_pil = Image.fromarray(mask)
        
        # Applica data augmentation se abilitata
        if self.augmentation:
            image, mask_pil = self.apply_augmentation(image, mask_pil)
        
        # Assicurati che le immagini abbiano SEMPRE la stessa dimensione alla fine 
        # (importante per risolvere il problema del batch)
        image = TF.resize(image, self.img_size)
        mask_pil = TF.resize(mask_pil, self.img_size, interpolation=TF.InterpolationMode.NEAREST)
        
        # Applica le trasformazioni standard dopo l'augmentation e il resize
        if self.transform:
            image = self.transform(image)
        else:
            # Se non ci sono trasformazioni, converti comunque in tensor
            image = TF.to_tensor(image)
        
        # Converti la maschera da PIL a tensor
        if self.mask_transform:
            mask = self.mask_transform(mask_pil)
        else:
            mask = torch.tensor(np.array(mask_pil), dtype=torch.long)

        return {
            'image': image,
            'mask': mask,
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
        class_mask = np.zeros((height, width), dtype=np.int64)
        
        # Crea un dizionario inverso per trovare l'ID classe dato un valore RGB
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
            sample = self[idx]
            mask = sample['mask'].numpy()
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
        # Resize iniziale per uniformare le dimensioni di partenza
        resize_size = (600, 600)  # Dimensione arbitraria più grande della dimensione finale
        image = TF.resize(image, resize_size)
        mask = TF.resize(mask, resize_size, interpolation=TF.InterpolationMode.NEAREST)
        
        # Random crop - assicura che tutte le immagini abbiano la stessa dimensione dopo il crop
        if random.random() > 0.3:  # 70% di probabilità di applicare il crop
            crop_size = (400, 400)  # Una dimensione fissa per tutti i crop
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
        
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random vertical flipping
        if random.random() > 0.7:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            
        # Random rotation - usiamo padded rotation per mantenere la dimensione originale
        if random.random() > 0.7:
            angle = random.randint(-15, 15)
            image = TF.rotate(image, angle, expand=False)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, expand=False)
        
        # Random Brightness adjustment (solo per l'immagine)
        if random.random() > 0.7:
            brightness_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)
            
        # Random Contrast adjustment (solo per l'immagine)
        if random.random() > 0.7:
            contrast_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_contrast(image, contrast_factor)
            
        # Random Color Jitter (solo per l'immagine)
        if random.random() > 0.7:
            saturation_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_saturation(image, saturation_factor)
        
        return image, mask


def prepare_dataset(img_dir, mask_dir, class_rgb_values, batch_size=8, test_size=0.2, val_size=0.1, num_workers=4, img_size=(224, 224)):
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
        img_size (tuple): Dimensione target per le immagini
    """
    # Definisci le trasformazioni base (normalizzazione)
    transform = transforms.Compose([
        # Rimuovo il resize da qui poiché lo facciamo già nel dataset
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Carica tutti i dati con la dimensione dell'immagine fissa
    full_dataset = SegmentationDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        class_rgb_values=class_rgb_values,
        transform=transform,
        mask_transform=None,
        augmentation=False,
        img_size=img_size  # Passa la dimensione target
    )
    
    # Ottieni tutti gli indici
    indices = list(range(len(full_dataset)))
    
    # Dividi in train, validation e test
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size/(1-test_size), random_state=42)
    
    print(f"Divisione dataset: {len(train_indices)} train, {len(val_indices)} validation, {len(test_indices)} test")
    
    # Crea dataset specializzati con augmentation solo per il training
    train_dataset = SegmentationDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        class_rgb_values=class_rgb_values,
        transform=transform,
        mask_transform=None,
        augmentation=True,
        img_size=img_size  # Dimensione fissa
    )
    
    val_dataset = SegmentationDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        class_rgb_values=class_rgb_values,
        transform=transform,
        mask_transform=None,
        augmentation=False,
        img_size=img_size  # Dimensione fissa
    )
    
    test_dataset = SegmentationDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        class_rgb_values=class_rgb_values,
        transform=transform,
        mask_transform=None,
        augmentation=False,
        img_size=img_size  # Dimensione fissa
    )
    
    # Crea subset
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Per sicurezza, utilizziamo un custom collate_fn per verificare che tutti gli elementi nel batch abbiano la stessa dimensione
    def my_collate(batch):
        images = []
        masks = []
        names = []
        for item in batch:
            images.append(item['image'])
            masks.append(item['mask'])
            names.append(item['image_name'])
        
        # Verifica dimensioni omogenee prima di creare il batch
        image_shapes = [img.shape for img in images]
        mask_shapes = [msk.shape for msk in masks]
        
        if len(set(image_shapes)) > 1 or len(set(mask_shapes)) > 1:
            print(f"ATTENZIONE: Batch con dimensioni non omogenee: {image_shapes} e {mask_shapes}")
        
        # Crea il batch
        images_batch = torch.stack(images)
        masks_batch = torch.stack(masks)
        
        return {
            'image': images_batch,
            'mask': masks_batch,
            'image_name': names
        }
    
    # Crea dataloader con il collate_fn personalizzato
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, collate_fn=my_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, collate_fn=my_collate)
    
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
    if isinstance(dataset, torch.utils.data.Subset):
        n_classes = dataset.dataset.n_classes
        class_rgb_values = dataset.dataset.class_rgb_values
    else:
        n_classes = dataset.n_classes
        class_rgb_values = dataset.class_rgb_values
    
    cmap = np.zeros((n_classes, 3), dtype=np.uint8)
    for class_idx, rgb in class_rgb_values.items():
        cmap[class_idx] = rgb
    
    # Applica la mappa colori
    mask_np = mask.numpy()
    colored_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    for h in range(mask_np.shape[0]):
        for w in range(mask_np.shape[1]):
            colored_mask[h, w] = cmap[mask_np[h, w]]
    
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


def visualize_augmentation(dataset, idx=0, num_samples=5):
    """
    Visualizza diverse versioni di data augmentation per un campione specifico
    
    Parametri:
        dataset: Dataset di segmentazione (assicurati che augmentation=True)
        idx (int): Indice del campione da visualizzare
        num_samples (int): Numero di variazioni da mostrare
    """
    plt.figure(figsize=(15, num_samples * 6))
    
    for i in range(num_samples):
        sample = dataset[idx]
        image = sample['image']
        mask = sample['mask']
        
        # De-normalizza l'immagine
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
        
        # Ottieni il dataset base se è un subset
        if isinstance(dataset, torch.utils.data.Subset):
            base_dataset = dataset.dataset
        else:
            base_dataset = dataset
            
        # Crea mappa colori per la visualizzazione della maschera
        cmap = np.zeros((base_dataset.n_classes, 3), dtype=np.uint8)
        for class_idx, rgb in base_dataset.class_rgb_values.items():
            cmap[class_idx] = rgb
        
        # Applica la mappa colori
        mask_np = mask.numpy()
        colored_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
        for h in range(mask_np.shape[0]):
            for w in range(mask_np.shape[1]):
                colored_mask[h, w] = cmap[mask_np[h, w]]
        
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(image)
        plt.title(f'Variazione {i+1} - Immagine')
        plt.axis('off')
        
        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(colored_mask)
        plt.title(f'Variazione {i+1} - Maschera')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()