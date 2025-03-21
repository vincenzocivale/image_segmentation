import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.nn.functional as F

# Imposta seed per riproducibilità
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Definizione del modello SegFormer
class SegFormerModel(nn.Module):
    def __init__(self, num_classes, id2label=None, pretrained=True):
        super(SegFormerModel, self).__init__()
        
        # Configura il mapping delle classi
        if id2label is None:
            id2label = {i: f"class_{i}" for i in range(num_classes)}
        label2id = {v: k for k, v in id2label.items()}
        
        # Inizializza il modello
        if pretrained:
            # Utilizza un modello pre-addestrato (SegFormer-B0 per default)
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512",
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True
            )
        else:
            # Inizializza un modello da zero
            config = SegformerConfig(
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
                # Puoi personalizzare la configurazione se necessario
                # hidden_sizes=[32, 64, 160, 256],
                # depths=[2, 2, 2, 2],
                # decoder_hidden_size=256,
            )
            self.segformer = SegformerForSemanticSegmentation(config)
    
    def forward(self, pixel_values):
        outputs = self.segformer(pixel_values=pixel_values)
        return outputs.logits

# Classe per la gestione del training
class SegmentationTrainer:
    def __init__(self, 
                 model, 
                 train_loader, 
                 val_loader,
                 test_loader=None,
                 device='cuda',
                 class_weights=None,
                 learning_rate=1e-4,
                 weight_decay=1e-4,
                 project_name='segmentation',
                 experiment_name='segformer'):
        """
        Trainer per modelli di segmentazione con integrazione Weights & Biases
        
        Parametri:
            model: Modello di segmentazione
            train_loader: Dataloader per il training
            val_loader: Dataloader per la validazione
            test_loader: Dataloader per il test
            device: Dispositivo su cui eseguire il training ('cuda' o 'cpu')
            class_weights: Pesi per le classi (per bilanciare dataset sbilanciati)
            learning_rate: Learning rate per l'ottimizzatore
            weight_decay: Weight decay per l'ottimizzatore
            project_name: Nome del progetto su W&B
            experiment_name: Nome dell'esperimento su W&B
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.class_weights = class_weights
        
        # Sposta il modello sul dispositivo
        self.model = self.model.to(self.device)
        
        # Definizione loss function
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)
            
        # Definizione ottimizzatore
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        # Nome del progetto e dell'esperimento
        self.project_name = project_name
        self.experiment_name = experiment_name
        
        # Metriche da tracciare
        self.best_val_iou = 0.0
        self.best_epoch = 0
        
    def train(self, epochs=100, save_dir='checkpoints', log_interval=10):
        """
        Esegue il training del modello
        
        Parametri:
            epochs: Numero di epoche
            save_dir: Directory in cui salvare i checkpoints
            log_interval: Intervallo di logging
        """
        # Inizializza Weights & Biases
        wandb.init(project=self.project_name, name=self.experiment_name)
        
        # Configura la directory per i checkpoints
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_iou = 0.0
            train_dice = 0.0
            
            # Progress bar per il training
            train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            
            for batch_idx, batch in enumerate(train_pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)

                if masks.shape[1:] != outputs.shape[2:]:
                    # Le maschere devono essere prima convertite in float per l'interpolazione
                    # Aggiungiamo un canale (unsqueeze) perché interpolate richiede un tensore 4D [B, C, H, W]
                    masks_resized = F.interpolate(
                        masks.float().unsqueeze(1),  # Aggiunge dimensione canale [B, 1, H, W]
                        size=outputs.shape[2:],      # Target size [64, 64]
                        mode='nearest'               # Usa 'nearest' per preservare le classi
                    ).squeeze(1).long()              # Rimuove la dimensione canale e converte in long
                else:
                    masks_resized = masks

                # Calcola la loss
                loss = self.criterion(outputs, masks_resized)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Aggiorna la loss
                train_loss += loss.item()
                
                # Calcola metriche
                batch_iou, batch_dice = self.calculate_metrics(outputs, masks_resized)
                train_iou += batch_iou
                train_dice += batch_dice
                
                # Aggiorna la progress bar
                train_pbar.set_postfix({
                    'loss': loss.item(),
                    'IoU': batch_iou,
                    'Dice': batch_dice
                })
                
                # Log di esempi ogni log_interval
                if batch_idx % log_interval == 0:
                    self.log_predictions(images, masks_resized, outputs, phase='train')
                    
            # Calcola medie
            train_loss /= len(self.train_loader)
            train_iou /= len(self.train_loader)
            train_dice /= len(self.train_loader)
            
            # Validazione
            val_loss, val_iou, val_dice = self.validate()
            
            # Aggiorna lo scheduler
            self.scheduler.step(val_loss)
            
            # Log su Weights & Biases
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_iou': train_iou,
                'train_dice': train_dice,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_dice': val_dice,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            
            # Salva il miglior modello
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou
                self.best_epoch = epoch + 1
                
                # Salva il checkpoint
                checkpoint_path = os.path.join(save_dir, f'best_model_{self.experiment_name}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_iou': val_iou,
                    'val_dice': val_dice
                }, checkpoint_path)
                
                wandb.save(checkpoint_path)
                print(f"Salvato il miglior modello con IoU: {val_iou:.4f}")
            
            # Stampa risultati dell'epoca
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f} | "
                  f"Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
        
        # Test finale
        if self.test_loader is not None:
            self.test()
        
        # Chiudi Weights & Biases
        wandb.finish()
        
    def validate(self):
        """
        Valuta il modello sul set di validazione
        
        Ritorna:
            val_loss: Loss media
            val_iou: IoU medio
            val_dice: Dice medio
        """
        self.model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        
        val_pbar = tqdm(self.val_loader, desc="Validation")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)

                masks = F.interpolate(masks.unsqueeze(1).float(), size=(64, 64), mode='nearest').squeeze(1).long()
                
                # Calcola la loss
                loss = self.criterion(outputs, masks)
                
                # Aggiorna la loss
                val_loss += loss.item()
                
                # Calcola metriche
                batch_iou, batch_dice = self.calculate_metrics(outputs, masks)
                val_iou += batch_iou
                val_dice += batch_dice
                
                # Aggiorna la progress bar
                val_pbar.set_postfix({
                    'loss': loss.item(),
                    'IoU': batch_iou,
                    'Dice': batch_dice
                })
                
                # Log del primo batch
                if batch_idx == 0:
                    self.log_predictions(images, masks, outputs, phase='val')
        
        # Calcola medie
        val_loss /= len(self.val_loader)
        val_iou /= len(self.val_loader)
        val_dice /= len(self.val_loader)
        
        return val_loss, val_iou, val_dice
    
    def test(self):
        """
        Valuta il modello sul set di test
        """
        self.model.eval()
        test_loss = 0.0
        test_iou = 0.0
        test_dice = 0.0
        
        # Prepara la matrice di confusione
        num_classes = self.model.segformer.config.num_labels
        conf_matrix = np.zeros((num_classes, num_classes))
        
        test_pbar = tqdm(self.test_loader, desc="Test")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calcola la loss
                loss = self.criterion(outputs, masks)
                
                # Aggiorna la loss
                test_loss += loss.item()
                
                # Calcola metriche
                batch_iou, batch_dice = self.calculate_metrics(outputs, masks)
                test_iou += batch_iou
                test_dice += batch_dice
                
                # Aggiorna la progress bar
                test_pbar.set_postfix({
                    'loss': loss.item(),
                    'IoU': batch_iou,
                    'Dice': batch_dice
                })
                
                # Log esempi di predizione
                if batch_idx % 5 == 0:
                    self.log_predictions(images, masks, outputs, phase='test')
                
                # Aggiorna la matrice di confusione
                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks_np = masks.detach().cpu().numpy()
                
                for cl in range(num_classes):
                    conf_matrix[cl] += np.bincount(
                        (masks_np == cl).flatten() * (num_classes) + preds.flatten(),
                        minlength=num_classes**2
                    ).reshape(num_classes, num_classes)[cl]
        
        # Calcola medie
        test_loss /= len(self.test_loader)
        test_iou /= len(self.test_loader)
        test_dice /= len(self.test_loader)
        
        # Calcola precisione e recall per classe
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        
        for cl in range(num_classes):
            precision[cl] = conf_matrix[cl, cl] / (conf_matrix[:, cl].sum() + 1e-10)
            recall[cl] = conf_matrix[cl, cl] / (conf_matrix[cl, :].sum() + 1e-10)
        
        # Normalizza la matrice di confusione
        conf_matrix_norm = conf_matrix / (conf_matrix.sum(axis=1, keepdims=True) + 1e-10)
        
        # Crea e salva la figura della matrice di confusione
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        
        # Log su W&B
        wandb.log({
            'test_loss': test_loss,
            'test_iou': test_iou,
            'test_dice': test_dice,
            'confusion_matrix': wandb.Image('confusion_matrix.png'),
            'class_precision': {f'precision_class_{i}': p for i, p in enumerate(precision)},
            'class_recall': {f'recall_class_{i}': r for i, r in enumerate(recall)}
        })
        
        # Stampa risultati finali
        print(f"Test - Loss: {test_loss:.4f}, IoU: {test_iou:.4f}, Dice: {test_dice:.4f}")
    
    def calculate_metrics(self, outputs, targets):
        """
        Calcola le metriche di segmentazione
        
        Parametri:
            outputs: Output del modello
            targets: Ground truth
            
        Ritorna:
            iou: Intersection over Union
            dice: Dice coefficient
        """
        # Ottieni le predizioni
        preds = torch.argmax(outputs, dim=1)
        
        # Calcola IoU e Dice per ogni classe
        num_classes = outputs.size(1)
        iou_sum = 0.0
        dice_sum = 0.0
        
        # Ignora il background (indice 0) se ci sono più di 1 classe
        start_class = 1 if num_classes > 1 else 0
        valid_classes = 0
        
        for c in range(start_class, num_classes):
            pred_mask = (preds == c)
            target_mask = (targets == c)
            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            if union > 0:
                # IoU per questa classe
                iou = intersection / union
                iou_sum += iou.item()
                
                # Dice per questa classe
                dice = (2 * intersection) / (pred_mask.sum() + target_mask.sum() + 1e-10)
                dice_sum += dice.item()
                
                valid_classes += 1
        
        # Calcola medie
        iou = iou_sum / (valid_classes + 1e-10)
        dice = dice_sum / (valid_classes + 1e-10)
        
        return iou, dice
    
    def log_predictions(self, images, masks, outputs, phase='train', max_samples=4):
        """
        Visualizza e logga le predizioni a W&B
        
        Parametri:
            images: Immagini di input
            masks: Ground truth
            outputs: Output del modello
            phase: Fase (train, val, test)
            max_samples: Numero massimo di campioni da visualizzare
        """
        # Limita il numero di campioni
        n_samples = min(max_samples, images.size(0))
        
        # Ottieni le predizioni
        preds = torch.argmax(outputs[:n_samples], dim=1).detach().cpu().numpy()
        masks = masks[:n_samples].detach().cpu().numpy()
        
        # De-normalizza le immagini
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        images_np = []
        for i in range(n_samples):
            img = images[i].detach().cpu()
            img = img * std + mean
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            images_np.append(img)
        
        # Crea mappa colori
        num_classes = outputs.size(1)
        cmap = plt.cm.get_cmap('tab10', num_classes)
        
        # Prepara le figure
        for i in range(n_samples):
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            # Immagine originale
            axs[0].imshow(images_np[i])
            axs[0].set_title('Input Image')
            axs[0].axis('off')
            
            # Ground truth
            axs[1].imshow(masks[i], cmap=cmap, vmin=0, vmax=num_classes-1)
            axs[1].set_title('Ground Truth')
            axs[1].axis('off')
            
            # Predizione
            axs[2].imshow(preds[i], cmap=cmap, vmin=0, vmax=num_classes-1)
            axs[2].set_title('Prediction')
            axs[2].axis('off')
            
            plt.tight_layout()
            
            # Salva e logga a W&B
            fig_path = f'{phase}_sample_{i}.png'
            plt.savefig(fig_path)
            plt.close(fig)
            
            wandb.log({f"{phase}_predictions_{i}": wandb.Image(fig_path)})


# Utilizzo del trainer
def train_segformer(train_loader, val_loader, test_loader=None, num_classes=4, 
                    epochs=50, learning_rate=1e-4, weight_decay=1e-4, 
                    device='cuda', class_weights=None, pretrained=True):
    """
    Funzione principale per il training di SegFormer
    
    Parametri:
        train_loader: Dataloader per il training
        val_loader: Dataloader per la validazione
        test_loader: Dataloader per il test
        num_classes: Numero di classi
        epochs: Numero di epoche
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Dispositivo (cuda/cpu)
        class_weights: Pesi delle classi
        pretrained: Se utilizzare un modello pre-addestrato
    """
    # Imposta seed per riproducibilità
    set_seed(42)
    
    # Definisci mapping id2label
    id2label = {i: f"class_{i}" for i in range(num_classes)}
    
    # Crea il modello
    model = SegFormerModel(num_classes=num_classes, id2label=id2label, pretrained=pretrained)
    
    # Configura il trainer
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        class_weights=class_weights,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        project_name='segmentation',
        experiment_name='segformer'
    )
    
    # Esegui il training
    trainer.train(epochs=epochs, save_dir='checkpoints')
    
    return model, trainer
