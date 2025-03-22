import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import wandb
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

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
            # Utilizza un modello pre-addestrato
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
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.class_weights = class_weights
        
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
        
        # Metriche
        self.best_val_iou = 0.0
        self.best_epoch = 0
        
        # Scaler per mixed precision
        self.scaler = GradScaler()
        
    def train(self, epochs=100, save_dir='checkpoints', log_interval=20):
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
                
                # Forward pass con mixed precision
                self.optimizer.zero_grad()
                
                with autocast():
                    outputs = self.model(images)
                    
                    if masks.shape[1:] != outputs.shape[2:]:
                        masks_resized = F.interpolate(
                            masks.unsqueeze(1).float(),
                            size=outputs.shape[2:],
                            mode='nearest'
                        ).squeeze(1).long()
                    else:
                        masks_resized = masks
                        
                    loss = self.criterion(outputs, masks_resized)

                # Backward con lo Scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Calcola metriche su CPU per liberare memoria
                with torch.no_grad():
                    batch_iou, batch_dice = self.calculate_metrics(outputs.detach(), masks_resized.detach())
                
                train_loss += loss.item()
                train_iou += batch_iou
                train_dice += batch_dice
                
                # Aggiorna la progress bar
                train_pbar.set_postfix({
                    'loss': loss.item(),
                    'IoU': batch_iou,
                    'Dice': batch_dice
                })
                
                # Log meno frequenti per risparmiare memoria
                if batch_idx % 50 == 0:
                    wandb.log({
                        'batch/train_loss': loss.item(),
                        'batch/train_iou': batch_iou,
                        'batch/train_dice': batch_dice,
                        'batch_step': epoch * len(self.train_loader) + batch_idx
                    })
                
                # Log di esempi solo occasionalmente
                if batch_idx % log_interval == 0 and batch_idx > 0:
                    self.log_predictions(images[:2], masks_resized[:2], outputs[:2], phase='train', max_samples=2)
                
                # Libera memoria
                del images, masks, masks_resized, outputs, loss
                torch.cuda.empty_cache()
                    
            # Calcola medie
            train_loss /= len(self.train_loader)
            train_iou /= len(self.train_loader)
            train_dice /= len(self.train_loader)
            
            # Validazione con minor uso di memoria
            val_loss, val_iou, val_dice = self.validate()
            
            # Aggiorna lo scheduler
            self.scheduler.step(val_loss)
            
            # Log su Weights & Biases (solo metriche essenziali)
            wandb.log({
                'epoch': epoch + 1,
                'epoch/train_loss': train_loss,
                'epoch/train_iou': train_iou,
                'epoch/val_loss': val_loss,
                'epoch/val_iou': val_iou,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            
            # Salva il miglior modello
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou
                self.best_epoch = epoch + 1

                encoder_name = self.model.segformer.config.model_type
                num_classes = self.model.segformer.config.num_labels
                
                checkpoint_filename = f"{self.experiment_name}_ep-{epoch+1}_iou-{val_iou:.4f}.pth"
                checkpoint_path = os.path.join(save_dir, checkpoint_filename)
                
                # Salva solo le informazioni essenziali
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'val_iou': val_iou,
                }, checkpoint_path)
                
                print(f"Salvato il miglior modello con IoU: {val_iou:.4f}")
            
            # Stampa risultati dell'epoca
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, IoU: {train_iou:.4f} | "
                  f"Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}")
            
            # Forza pulizia memoria
            torch.cuda.empty_cache()
        
        # Test finale se necessario
        if self.test_loader is not None:
            self.test()
        
        # Chiudi Weights & Biases
        wandb.finish()

    def validate(self):
        """Valuta il modello sul set di validazione con gestione efficiente della memoria"""
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

                # Resize masks se necessario
                if masks.shape[1:] != outputs.shape[2:]:
                    masks = F.interpolate(
                        masks.unsqueeze(1).float(), 
                        size=outputs.shape[2:], 
                        mode='nearest'
                    ).squeeze(1).long()
                
                # Calcola la loss
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calcola metriche
                batch_iou, batch_dice = self.calculate_metrics(outputs, masks)
                val_iou += batch_iou
                val_dice += batch_dice
                
                val_pbar.set_postfix({
                    'loss': loss.item(),
                    'IoU': batch_iou
                })
                
                # Log solo del primo batch per risparmiare memoria
                if batch_idx == 0:
                    self.log_predictions(images[:2], masks[:2], outputs[:2], phase='val', max_samples=2)
                
                # Libera memoria
                del images, masks, outputs, loss
                torch.cuda.empty_cache()
        
        # Calcola medie
        val_loss /= len(self.val_loader)
        val_iou /= len(self.val_loader)
        val_dice /= len(self.val_loader)
        
        return val_loss, val_iou, val_dice
    
    def test(self):
        """Versione ottimizzata del test"""
        self.model.eval()
        test_loss = 0.0
        test_iou = 0.0
        test_dice = 0.0
        
        num_classes = self.model.segformer.config.num_labels
        conf_matrix = np.zeros((num_classes, num_classes))
        
        test_pbar = tqdm(self.test_loader, desc="Test")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)

                # Resize masks se necessario
                if masks.shape[1:] != outputs.shape[2:]:
                    masks = F.interpolate(
                        masks.float().unsqueeze(1), 
                        size=outputs.shape[2:], 
                        mode='nearest'
                    ).squeeze(1).long()
                
                # Calcola la loss
                loss = self.criterion(outputs, masks)
                test_loss += loss.item()
                
                # Calcola metriche
                batch_iou, batch_dice = self.calculate_metrics(outputs, masks)
                test_iou += batch_iou
                test_dice += batch_dice
                
                test_pbar.set_postfix({
                    'loss': loss.item(),
                    'IoU': batch_iou
                })
                
                # Log meno frequente
                if batch_idx % 10 == 0:
                    self.log_predictions(images[:1], masks[:1], outputs[:1], phase='test', max_samples=1)
                
                # Libera memoria
                del images, masks, outputs, loss
                torch.cuda.empty_cache()
        
        # Calcola medie
        test_loss /= len(self.test_loader)
        test_iou /= len(self.test_loader)
        test_dice /= len(self.test_loader)
        
        # Log minimo su W&B
        wandb.log({
            'test_loss': test_loss,
            'test_iou': test_iou,
            'test_dice': test_dice
        })
        
        print(f"Test - Loss: {test_loss:.4f}, IoU: {test_iou:.4f}, Dice: {test_dice:.4f}")
    
    def calculate_metrics(self, outputs, targets):
        """Calcola le metriche di segmentazione in modo efficiente per la memoria"""
        # Porta i tensori su CPU per liberare memoria GPU
        outputs_cpu = outputs.detach().cpu()
        targets_cpu = targets.detach().cpu()
        
        # Ottieni le predizioni
        preds = torch.argmax(outputs_cpu, dim=1)
        
        # Calcola IoU e Dice per ogni classe
        num_classes = outputs_cpu.size(1)
        iou_sum = 0.0
        dice_sum = 0.0
        
        # Ignora il background (indice 0) se ci sono più di 1 classe
        start_class = 1 if num_classes > 1 else 0
        valid_classes = 0
        
        for c in range(start_class, num_classes):
            pred_mask = (preds == c)
            target_mask = (targets_cpu == c)
            
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
        
        # Libera memoria
        del outputs_cpu, targets_cpu, preds
        
        return iou, dice
    
    def log_predictions(self, images, masks, outputs, phase='train', max_samples=2):
        """Versione ottimizzata per logging con minor uso di memoria"""
        import matplotlib.pyplot as plt
        
        n_samples = min(max_samples, images.size(0))
        
        # Processa su CPU per risparmiare memoria GPU
        images_cpu = images[:n_samples].detach().cpu()
        masks_cpu = masks[:n_samples].detach().cpu()
        outputs_cpu = outputs[:n_samples].detach().cpu()
        
        # Ottieni predizioni
        preds = torch.argmax(outputs_cpu, dim=1).numpy()
        masks_np = masks_cpu.numpy()
        
        # Denormalizza le immagini
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        for i in range(n_samples):
            plt.figure(figsize=(12, 4))
            
            # Immagine
            img = images_cpu[i]
            img = img * std + mean
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title('Input')
            plt.axis('off')
            
            # Ground truth
            plt.subplot(1, 3, 2)
            plt.imshow(masks_np[i], cmap='tab10')
            plt.title('Ground Truth')
            plt.axis('off')
            
            # Predizione
            plt.subplot(1, 3, 3)
            plt.imshow(preds[i], cmap='tab10')
            plt.title('Prediction')
            plt.axis('off')
            
            plt.tight_layout()
            
            # Log a W&B
            wandb.log({f"{phase}_prediction": wandb.Image(plt.gcf())})
            plt.close()
        
        # Libera memoria
        del images_cpu, masks_cpu, outputs_cpu, preds

# Funzione principale per training
def train_segformer(train_loader, val_loader, test_loader=None, num_classes=4, 
                    epochs=50, learning_rate=1e-4, weight_decay=1e-4, 
                    device='cuda', class_weights=None, pretrained=True):
    
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