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
import torchvision.models.segmentation as segmentation
import torchvision
import time
from datetime import datetime
import transformers
from transformers import AutoModelForSemanticSegmentation, AutoConfig

# Imposta seed per riproducibilità
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class DeepLabV3_MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3_MobileNetV2, self).__init__()
        # Usa direttamente il modello deeplabv3_mobilenet_v3_large
        self.model = segmentation.deeplabv3_mobilenet_v3_large(pretrained=False)
        # Modifica il classificatore per il numero corretto di classi
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    
    def forward(self, x):
        return self.model(x)['out']
    
    def get_model_name(self):
        return "DeepLabV3_MobileNetV2"
    
class Mask2FormerModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(Mask2FormerModel, self).__init__()
        
        # Inizializza il modello Mask2Former
        if pretrained:
            # Carica il modello pre-addestrato
            self.model = transformers.Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-base-coco-instance",
                ignore_mismatched_sizes=True
            )
            
            # Modifica la configurazione per il numero di classi
            self.model.config.num_labels = num_classes
            
            # Debug: stampa la struttura del modello per ispezione
            print("Struttura del modello (primi livelli):")
            for name, module in self.model.named_children():
                print(f"- {name}: {type(module)}")
            
            # Adatta il modello al numero di classi desiderato
            # Mask2Former ha una struttura specifica per la segmentazione
            if hasattr(self.model, 'mask_classifier'):
                # Se esiste un classificatore di maschere dedicato
                in_features = self.model.mask_classifier.in_features
                self.model.mask_classifier = nn.Linear(in_features, num_classes)
                print(f"Modificato mask_classifier con {num_classes} classi")
            elif hasattr(self.model, 'class_predictor'):
                # Alcuni modelli usano un predittore di classe
                self.model.class_predictor = nn.Linear(
                    self.model.class_predictor.in_features, 
                    num_classes
                )
                print(f"Modificato class_predictor con {num_classes} classi")
        else:
            # Inizializzazione da zero con configurazione personalizzata
            config = transformers.Mask2FormerConfig(
                num_labels=num_classes,
                backbone="swin-tiny-patch4-window7-224"
            )
            self.model = transformers.Mask2FormerForUniversalSegmentation(config)
    
    def forward(self, pixel_values):
        # Esegui il forward pass del modello
        outputs = self.model(pixel_values=pixel_values)
        
        # Ispeziona l'output per debug (solo per il primo batch)
        if not hasattr(self, '_debug_done'):
            print("\nOutput keys disponibili:", [k for k in outputs.keys() if not k.startswith('_')])
            self._debug_done = True
        
        # Gestisci i diversi tipi di output che Mask2Former potrebbe avere
        if hasattr(outputs, 'masks_queries_logits'):
            # Questa è l'output per segmentazione di istanze
            # Qui dobbiamo processare ulteriormente per ottenere una mappa di segmentazione
            # Per semplicità, prendiamo il massimo valore per ogni pixel
            batch_size = outputs.masks_queries_logits.shape[0]
            height, width = outputs.masks_queries_logits.shape[-2:]
            
            # Reshape e permute per ottenere [B, H, W, Q]
            masks = outputs.masks_queries_logits.sigmoid().reshape(
                batch_size, -1, height, width
            )
            
            # Ora combiniamo con class_queries_logits [B, Q, C]
            if hasattr(outputs, 'class_queries_logits'):
                classes = outputs.class_queries_logits.softmax(dim=-1)
                
                # Creiamo una mappa di segmentazione completa [B, C, H, W]
                segmentation = torch.zeros(
                    batch_size, 
                    self.model.config.num_labels, 
                    height, 
                    width, 
                    device=masks.device
                )
                
                # Per ogni query, aggiungiamo il suo contributo alla mappa di segmentazione
                num_queries = masks.shape[1]
                for i in range(num_queries):
                    mask_i = masks[:, i].unsqueeze(1)  # [B, 1, H, W]
                    class_i = classes[:, i].unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
                    contribution = mask_i * class_i  # [B, C, H, W]
                    segmentation += contribution
                
                return segmentation
            else:
                # Fallback: restituisci solo le maschere
                return masks
        elif hasattr(outputs, 'segmentation_logits'):
            # Questa è l'output per segmentazione semantica
            return outputs.segmentation_logits
        elif hasattr(outputs, 'logits'):
            # Output generico
            return outputs.logits
        else:
            # Se non troviamo attributi specifici, cerchiamo di identificare
            # il tensore più promettente nell'output
            for key in outputs.keys():
                if key.startswith('_'):
                    continue
                value = outputs[key]
                if isinstance(value, torch.Tensor) and len(value.shape) == 4:
                    print(f"Usando l'output '{key}' con forma {value.shape}")
                    return value
            
            # Se arriviamo qui, non siamo stati in grado di trovare un output adatto
            raise ValueError(
                "Non è stato possibile individuare un output valido nel modello Mask2Former. "
                f"Output disponibili: {list(outputs.keys())}"
            )
    
    def get_model_name(self):
        return "Mask2Former"

# Classe per la gestione del training
class ModelTrainer:
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
                 experiment_name=None,
                 max_checkpoints_to_keep=3):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.class_weights = class_weights
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
        
        # Determina il nome del modello per i log
        self.model_name = getattr(self.model, 'get_model_name', lambda: 'Unknown')()
        
        # Imposta automaticamente il nome dell'esperimento se non specificato
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            experiment_name = f"{self.model_name}_{timestamp}"
        
        self.experiment_name = experiment_name
        
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
        
        # Metriche
        self.best_val_iou = 0.0
        self.best_epoch = 0
        
        # Lista per tenere traccia dei checkpoint salvati
        self.saved_checkpoints = []
        
    def check_mask_values(self, mask_tensor, num_classes):
        """Verifica che i valori della maschera siano validi"""
        min_val = mask_tensor.min().item()
        max_val = mask_tensor.max().item()
        
        if min_val < 0 or max_val >= num_classes:
            raise ValueError(f"Valori della maschera fuori range: min={min_val}, max={max_val}, num_classes={num_classes}")
        
        return True
        
    def train(self, epochs=100, save_dir='checkpoints', log_interval=20, improvement_threshold=0.005):
        """
        Addestra il modello
        
        Args:
            epochs: Numero di epoche di training
            save_dir: Directory dove salvare i checkpoint
            log_interval: Frequenza dei log visivi
            improvement_threshold: Soglia di miglioramento per salvare un nuovo checkpoint
                                  (percentuale di miglioramento necessaria)
        """
        # Crea una directory specifica per il modello
        model_save_dir = os.path.join(save_dir, self.model_name)
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Inizializza Weights & Biases con configurazione espansa
        wandb_config = {
            "model": self.model_name,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "epochs": epochs,
            "batch_size": next(iter(self.train_loader))['image'].shape[0],
            "weight_decay": self.optimizer.param_groups[0]['weight_decay'],
            "optimizer": self.optimizer.__class__.__name__,
            "scheduler": self.scheduler.__class__.__name__,
            "experiment_name": self.experiment_name
        }
        
        wandb.init(project=self.project_name, name=self.experiment_name, config=wandb_config)
        
        # Log delle architetture dei modelli (se disponibile)
        try:
            wandb.watch(self.model, log="all", log_freq=100)
        except:
            print("Non è stato possibile utilizzare wandb.watch su questo modello")
        
        # Tempo di inizio training
        training_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.model.train()
            train_loss = 0.0
            train_iou = 0.0
            train_dice = 0.0
            
            # Progress bar per il training
            train_pbar = tqdm(self.train_loader, desc=f"Epoca {epoch+1}/{epochs} [{self.model_name} - Train]")
            
            for batch_idx, batch in enumerate(train_pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                self.check_mask_values(masks, 5)
                
                self.optimizer.zero_grad()
                
                # Forward pass
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
                
                # Backward
                loss.backward()
                self.optimizer.step()
                
                # Calcola metriche su CPU per liberare memoria
                with torch.no_grad():
                    batch_iou, batch_dice = self.calculate_metrics(outputs.detach(), masks_resized.detach())
                
                train_loss += loss.item()
                train_iou += batch_iou
                train_dice += batch_dice
                
                # Aggiorna la progress bar
                train_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'IoU': f"{batch_iou:.4f}",
                    'Dice': f"{batch_dice:.4f}"
                })
                
                # Log meno frequenti per risparmiare memoria
                if batch_idx % 50 == 0:
                    wandb.log({
                        f'{self.model_name}/batch/train_loss': loss.item(),
                        f'{self.model_name}/batch/train_iou': batch_iou,
                        f'{self.model_name}/batch/train_dice': batch_dice,
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
            
            # Calcola tempo trascorso per questa epoca
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - training_start_time
            
            # Log su Weights & Biases (solo metriche essenziali)
            wandb.log({
                'epoch': epoch + 1,
                f'{self.model_name}/epoch/train_loss': train_loss,
                f'{self.model_name}/epoch/train_iou': train_iou,
                f'{self.model_name}/epoch/train_dice': train_dice,
                f'{self.model_name}/epoch/val_loss': val_loss,
                f'{self.model_name}/epoch/val_iou': val_iou,
                f'{self.model_name}/epoch/val_dice': val_dice,
                f'{self.model_name}/learning_rate': self.optimizer.param_groups[0]['lr'],
                f'{self.model_name}/epoch_time_seconds': epoch_time,
                f'{self.model_name}/total_time_minutes': total_time / 60
            })
            
            # Controlla se salvare il checkpoint
            should_save = False
            relative_improvement = 0
            
            if val_iou > self.best_val_iou:
                relative_improvement = (val_iou - self.best_val_iou) / max(self.best_val_iou, 1e-5)
                
                # Salva se è il miglior modello finora o se l'improvement è significativo
                if relative_improvement > improvement_threshold or self.best_val_iou == 0.0:
                    should_save = True
                    self.best_val_iou = val_iou
                    self.best_epoch = epoch + 1
            
            if should_save:
                # Crea un nome significativo per il checkpoint
                timestamp = datetime.now().strftime("%Y%m%d-%H%M")
                checkpoint_filename = (
                    f"{self.model_name}_"
                    f"epoch{epoch+1}_"
                    f"iou{val_iou:.4f}_"
                    f"dice{val_dice:.4f}_"
                    f"{timestamp}.pth"
                )
                checkpoint_path = os.path.join(model_save_dir, checkpoint_filename)
                
                # Salva informazioni complete
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_iou': val_iou,
                    'val_dice': val_dice,
                    'val_loss': val_loss,
                    'train_iou': train_iou,
                    'train_dice': train_dice,
                    'train_loss': train_loss,
                    'model_name': self.model_name,
                    'timestamp': timestamp,
                }, checkpoint_path)
                
                # Aggiungi alla lista dei checkpoint salvati
                self.saved_checkpoints.append(checkpoint_path)
                
                # Limita il numero di checkpoint
                if len(self.saved_checkpoints) > self.max_checkpoints_to_keep:
                    # Mantieni sempre il checkpoint migliore e quelli più recenti
                    checkpoints_to_delete = sorted(
                        self.saved_checkpoints[:-self.max_checkpoints_to_keep], 
                        key=lambda x: os.path.getmtime(x)
                    )
                    
                    for old_checkpoint in checkpoints_to_delete:
                        if os.path.exists(old_checkpoint):
                            os.remove(old_checkpoint)
                            print(f"Rimosso vecchio checkpoint: {os.path.basename(old_checkpoint)}")
                        
                        self.saved_checkpoints.remove(old_checkpoint)
                
                print(f"Salvato modello - Epoca: {epoch+1}, IoU: {val_iou:.4f}, Miglioramento: {relative_improvement:.2%}")
            
            # Forza pulizia memoria
            torch.cuda.empty_cache()
        
        # Riepilogo finale
        print("\n" + "="*50)
        print(f"Training completato per {self.model_name}")
        print(f"Miglior IoU: {self.best_val_iou:.4f} ottenuto all'epoca {self.best_epoch}")
        print(f"Checkpoint salvati: {len(self.saved_checkpoints)}")
        for i, ckpt in enumerate(self.saved_checkpoints):
            print(f"  {i+1}. {os.path.basename(ckpt)}")
        print("="*50)
        
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
        
        val_pbar = tqdm(self.val_loader, desc=f"Validazione [{self.model_name}]")
        
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
                    'loss': f"{loss.item():.4f}",
                    'IoU': f"{batch_iou:.4f}"
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
        
        num_classes = 5
        conf_matrix = np.zeros((num_classes, num_classes))
        
        test_pbar = tqdm(self.test_loader, desc=f"Test [{self.model_name}]")
        
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
                    'loss': f"{loss.item():.4f}",
                    'IoU': f"{batch_iou:.4f}"
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
        
        # Log con nome del modello
        wandb.log({
            f'{self.model_name}/test_loss': test_loss,
            f'{self.model_name}/test_iou': test_iou,
            f'{self.model_name}/test_dice': test_dice
        })
        
        print(f"Test [{self.model_name}] - Loss: {test_loss:.4f}, IoU: {test_iou:.4f}, Dice: {test_dice:.4f}")
        
        # Salva i risultati finali del test in un file
        results_dir = os.path.join("results", self.model_name)
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        results_file = os.path.join(results_dir, f"test_results_{timestamp}.txt")
        
        with open(results_file, 'w') as f:
            f.write(f"Modello: {self.model_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"IoU: {test_iou:.6f}\n")
            f.write(f"Dice: {test_dice:.6f}\n")
            f.write(f"Loss: {test_loss:.6f}\n")
            f.write(f"Miglior IoU in validation: {self.best_val_iou:.6f} (Epoca {self.best_epoch})\n")
        
        print(f"Risultati del test salvati in: {results_file}")
    
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
            
            # Log a W&B con nome del modello
            wandb.log({f"{self.model_name}/{phase}_prediction": wandb.Image(plt.gcf())})
            plt.close()
        
        # Libera memoria
        del images_cpu, masks_cpu, outputs_cpu, preds

# Funzione principale per training
def train_model(train_loader, val_loader, test_loader=None, num_classes=4, 
                    epochs=50, learning_rate=1e-4, weight_decay=1e-4, 
                    device='cuda', class_weights=None, pretrained=True,
                    model_type="segformer", max_checkpoints=3,
                    improvement_threshold=0.005):
    
    # Imposta seed per riproducibilità
    set_seed(42)
    
    # Definisci mapping id2label
    id2label = {i: f"class_{i}" for i in range(num_classes)}
    
    # Crea il modello in base al tipo specificato
    if model_type.lower() == "segformer":
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(device)
        model_name = "Segformer"
    elif model_type.lower() == "deeplabv3":
        model = DeepLabV3_MobileNetV2(num_classes=num_classes)
        model_name = "DeepLabV3"
    elif model_type.lower() == "mask2former":
        model = Mask2FormerModel(num_classes=num_classes, pretrained=pretrained)
        model_name = "Mask2Former"
    else:
        raise ValueError(f"Tipo di modello non supportato: {model_type}. Scegli tra 'segformer' o 'deeplabv3'")
    
    # Timestamp per nome esperimento
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_name = f"{model_name}_{timestamp}"
    
    # Configura il trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        class_weights=class_weights,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        project_name='segmentation',
        experiment_name=experiment_name,
        max_checkpoints_to_keep=max_checkpoints
    )
    
    # Esegui il training
    trainer.train(
        epochs=epochs, 
        save_dir='checkpoints',
        improvement_threshold=improvement_threshold
    )
    
    return model, trainer