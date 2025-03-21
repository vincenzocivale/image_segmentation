import torch
import wandb

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Esegue una singola epoca di training.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: images type: {type(images)}, labels type: {type(labels)}")
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Logga la perdita su Weights & Biases
        wandb.log({"Training Loss": loss.item()})

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Valida il modello su un set di validazione.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    
    # Logga la perdita di validazione
    wandb.log({"Validation Loss": avg_loss})

    return avg_loss


def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device):
    """
    Funzione principale per il training.
    """
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        print(f"Inside train_model - Device: {device} - Type: {type(device)}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Salva il modello se migliora la loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "/home/inside-tech/Desktop/image_segmentation/model/best_model.pth")
            print("Modello migliorato e salvato!")
    print("Training completato!")
