# Training + validation function (without LOOCV)
import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, device):
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(n_epochs):
        # ---- Training phase ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            # Move to device
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == y).sum().item()
            train_total += y.size(0)

        # Calculate training metrics
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total
        
        # ---- Validation phase ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                
                val_loss += loss.item() * x.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == y).sum().item()
                val_total += y.size(0)
        
        # Calculate validation metrics
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total
        
        # Store metrics
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
            }, 'best_model.pt')
        
        # Print progress
        print(f"Epoch [{epoch+1}/{n_epochs}]")
        print(f"Train - Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}%")
        print(f"Valid - Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2f}%")
        print("-" * 50)
    
    return history