import torch

def train_model4(model, train_dataloader, criterion, optimizer, n_epochs, device, patience, min_delta):  
    
    history = []
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training phase 
    for epoch in range(n_epochs):
        # Training phase 
        model.train()                   # model set to training mode
        epoch_loss = 0.0                # accumulates loss for the epoch
        correct = 0                     # to compute accuracy
        total = 0 

        for x, y in train_dataloader:   # iterate training data in batches
            # Move to device
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()        # Reset gradients
            outputs = model(x)           # Forward pass
            loss = criterion(outputs, y) # Compute loss
            loss.backward()              # Backward pass
            optimizer.step()             # Update weights

            epoch_loss += loss.item() * x.size(0) # loss * batch_size
            # ---- Compute training accuracy ----
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        # Epoch summary
        epoch_train_loss = epoch_loss / total  # Fixed: use epoch_train_loss instead of epoch_loss
        epoch_train_acc = 100.0 * correct / total

        history.append((epoch_train_loss, epoch_train_acc))

        print(f"Epoch [{epoch+1}/{n_epochs}] Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}%")  # Fixed: use epoch_train_loss and epoch_train_acc
        
        # Early stopping check
        if epoch_train_loss < best_loss - min_delta:
            best_loss = epoch_train_loss
            patience_counter = 0
            # Save best model state
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            # Restore best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state['model_state_dict'])
            break

    return history, best_model_state