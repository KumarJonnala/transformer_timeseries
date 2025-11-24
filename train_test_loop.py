import torch

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, n_epochs, device, patience, min_delta):  
    """
    Train model with validation set monitoring.
    
    Args:
        model: Neural network model
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        n_epochs: Maximum number of epochs
        device: Device to train on (CPU/GPU)
        patience: Early stopping patience
        min_delta: Minimum improvement for early stopping
        
    Returns:
        history: List of tuples (train_loss, train_acc, val_loss, val_acc)
        best_model_state: Dictionary containing best model state
    """
    
    history = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training phase 
    for epoch in range(n_epochs):
        # Training phase 
        model.train()                   # model set to training mode
        epoch_loss = 0.0                # accumulates loss for the epoch
        train_correct = 0                     # to compute accuracy
        train_total = 0 

        for x, y in train_dataloader:   # iterate training data in batches
            # Move to device
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()        # Reset gradients
            outputs = model(x)           # Forward pass
            loss = criterion(outputs, y) # Compute loss
            loss.backward()              # Backward pass
            optimizer.step()             # Update weights

            epoch_loss += loss.item() * x.size(0) # loss * batch_size
            # Compute training accuracy 
            _, predicted = torch.max(outputs, dim=1)
            train_correct += (predicted == y).sum().item()
            train_total += y.size(0)

        # Calculate training metrics
        epoch_train_loss = epoch_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in val_dataloader:
                x, y = x.to(device), y.to(device)
                
                outputs = model(x)
                loss = criterion(outputs, y)
                
                epoch_val_loss += loss.item() * x.size(0)
                _, predicted = torch.max(outputs, dim=1)
                val_correct += (predicted == y).sum().item()
                val_total += y.size(0)
        
        # Calculate validation metrics
        epoch_val_loss = epoch_val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total
        
        # Store history
        history.append((epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc))
        
        print(f"Epoch [{epoch+1}/{n_epochs}]")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"  Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
        # Step the scheduler based on validation loss
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning rate: {current_lr:.6f}")

        # Early stopping check based on validation loss
        if epoch_val_loss < best_val_loss - min_delta:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc,
            }
            print(f"  ✓ New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  Early stopping counter: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_model_state['epoch']+1}")
            break

    # Load best model if training completed without early stopping
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])

    return history, best_model_state


def evaluate_model(model, best_model_state, test_loader, device):
    """
    Evaluate model on test set.
    
    Args:
        model: Neural network model
        best_model_state: Dictionary containing best model state
        test_loader: DataLoader for test data
        device: Device to evaluate on (CPU/GPU)
        
    Returns:
        test_acc: Test accuracy percentage
        test_preds: List of predictions
        test_labels: List of true labels
    """
    
    model.load_state_dict(best_model_state['model_state_dict']) 
    model.eval()
    test_preds = []
    test_labels = []
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, dim=1)
            test_correct += (predicted == y).sum().item()
            test_total += y.size(0)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(y.cpu().numpy())
    
    test_acc = 100.0 * test_correct / test_total
    
    return test_acc, test_preds, test_labels