import torch
import numpy as np


def create_time_marks(batch_size, seq_len, device):
    """
    Create dummy time marks for WESAD data (which has no temporal information).
    Returns tensor of ones that will be used for masking in the classification method.
    
    Args:
        batch_size: Number of samples in batch
        seq_len: Sequence length
        device: Device to create tensor on
    
    Returns:
        x_mark: Tensor of shape (batch_size, seq_len) with all ones
    """
    # For classification, x_mark is used for padding mask (1 = valid, 0 = padding)
    # Since WESAD has no padding, we use all ones
    return torch.ones(batch_size, seq_len).to(device)


def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, 
                n_epochs, device, patience, min_delta):
    """
    Train pyraformer model with validation set monitoring.
    
    Args:
        model: Pyraformer model
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

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            
            # Create time marks (dummy for WESAD)
            x_mark_enc = create_time_marks(x.size(0), x.size(1), device)
            
            # For classification task, decoder inputs are not used
            # But we still need to pass dummy values
            x_dec = torch.zeros_like(x).to(device)
            x_mark_dec = create_time_marks(x.size(0), x.size(1), device)
            
            optimizer.zero_grad()
            
            # Forward pass: pyraformer expects (x_enc, x_mark_enc, x_dec, x_mark_dec)
            outputs = model(x, x_mark_enc, x_dec, x_mark_dec)
            loss = criterion(outputs, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
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
                
                x_mark_enc = create_time_marks(x.size(0), x.size(1), device)
                x_dec = torch.zeros_like(x).to(device)
                x_mark_dec = create_time_marks(x.size(0), x.size(1), device)
                
                outputs = model(x, x_mark_enc, x_dec, x_mark_dec)
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
        
        # Step the scheduler
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning rate: {current_lr:.6f}")

        # Early stopping check
        if epoch_val_loss < best_val_loss - min_delta:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  New best model! Val Loss improved to {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    return history, best_model_state


def evaluate_model(model, best_model_state, test_loader, device):
    """
    Evaluate the pyraformer model on test set.
    
    Args:
        model: Pyraformer model
        best_model_state: Best model state dictionary
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        test_acc: Test accuracy
        test_preds: Predictions
        test_labels: True labels
    """
    model.load_state_dict(best_model_state)
    model.eval()
    
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            x_mark_enc = create_time_marks(x.size(0), x.size(1), device)
            x_dec = torch.zeros_like(x).to(device)
            x_mark_dec = create_time_marks(x.size(0), x.size(1), device)
            
            outputs = model(x, x_mark_enc, x_dec, x_mark_dec)
            _, predicted = torch.max(outputs, dim=1)
            
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(y.cpu().numpy())
    
    test_acc = 100.0 * np.sum(np.array(test_preds) == np.array(test_labels)) / len(test_labels)
    
    return test_acc, test_preds, test_labels
