import torch

def train_model(model, train_dataloader, criterion, optimizer, scheduler, n_epochs, device, patience, min_delta):  
    
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
            # Compute training accuracy 
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        # Epoch summary
        epoch_train_loss = epoch_loss / total  
        epoch_train_acc = 100.0 * correct / total
        history.append((epoch_train_loss, epoch_train_acc))
        print(f"Epoch [{epoch+1}/{n_epochs}] Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}%")  
        
        # Step the scheduler based on training loss
        scheduler.step(epoch_train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")

        # Early stopping check
        if epoch_train_loss < best_loss - min_delta: # Improvement compared to best loss
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

# Evaluation function

def evaluate_model(model, best_model_state, test_loader, device):

    model.load_state_dict(best_model_state['model_state_dict']) 
    model.eval()
    test_preds = []
    test_labels = []
    test_correct = 0
    test_total = 0
    
    # Convert test data to tensors and move to device
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, dim = 1)
            test_correct += (predicted == y).sum().item()
            test_total += y.size(0)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(y.cpu().numpy())
    
    test_acc = 100.0 * test_correct / test_total
    
    return test_acc, test_preds, test_labels