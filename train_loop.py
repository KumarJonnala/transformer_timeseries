import torch

def train_model(model, train_dataloader, criterion, optimizer, n_epochs, device):  
    history = []

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

    return history