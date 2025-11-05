# Evaluation function
import torch
from sklearn.metrics import f1_score

def evaluate_model(model, test_loader, device):

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
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    
    return test_acc, test_f1, test_preds, test_labels