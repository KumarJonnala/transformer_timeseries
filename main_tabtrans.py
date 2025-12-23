import os
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import recall_score, precision_score, f1_score, matthews_corrcoef, confusion_matrix

# import module files
from config import Config
from data import WESADDataset
from models import TabTransformer
from train_test import train_model, evaluate_model


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                   
    print(f"Using device: {device}")

    # Load dataset
    # ds = WESADDataset(Config.DATASET_PATH)
    ds = WESADDataset(None)
    with open(Config.PICKLE_PATH, 'rb') as f:
        saved_data = pickle.load(f)
        ds.data = saved_data['data']
        ds.labels = saved_data['labels']

    # Calculate subject indices
    subject_indices = {}
    start = 0
    for subject, count in Config.SUBJECT_COUNTS.items():
        end = start + count
        subject_indices[subject] = [start, end]
        start = end
    print(Config.SUBJECT_COUNTS, '\n', subject_indices)

    # LOOCV training and evaluation
    loocv_results = []
    for test_subject, test_range in subject_indices.items():
        print(f"\n{'='*60}")
        print(f"Test subject: {test_subject}, Range: {test_range}")
        print(f"{'='*60}")
        
        # Create test set
        test_indices = list(range(test_range[0], test_range[1]))

        # Train on all other subjects
        train_indices = []
        for subj, subj_range in subject_indices.items():
            if subj != test_subject:
                train_indices.extend(range(subj_range[0], subj_range[1]))
        
        
        # Get train and test data
        train_data = ds.data[train_indices]
        test_data = ds.data[test_indices]

        # Normalize using only training data
        train_means = train_data.mean(axis=(0, 1))
        train_stds = train_data.std(axis=(0, 1))

        # Apply normalization to train and test sets
        epsilon = 1e-8
        train_data_normalized = (train_data - train_means[None, None, :]) / (train_stds[None, None, :] + epsilon)
        test_data_normalized = (test_data - train_means[None, None, :]) / (train_stds[None, None, :] + epsilon)

        # Create datasets and dataloaders
        train_ds = TensorDataset(torch.FloatTensor(train_data_normalized), 
                                torch.LongTensor(ds.labels[train_indices]))
        test_ds = TensorDataset(torch.FloatTensor(test_data_normalized), 
                            torch.LongTensor(ds.labels[test_indices]))
        
        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        print(f"Train: {len(train_ds)} samples, Test: {len(test_ds)} samples, Total: {len(train_ds) + len(test_ds)} samples")
        
        # Initialize model
        model = TabTransformer(
            num_features=Config.NUM_FEATURES,
            num_classes=Config.NUM_CLASSES,
            dim_embedding=Config.EMBEDDING_DIM,
            num_heads=Config.NUM_HEADS,
            num_layers=Config.NUM_LAYERS,
            dropout=Config.DROPOUT
        ).to(device)
        
        # Training setup
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=Config.SCHEDULER_FACTOR, patience=Config.SCHEDULER_PATIENCE)
        
        # Train model
        history, best_model_state = train_model(
            model=model,
            train_dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=Config.NUM_EPOCHS,
            device=device,
            patience=Config.EARLY_STOPPING_PATIENCE,
            min_delta=Config.EARLY_STOPPING_MIN_DELTA
        )
        
        # Evaluate best model
        test_acc, test_preds, test_labels = evaluate_model(
        model=model,
        best_model_state=best_model_state,
        test_loader=test_loader,
        device=device
        )

        test_preds = np.array(test_preds)
        test_labels = np.array(test_labels)

        metrics = {
            'accuracy': test_acc,
            'confusion_matrix': confusion_matrix(test_labels, test_preds),
            'f1_score': f1_score(test_labels, test_preds, average='weighted'),
            'precision': precision_score(test_labels, test_preds, average='weighted'),
            'recall': recall_score(test_labels, test_preds, average='weighted'),
            'mcc': matthews_corrcoef(test_labels, test_preds)
        }
        
        loocv_results.append({
            'subject': test_subject,
            'metrics': metrics,
            'history': history,
            'predictions': test_preds,
            'true_labels': test_labels
        })
        
        print(f"Subject {test_subject} Results:")
        print(f"Test Labels: {test_labels}")
        print(f"Test Pred: {test_preds}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
        print(f"Accuracy: {metrics['accuracy']:.2f}%")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")

    # Print final summary
    print("\nConfiguration Parameters:")
    print("=" * 60)
    print("Training Configuration:")
    print(f"  Learning Rate: {Config.LEARNING_RATE}")
    print(f"  Number of Epochs: {Config.NUM_EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Early Stopping Patience: {Config.EARLY_STOPPING_PATIENCE}")
    print(f"  Early Stopping Min Delta: {Config.EARLY_STOPPING_MIN_DELTA}")
    
    print("\nModel Configuration:")
    print(f"  Number of Features: {Config.NUM_FEATURES}")
    print(f"  Number of Classes: {Config.NUM_CLASSES}")
    print(f"  Embedding Dimension: {Config.EMBEDDING_DIM}")
    print(f"  Number of Heads: {Config.NUM_HEADS}")
    print(f"  Number of Layers: {Config.NUM_LAYERS}")
    print(f"  Dropout Rate: {Config.DROPOUT}")
    
    print("\nOptimizer Configuration:")
    print(f"  Scheduler Factor: {Config.SCHEDULER_FACTOR}")
    print(f"  Scheduler Patience: {Config.SCHEDULER_PATIENCE}")
    print("=" * 60)

    print("\nMetrics of Each Subjects:")
    print("=" * 60)
    for r in loocv_results:
        print(f"{r['subject']}:"
              f"Acc={r['metrics']['accuracy']:.2f}, "
              f"F1={r['metrics']['f1_score']:.4f}, "
              f"Prec={r['metrics']['precision']:.4f}, "
              f"Rec={r['metrics']['recall']:.4f}, "
              f"MCC={r['metrics']['mcc']:.4f}")
    
    # Calculate and print mean metrics
    print(f"\nMean Metrics Across All Subjects:")
    print("=" * 60)
    print(f"Accuracy:  {np.mean([r['metrics']['accuracy'] for r in loocv_results]):.2f}%")
    print(f"F1 Score:  {np.mean([r['metrics']['f1_score'] for r in loocv_results]):.4f}")
    print(f"Precision: {np.mean([r['metrics']['precision'] for r in loocv_results]):.4f}")
    print(f"Recall:    {np.mean([r['metrics']['recall'] for r in loocv_results]):.4f}")
    print(f"MCC:       {np.mean([r['metrics']['mcc'] for r in loocv_results]):.4f}")

    return loocv_results

if __name__ == "__main__":
    results = main()