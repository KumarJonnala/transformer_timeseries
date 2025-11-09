import os
import torch
import numpy as np
import pickle
from load_data import WESADDataset
from transformer_model import TabTransformer
from train_test_loop import train_model, evaluate_model
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import recall_score, precision_score, f1_score

def main():
    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else
                         "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    config = {
        'lr_rate': 0.0001,
        'n_epochs': 100,
        'early_stopping_patience': 5,
        'early_stopping_min_delta': 0.001,
    }

    # Load dataset

    # DATASET_PATH = '~/Library/Mobile Documents/com~apple~CloudDocs/Phoenix/OVGU/HiWi2/Tasks/10_WESAD/WESAD.nosync'
    # ds = WESADDataset(DATASET_PATH)

    ds = WESADDataset()
    pickle_path = os.path.join('/Users/kumar/Desktop/Projects/transformer_timeseries', 'wesad_raw.pkl')
    with open(pickle_path, 'rb') as f:
        saved_data = pickle.load(f)
        ds.data = saved_data['data']
        ds.labels = saved_data['labels']

    # Subject bins for LOOCV
    subject_counts = {
        'S2': 440, 'S3': 445, 'S4': 449, 'S5': 460, 'S6': 458, 'S7': 457,
        'S8': 460, 'S9': 456, 'S10': 476, 'S11': 465, 'S13': 461, 'S14': 464,
        'S15': 464, 'S16': 463, 'S17': 476
    }

    # Calculate subject indices
    subject_indices = {}
    start = 0
    for subject, count in subject_counts.items():
        end = start + count
        subject_indices[subject] = [start, end]
        start = end
    print(subject_counts, '\n', subject_indices)

    # LOOCV training and evaluation
    loocv_results = []
    for test_subject, test_range in subject_indices.items():
        print(f"\n{'='*60}")
        print(f"Testing on Subject: {test_subject}")
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
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        
        print(f"Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")
        
        # Initialize model
        model = TabTransformer(
            num_features=6, 
            num_classes=2, 
            dim_embedding=64, 
            num_heads=4, 
            num_layers=4,
            dropout=0.1
        ).to(device)
        
        # Training setup
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.5, patience=3)
        
        # Train model
        history, best_model_state = train_model(
            model=model,
            train_dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=config['n_epochs'],
            device=device,
            patience=config['early_stopping_patience'],
            min_delta=config['early_stopping_min_delta']
        )
        
        # Evaluate best model
        test_acc, test_preds, test_labels = evaluate_model(
            model=best_model_state['model_state_dict'],
            test_loader=test_loader,
            device=device
        )

        test_preds = np.array(test_preds)
        test_labels = np.array(test_labels)
        
        metrics = {
            'accuracy': test_acc * 100,  # Convert to percentage
            'f1_score': f1_score(test_labels, test_preds, average='weighted'),
            'precision': precision_score(test_labels, test_preds, average='weighted'),
            'recall': recall_score(test_labels, test_preds, average='weighted')
        }
        
        loocv_results.append({
            'subject': test_subject,
            'metrics': metrics,
            'history': history
        })
        
        print(f"Subject {test_subject} Results:")
        print(f"Accuracy: {metrics['accuracy']:.2f}%")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")

    # Print final summary
    print("\nMean Metrics Across All Subjects:")
    print("=" * 60)
    metrics_to_average = ['accuracy', 'f1_score', 'precision', 'recall']
    for metric in metrics_to_average:
        mean_value = np.mean([r['metrics'][metric] for r in loocv_results])
        std_value = np.std([r['metrics'][metric] for r in loocv_results])
        print(f"Mean {metric}: {mean_value:.2f} ± {std_value:.2f}")
        
    # Save results
    results_path = os.path.join(os.path.dirname(__file__), 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            'config': config,
            'results': loocv_results
        }, f)
    print(f"\nResults saved to: {results_path}")

    return loocv_results

if __name__ == "__main__":
    results = main()