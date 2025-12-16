import os
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import recall_score, precision_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.trial import Trial

# import module files
from config import Config
from data.loader import WESADDataset
from models.Tabtransformer import TabTransformer
from train_test_loop import train_model, evaluate_model


def objective(trial: Trial, device, ds, subject_indices):
    """
    Objective function for Optuna hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        device: PyTorch device (CPU/GPU)
        ds: WESADDataset instance
        subject_indices: Dictionary mapping subjects to their data indices
    
    Returns:
        Mean F1 score across all LOOCV folds
    """
    # Suggest hyperparameters
    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])
    lr = trial.suggest_categorical("lr", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    
    # Aggregated results across LOOCV folds
    fold_accuracies = []
    fold_f1_scores = []
    fold_mcc_scores = []
    fold_metrics = {}  # Store per-subject metrics
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: Testing hyperparameters:")
    print(f"dropout={dropout:.3f}")
    print(f"lr={lr:.6f}, batch_size={batch_size}")
    print(f"{'='*60}")
    
    for fold_idx, (test_subject, test_range) in enumerate(subject_indices.items()):
        print(f"\nFold {fold_idx + 1}: Testing on Subject {test_subject}")
        
        # LOOCV split logic
        test_indices = list(range(test_range[0], test_range[1]))
        train_val_indices = []
        for subj, subj_range in subject_indices.items():
            if subj != test_subject:
                train_val_indices.extend(range(subj_range[0], subj_range[1]))
        
        # Split train_val into train and validation (80/20 split)
        train_indices, val_indices = train_test_split(
            train_val_indices, 
            test_size=0.2, 
            random_state=Config.RANDOM_SEED,
            stratify=ds.labels[train_val_indices]  # Stratified split to maintain class distribution
        )
        
        # Normalize data using only training data statistics
        train_data = ds.data[train_indices]
        val_data = ds.data[val_indices]
        test_data = ds.data[test_indices]
        
        train_means = train_data.mean(axis=(0, 1))
        train_stds = train_data.std(axis=(0, 1))
        epsilon = 1e-8
        
        train_data_normalized = (train_data - train_means[None, None, :]) / (train_stds[None, None, :] + epsilon)
        val_data_normalized = (val_data - train_means[None, None, :]) / (train_stds[None, None, :] + epsilon)
        test_data_normalized = (test_data - train_means[None, None, :]) / (train_stds[None, None, :] + epsilon)
        
        # Create dataloaders
        train_ds = TensorDataset(
            torch.FloatTensor(train_data_normalized), 
            torch.LongTensor(ds.labels[train_indices])
        )
        val_ds = TensorDataset(
            torch.FloatTensor(val_data_normalized), 
            torch.LongTensor(ds.labels[val_indices])
        )
        test_ds = TensorDataset(
            torch.FloatTensor(test_data_normalized), 
            torch.LongTensor(ds.labels[test_indices])
        )
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        print(f"  Train samples: {len(train_indices)}, Val samples: {len(val_indices)}, Test samples: {len(test_indices)}")
        
        # Initialize model with trial hyperparameters
        model = TabTransformer(
            num_features=Config.NUM_FEATURES,
            num_classes=Config.NUM_CLASSES,
            dim_embedding=Config.EMBEDDING_DIM,
            num_heads=Config.NUM_HEADS,
            num_layers=Config.NUM_LAYERS,
            dropout=dropout
        ).to(device)
        
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Train model with validation set
        history, best_model_state = train_model(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,  
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=Config.NUM_EPOCHS,
            device=device,
            patience=Config.EARLY_STOPPING_PATIENCE,
            min_delta=Config.EARLY_STOPPING_MIN_DELTA
        )
        
        # Evaluate on test set
        test_acc, test_preds, test_labels = evaluate_model(
            model=model,
            best_model_state=best_model_state,
            test_loader=test_loader,
            device=device
        )
        
        test_preds = np.array(test_preds)
        test_labels = np.array(test_labels)

        # Calculate metrics for this fold
        test_f1 = f1_score(test_labels, test_preds, average='weighted')
        test_confusion = confusion_matrix(test_labels, test_preds)
        test_precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
        test_recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
        test_mcc = matthews_corrcoef(test_labels, test_preds)
        
        fold_accuracies.append(test_acc)
        fold_f1_scores.append(test_f1)
        fold_mcc_scores.append(test_mcc)

        # Store per-subject metrics
        fold_metrics[test_subject] = {
            'accuracy': test_acc,
            'confusion_matrix': test_confusion,
            'f1_score': test_f1,
            'precision': test_precision,
            'recall': test_recall,
            'mcc': test_mcc
        }
        
        print(f"  {test_subject}: Test Acc={test_acc:.2f}%, Test F1={test_f1:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}, MCC={test_mcc:.4f}")
        print(f"  Confusion Matrix:\n{test_confusion}")
        print(f"Test Labels: {test_labels}")
        print(f"Test Pred: {test_preds}")

        # Report intermediate value for pruning (use current mean F1)
        current_mean_f1 = np.mean(fold_f1_scores)
        trial.report(current_mean_f1, fold_idx)
        
        # Prune trial if it's not promising
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # Return mean metrics across all folds
    mean_acc = np.mean(fold_accuracies)
    mean_f1 = np.mean(fold_f1_scores)
    mean_mcc = np.mean(fold_mcc_scores)
    
    print(f"\nTrial {trial.number} Results:")
    print(f"  Mean Test Accuracy: {mean_acc:.2f}%")
    print(f"  Mean Test F1 Score: {mean_f1:.4f}")
    print(f"  Mean Test MCC Score: {mean_mcc:.4f}")
    
    # Store F1 score and per-subject metrics as user attributes
    trial.set_user_attr("mean_f1_score", mean_f1)
    trial.set_user_attr("mean_accuracy", mean_acc)
    trial.set_user_attr("mean_mcc", mean_mcc)  # Add this line
    trial.set_user_attr("fold_metrics", fold_metrics)
    
    return mean_f1


def main():
    """
    Main function for hyperparameter tuning using Optuna with LOOCV.
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    ds = WESADDataset(None)
    with open(Config.PICKLE_PATH, 'rb') as f:
        saved_data = pickle.load(f)
        ds.data = saved_data['data']
        ds.labels = saved_data['labels']
    
    print(f"Dataset loaded: {len(ds)} samples")

    # Calculate subject indices
    subject_indices = {}
    start = 0
    for subject, count in Config.SUBJECT_COUNTS.items():
        end = start + count
        subject_indices[subject] = [start, end]
        start = end
    
    print(f"Subjects: {list(subject_indices.keys())}")
    print(f"Total subjects: {len(subject_indices)}")

    # Create Optuna study
    print("\n" + "="*60)
    print("Starting Hyperparameter Optimization with Optuna")
    print("Validation split: 80/20 (train/val) from remaining subjects")
    print("="*60)
    
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        study_name="TabTransformer_WESAD_LOOCV"
    )

    # Optimize with lambda to pass additional arguments
    study.optimize(
        lambda trial: objective(trial, device, ds, subject_indices),
        n_trials=Config.N_TRIALS,
        timeout=Config.TIMEOUT,  
        show_progress_bar=True
    )

    # Print optimization results
    print("\n" + "="*60)
    print("Hyperparameter Optimization Complete")
    print("="*60)
    
    print("\nBest Trial:")
    print(f"  Trial Number: {study.best_trial.number}")
    best_f1 = study.best_trial.user_attrs.get('mean_f1_score', None)
    best_acc = study.best_trial.user_attrs.get('mean_accuracy', None)
    best_mcc = study.best_trial.user_attrs.get('mean_mcc', None)
    print(f"  Mean F1 Score: {best_f1:.4f}" if best_f1 is not None else "  Mean F1 Score: N/A")
    print(f"  Mean Accuracy: {best_acc:.2f}%" if best_acc is not None else "  Mean Accuracy: N/A")
    print(f"  Mean MCC Score: {best_mcc:.4f}" if best_mcc is not None else "  Mean MCC Score: N/A")

    print("\nBest Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Print per-subject metrics for best trial
    print("\nPer-Subject Metrics (Best Trial):")
    print("-" * 95)
    fold_metrics = study.best_trial.user_attrs.get('fold_metrics', {})
    if fold_metrics:
        print(f"{'Subject':<10} {'Accuracy':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12} {'MCC':<12}")
        print("-" * 95)
        for subject, metrics in fold_metrics.items():
            print(f"{subject:<10} {metrics['accuracy']:>10.2f}% {metrics['f1_score']:>11.4f} "
                  f"{metrics['precision']:>11.4f} {metrics['recall']:>11.4f} {metrics['mcc']:>11.4f}")
        print("-" * 95)
    else:
        print("No per-subject metrics available.")
    
    # Print top 5 trials
    print("\nTop 5 Trials:")
    print("-" * 60)
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else 0, reverse=True)
    for i, trial in enumerate(trials_sorted[:5], 1):
        if trial.value is not None:
            f1 = trial.user_attrs.get('mean_f1_score', None)
            acc = trial.user_attrs.get('mean_accuracy', None)
            f1_str = f"{f1:.4f}" if isinstance(f1, float) else "N/A"
            acc_str = f"{acc:.2f}%" if isinstance(acc, float) else "N/A"
            print(f"{i}. Trial {trial.number}: F1={f1_str}, Acc={acc_str}")
            print(f"   Params: {trial.params}")
    
    # Save study results
    results_dir = os.path.join(Config.CHECKPOINT_DIR, "optuna_results")
    os.makedirs(results_dir, exist_ok=True)
 
    # Save best hyperparameters
    best_params_path = os.path.join(results_dir, "best_hyperparameters.txt")
    with open(best_params_path, 'w') as f:
        f.write("Best Hyperparameters:\n")
        f.write("="*60 + "\n")
        f.write(f"Trial Number: {study.best_trial.number}\n")
        best_f1 = study.best_trial.user_attrs.get('mean_f1_score', None)
        best_acc = study.best_trial.user_attrs.get('mean_accuracy', None)
        best_mcc = study.best_trial.user_attrs.get('mean_mcc', None)
        f1_str = f"{best_f1:.4f}" if isinstance(best_f1, float) else "N/A"
        acc_str = f"{best_acc:.2f}%" if isinstance(best_acc, float) else "N/A"
        mcc_str = f"{best_mcc:.4f}" if isinstance(best_mcc, float) else "N/A"
        f.write(f"Mean F1 Score: {f1_str}\n")
        f.write(f"Mean Accuracy: {acc_str}\n")
        f.write(f"Mean MCC Score: {mcc_str}\n\n")
        f.write("Parameters:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\n")
        
        # Write per-subject metrics
        f.write("\n" + "="*60 + "\n")
        f.write("Per-Subject Metrics:\n")
        f.write("-" * 95 + "\n")
        fold_metrics = study.best_trial.user_attrs.get('fold_metrics', {})
        if fold_metrics:
            f.write(f"{'Subject':<10} {'Accuracy':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12} {'MCC':<12}\n")
            f.write("-" * 95 + "\n")
            for subject, metrics in fold_metrics.items():
                f.write(f"{subject:<10} {metrics['accuracy']:>10.2f}% {metrics['f1_score']:>11.4f} "
                       f"{metrics['precision']:>11.4f} {metrics['recall']:>11.4f} {metrics['mcc']:>11.4f}\n")
            
            # Write confusion matrices
            f.write("\n" + "="*60 + "\n")
            f.write("Confusion Matrices per Subject:\n")
            f.write("-" * 60 + "\n")
            for subject, metrics in fold_metrics.items():
                f.write(f"\n{subject}:\n")
                f.write(str(metrics['confusion_matrix']) + "\n")
    
    print(f"\nResults saved to: {results_dir}")
    print(f"Best hyperparameters saved to: {best_params_path}")
    
    # Optionally save the study for later analysis
    study_path = os.path.join(results_dir, "optuna_study.pkl")
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    print(f"Study object saved to: {study_path}")
    
    return study



if __name__ == "__main__":
    study = main()