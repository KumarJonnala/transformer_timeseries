# Read WESAD Dataset

import os
import pickle
import numpy as np
from scipy.signal import resample
from scipy.stats import mode
import torch
from torch.utils.data import Dataset

class WESADDataset(Dataset):
    def __init__(self, data_path, window_size=128, overlap=0.0):
        self.data_path = data_path
        self.window_size = window_size
        self.overlap = overlap
        self.signal_names = ['ACC','Resp','EDA','Temp','ECG','EMG']

        self.data = None
        self.labels = None
        self.subjects = None
        if data_path is not None:
            self.data, self.labels, self.subjects = self.load_dataset()  
    
    def load_dataset(self):
        subjects = [f'S{i}' for i in range(1, 18) if i not in [1, 12]]  # S1 and S12 are not available (Problem with sensors)
        all_data = []
        all_labels = []
        all_subjects = []
        
        orig_fs = 700
        target_fs = 32
        
        for subject in subjects:
            subj_dir = os.path.join(self.data_path, subject)
            data_file = os.path.join(subj_dir, f'{subject}.pkl')
            
            if not os.path.exists(data_file):
                print(f'Warning: {data_file} does not exist')
                continue
            
            try:
                with open(data_file, 'rb') as f:
                    raw = torch.load(f) if self.data_path.endswith('.pt') else pickle.load(f, encoding='latin1')
                
                # Extract chest data and label
                chest_data = raw['signal']['chest']
                labels = raw['label']
                
                # Process signals
                signals = []
                for name in self.signal_names:
                    if name in chest_data:
                        sig = chest_data[name]
                        
                        # Handle multi-dimensional signals (like ACC with x,y,z components)
                        if len(sig.shape) > 1:
                            if name == 'ACC':
                                # For accelerometer, compute magnitude from 3D components
                                if sig.shape[1] == 3:  # x, y, z components
                                    sig = np.sqrt(np.sum(sig**2, axis=1))  # Magnitude
                                else:
                                    sig = sig.flatten()
                            else:
                                sig = sig.flatten()
                        
                        # Resample signal
                        sig_resampled = resample(sig, int(len(sig) * target_fs / orig_fs))
                        signals.append(sig_resampled)
                    else:
                        print(f'Warning: {name} missing for {subject}')
                
                if len(signals) != len(self.signal_names):
                    print(f'Skipping {subject} due to missing modalities')
                    continue
                
                # Ensure all signals have same length
                min_len = min(map(len, signals))
                signals = [s[:min_len] for s in signals]
                signal_matrix = np.stack(signals, axis=1)
                
                # Resample labels
                labels_resampled = resample(labels, min_len)
                labels_resampled = np.round(labels_resampled).astype(int)
                
                # Create sliding windows
                win_data, win_labels = self.create_windows(signal_matrix, labels_resampled)
                
                all_data.extend(win_data)
                all_labels.extend(win_labels)
                all_subjects.extend([subject]*len(win_data))
                
                print(f'Loaded {len(win_data)} sliding windows for {subject}')
                
            except Exception as e:
                print(f'Error processing {subject}: {e}')
                continue
        
        return np.array(all_data), np.array(all_labels), np.array(all_subjects)
    
    def create_windows(self, data, labels):
        step = int(self.window_size * (1 - self.overlap))
        windows = []
        window_labels = []
        
        for start in range(0, data.shape[0] - self.window_size + 1, step):
            end = start + self.window_size
            label_window = labels[start:end]
            
            # Handle newer scipy versions
            mode_result = mode(label_window, keepdims=True)
            lbl = int(mode_result[0][0])
            
            if lbl == 1:  # Baseline
                windows.append(data[start:end])
                window_labels.append(0)
            elif lbl == 2:  # Stress
                windows.append(data[start:end])
                window_labels.append(1)
        
        return windows, window_labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
