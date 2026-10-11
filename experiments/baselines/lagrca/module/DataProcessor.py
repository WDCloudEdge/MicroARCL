import pandas as pd
import torch
from torch_geometric.data import Batch
from collections import defaultdict


class DataProcessor:
    def __init__(self, df, config, window_size=5, stride=1, window_stride=2,small_window_size=5):
        
        
        self.config = config
        self.window_size = window_size
        self.stride = stride
        self.window_stride = window_stride
        self.small_window_size = small_window_size

        self.df = df
       
        
        self.reorder_columns()
        
        self.feature_columns = self.df.columns[1:] 
        
       
        self.save_metric_names_to_config()

        
    def reorder_columns(self):
        feature_columns = self.df.columns[1:]  # drop the timestamp column
        instance_name_map = defaultdict(list)  # k: instance name, v: list of full feature names
        for col in feature_columns:
            if "&" in col:
                _, instance = col.split("&", 1)
                instance_name_map[instance].append(col)

        ordered_feature_cols = []
        for instance in self.config.all_enum.keys():
            if instance in instance_name_map:
                ordered_feature_cols.extend(instance_name_map[instance])  # add all metric columns belonging to this instance
            else:
                raise ValueError(f"Instance {instance} not found in raw data columns!")

        self.df = pd.concat([self.df.iloc[:, [0]], self.df[ordered_feature_cols]], axis=1)
    
        self.instance_metric_count_dict = {inst: len(cols) for inst, cols in instance_name_map.items()}
        
    
        self.ordered_feature_columns = ordered_feature_cols
    
    def save_metric_names_to_config(self):
       
        self.config.ordered_feature_names = self.ordered_feature_columns

        self.config.instance_metric_names = {}
        start_idx = 0
        
       
        for instance in self.config.all_enum.keys():
            if instance in self.instance_metric_count_dict:
                metric_count = self.instance_metric_count_dict[instance]
                end_idx = start_idx + metric_count
                instance_metrics = self.ordered_feature_columns[start_idx:end_idx]
                
                self.config.instance_metric_names[instance] = {
                    'full_names': instance_metrics,  
                    'start_idx': start_idx,          
                    'end_idx': end_idx,              
                    'metric_count': metric_count     
                }
                
                start_idx = end_idx
        
   
        self.config.instance_metric_count_dict = self.instance_metric_count_dict
        
        
    def generate_patches_tensor(self,time_series, patch_size, stride=None):

        N, L = time_series.shape
        num_patches = (L - patch_size) // stride + 1

        
        patches = time_series.unfold(dimension=1, size=patch_size, step=stride)
        return patches

        

    def generate_windows(self):
       
        feature_data = torch.tensor(self.df.iloc[:, 1:].values, dtype=torch.float)  # [T, num_features]

        windows = []
        for start_idx in range(0, feature_data.size(0) - self.window_size, self.stride):
            x_window = feature_data[start_idx: start_idx + self.window_size]       # input window
           
            x_window = x_window.permute(1,0)            
            x_window = self.generate_patches_tensor(x_window, patch_size=self.small_window_size, stride=self.window_stride)  # [num_patches, num_features, patch_size]
            y_window = feature_data[start_idx + self.window_size]                  # prediction target (single-step)
            windows.append((x_window, y_window))
        return windows
    def generate_windows_gnet(self):
       
        feature_data = torch.tensor(self.df.iloc[:, 1:].values, dtype=torch.float)  # [T, num_features]

        windows = []
        for start_idx in range(0, feature_data.size(0) - self.window_size, self.stride):
            # [window_size, num_features]
            x_window = feature_data[start_idx: start_idx + self.window_size]

            # convert to (num_features, window_size)
            x_window = x_window.T  # [N, T]

            # add channel dim C_in=1 and expand the batch dim
            # final shape: (1, 1, N, T)
            x_window = x_window.unsqueeze(0)  

            # single-step prediction target: [num_features]
            y_window = feature_data[start_idx + self.window_size]

            windows.append((x_window, y_window))
        return windows
    def generate_windows_normal(self):
        """Generate sliding-window data pairs (past, future); output shape (B, C_in, N, T)."""
        feature_data = torch.tensor(self.df.iloc[:, 1:].values, dtype=torch.float)  # [T, num_features]

        windows = []
        for start_idx in range(0, feature_data.size(0) - self.window_size, self.stride):
            # [window_size, num_features]
            x_window = feature_data[start_idx: start_idx + self.window_size]

            # convert to (num_features, window_size)
            x_window = x_window.T  # [N, T]
          

            # single-step prediction target: [num_features]
            y_window = feature_data[start_idx + self.window_size]

            windows.append((x_window, y_window))
        return windows

    def generate_multi_windows(self, num_windows):
        """
        Generate an input sequence of num_windows windows, each of length window_size, sliding between windows by window_stride.

        :param num_windows: number of windows in the input sequence
        :return: list[(x_window, y_window)], where x_window has shape [num_windows, num_features, window_size],
                 y_window is the single-step target after the last window [num_features]
        """
        if num_windows < 1:
            raise ValueError("num_windows must be a positive integer")

        feature_data = torch.tensor(self.df.iloc[:, 1:].values, dtype=torch.float)
        total_steps = feature_data.size(0)

        required_span = (num_windows - 1) * self.window_stride + self.window_size
        max_start = total_steps - required_span - 1

        if max_start < 0:
            return []

        windows = []
        for start_idx in range(0, max_start + 1, self.stride):
            window_stack = []
            for win_idx in range(num_windows):
                segment_start = start_idx + win_idx * self.window_stride
                segment_end = segment_start + self.window_size
                window_stack.append(feature_data[segment_start:segment_end].T)

            x_window = torch.stack(window_stack, dim=0)
            y_idx = start_idx + required_span
            y_window = feature_data[y_idx]

            windows.append((x_window, y_window))

        return windows


from torch.utils.data import Dataset

class TimeWindowDataset(Dataset):
    def __init__(self,data_processor, config):
        self.data_processor = data_processor
        self.config = config
        

        #self.windows = self.data_processor.generate_windows_normal()  # generate sliding-window data pairs
        self.windows = self.data_processor.generate_multi_windows(5)  # generate sliding-window data pairs



    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x_window, y_window = self.windows[idx]
        
        instance_names = list(map(str, self.config.all_enum.keys()))
        return x_window, instance_names, y_window,