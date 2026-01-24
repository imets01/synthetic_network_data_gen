import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
import zipfile
import json
import joblib
import os
from typing import Optional, List, Dict, Any, Tuple
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


# --- WGAN-GP Specific Hyperparameters ---
DEFAULT_WGAN_PARAMS = {
    'critic_iterations': 5,  # Train the critic 5 times for every 1 generator training
    'lambda_gp': 10,         # The gradient penalty lambda
    'latent_dim': 20,
    'hidden_dim': 256,
    'num_layers_generator': 2,
    'num_layers_critic': 1,
    'batch_size': 128,
    'epochs': 2000,
    'g_lr': 5e-5,
    'd_lr': 5e-5,
    'weight_decay': 1e-6,
    'betas': (0.0, 0.999),
    'max_grad_norm': 1.0,
    'device': 'auto'
}


class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim, sequence_feature_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=condition_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc_init_h = nn.Linear(latent_dim + condition_dim, hidden_dim * num_layers)
        self.fc_init_c = nn.Linear(latent_dim + condition_dim, hidden_dim * num_layers)
        self.fc_out = nn.Linear(hidden_dim, sequence_feature_dim)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, noise, condition, seq_len):
        batch_size = noise.shape[0]
        combined_input = torch.cat([noise, condition], dim=1)
        h0 = self.fc_init_h(combined_input).view(self.num_layers, batch_size, self.hidden_dim)
        c0 = self.fc_init_c(combined_input).view(self.num_layers, batch_size, self.hidden_dim)
        initial_state = (h0, c0)
        condition_expanded = condition.unsqueeze(1).repeat(1, seq_len, 1)
        rnn_out, _ = self.rnn(condition_expanded, initial_state)
        output = torch.tanh(self.fc_out(rnn_out))
        return output


class Critic(nn.Module):
    def __init__(self, condition_dim, sequence_feature_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=sequence_feature_dim + condition_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, sequence, condition):
        seq_len = sequence.shape[1]
        condition_expanded = condition.unsqueeze(1).repeat(1, seq_len, 1)
        combined_input = torch.cat([sequence, condition_expanded], dim=2)
        _, (hn, cn) = self.rnn(combined_input)
        out = self.fc_out(hn[-1])
        return out


class QuicSequenceDataset(Dataset):
    def __init__(
        self,
        high_level_csv: str,
        low_level_zip_path: Optional[str] = None,
        low_level_dir: Optional[str] = None,
        folder_name_in_zip: str = 'separate_low_level_files',
        condition_cols: Optional[List[str]] = None,
        remove_negative_rows: bool = True
    ):
        # Load high-level data
        self.high_level_df = pd.read_csv(high_level_csv)
        self.high_level_df = self.high_level_df.replace([np.inf, -np.inf], np.nan).dropna(how='any')
        
        self.flow_ids_int = self.high_level_df['file_id'].copy()
        
        # Setup paths
        self.low_level_zip_path = low_level_zip_path
        self.low_level_dir = low_level_dir
        self.folder_name_in_zip = folder_name_in_zip
        self.remove_negative_rows = remove_negative_rows
        
        # Define default condition features
        default_condition_cols = [
            'implementation', 'connection_duration', 'version_negotiation_occurred', 
            'retry_occurred', 'migration_type', 'first_path_validation_response_latency', 
            'path_validation_initiated', 'packets_sent_client', 'packets_sent_server', 
            'handshake_duration', 'time_to_migration', 'migration_duration',
            'packets_before_migration', 'total_bidi_streams_client_init',
            'total_udi_streams_client_init'
        ]
        
        # Use provided condition columns or defaults
        to_keep = condition_cols if condition_cols is not None else default_condition_cols
        
        self.condition_features_df = self.high_level_df.copy()
        
        # Ensure all required columns exist, add with 0 if missing
        for col in to_keep:
            if col not in self.condition_features_df.columns:
                self.condition_features_df[col] = 0
        
        self.condition_features_df = self.condition_features_df[to_keep + ['file_id']]
        
        # Encode categorical columns
        categorical_cols = ['migration_type', 'implementation']
        existing_categorical_cols = [col for col in categorical_cols if col in self.condition_features_df.columns]
        for col in existing_categorical_cols:
            if self.condition_features_df[col].dtype == 'object':
                self.condition_features_df[col] = self.condition_features_df[col].astype('category')
        
        categorical_cols_to_dummy = self.condition_features_df.select_dtypes(include=['category']).columns
        self.condition_features_df = pd.get_dummies(
            self.condition_features_df, 
            columns=categorical_cols_to_dummy, 
            prefix=categorical_cols_to_dummy
        )
        
        # Initialize scalers
        self.condition_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.sequence_scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Storage for sequences
        self.preloaded_sequences = []
        self.flow_id_to_path = {}
        self.sequence_columns = None
        
        # Load sequences from either ZIP or directory
        self._load_sequences()
        
    def _load_sequences(self):
        """Load and validate sequences from ZIP or directory"""
        target_flow_ids = set(self.flow_ids_int.astype(str))
        temp_sequences = []
        temp_valid_ids = []
        dropped_count = 0
        
        if self.low_level_zip_path:
            self._load_from_zip(target_flow_ids, temp_sequences, temp_valid_ids, dropped_count)
        elif self.low_level_dir:
            self._load_from_directory(target_flow_ids, temp_sequences, temp_valid_ids, dropped_count)
        else:
            raise ValueError("Either low_level_zip_path or low_level_dir must be provided")
        
        if dropped_count > 0:
            print(f"Dropped {dropped_count} sequences containing negative values.")
        
        # Update dataframe with valid sequences only
        self.flow_ids = pd.Series(temp_valid_ids)
        self.condition_features_df = self.condition_features_df[
            self.condition_features_df['file_id'].isin(temp_valid_ids)
        ].reset_index(drop=True)
        
        # Fit scalers and transform data
        if temp_sequences:
            self._fit_scalers_and_transform(temp_sequences)
        else:
            print("Error: No valid sequence data found!")
    
    def _load_from_zip(self, target_flow_ids, temp_sequences, temp_valid_ids, dropped_count):
        """Load sequences from ZIP file"""
        with zipfile.ZipFile(self.low_level_zip_path, 'r') as zf:
            zip_names = zf.namelist()
            target_prefix = f"{self.folder_name_in_zip}/"
            
            potential_files = {}
            for name in zip_names:
                if name.startswith(target_prefix) and name.endswith('.csv'):
                    relative_name = name[len(target_prefix):]
                    flow_id_match = relative_name.split('_')[0]
                    if flow_id_match in target_flow_ids:
                        potential_files[int(flow_id_match)] = name
                        target_flow_ids.discard(flow_id_match)
            
            potential_ids = list(potential_files.keys())
            
            for flow_id in tqdm(potential_ids, desc="Loading & Validating Sequences"):
                file_name_in_zip = potential_files[flow_id]
                
                with zf.open(file_name_in_zip) as f:
                    try:
                        df = pd.read_csv(f)
                        if 'frame_number' in df.columns:
                            df = df.drop('frame_number', axis=1)
                        vals = df.values
                        
                        # Check for negative values
                        if self.remove_negative_rows and (vals < 0).any():
                            dropped_count += 1
                            continue
                        
                        if self.sequence_columns is None:
                            self.sequence_columns = df.columns.tolist()
                        
                        temp_sequences.append(vals)
                        temp_valid_ids.append(flow_id)
                        self.flow_id_to_path[flow_id] = file_name_in_zip
                    
                    except Exception as e:
                        print(f"Error processing {file_name_in_zip}: {e}")
                        continue
    
    def _load_from_directory(self, target_flow_ids, temp_sequences, temp_valid_ids, dropped_count):
        """Load sequences from directory"""
        csv_files = [f for f in os.listdir(self.low_level_dir) if f.endswith('.csv')]
        
        for csv_file in tqdm(csv_files, desc="Loading & Validating Sequences"):
            flow_id_match = csv_file.split('_')[0]
            
            if flow_id_match not in target_flow_ids:
                continue
            
            try:
                flow_id = int(flow_id_match)
                df = pd.read_csv(os.path.join(self.low_level_dir, csv_file))
                
                if 'frame_number' in df.columns:
                    df = df.drop('frame_number', axis=1)
                vals = df.values
                
                # Check for negative values
                if self.remove_negative_rows and (vals < 0).any():
                    dropped_count += 1
                    continue
                
                if self.sequence_columns is None:
                    self.sequence_columns = df.columns.tolist()
                
                temp_sequences.append(vals)
                temp_valid_ids.append(flow_id)
                self.flow_id_to_path[flow_id] = csv_file
            
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue
    
    def _fit_scalers_and_transform(self, temp_sequences):
        """Fit scalers on all data and transform sequences"""
        full_sequence_data = np.concatenate(temp_sequences, axis=0)
        
        if np.isnan(full_sequence_data).any() or np.isinf(full_sequence_data).any():
            print("WARNING: NaN or INF found in the unscaled sequence data.")
        
        print(f"Unscaled Sequence Data Min/Max: {full_sequence_data.min():.4f} / {full_sequence_data.max():.4f}")
        self.sequence_scaler.fit(full_sequence_data)
        
        # Fit condition scaler
        condition_features_for_scaling = self.condition_features_df.drop('file_id', axis=1)
        self.scaled_conditions = self.condition_scaler.fit_transform(condition_features_for_scaling)
        self.scaled_conditions = torch.FloatTensor(self.scaled_conditions)
        
        # Transform sequences to tensors
        for unscaled_seq in tqdm(temp_sequences, desc="Scaling Sequences to Tensors"):
            scaled_seq = self.sequence_scaler.transform(unscaled_seq)
            if scaled_seq.min() < -1.01 or scaled_seq.max() > 1.01:
                print("WARNING: SCALED DATA OUT OF RANGE!")
            self.preloaded_sequences.append(torch.FloatTensor(scaled_seq))

    def __len__(self):
        return len(self.preloaded_sequences)

    def __getitem__(self, idx):
        scaled_sequence = self.preloaded_sequences[idx]
        scaled_condition = self.scaled_conditions[idx]
        return {
            'sequence': scaled_sequence,
            'condition': scaled_condition
        }


def collate_fn(batch):
    """Collate function for padding variable-length sequences"""
    sequences = [item['sequence'] for item in batch]
    conditions = torch.stack([item['condition'] for item in batch])
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return {'sequence': padded_sequences, 'condition': conditions}


def compute_gradient_penalty(critic, real_samples, fake_samples, condition, device):
    """Compute gradient penalty for WGAN-GP"""
    alpha = torch.randn(real_samples.size(0), 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Temporarily disable cuDNN for the double backward pass
    with torch.backends.cudnn.flags(enabled=False):
        d_interpolates = critic(interpolates, condition)
    
    fake = torch.ones(d_interpolates.size(), device=device)
    
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class WGANHandler: 
    def __init__(self, device: str = "auto"):
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # State storage
        self.dataset: Optional[QuicSequenceDataset] = None
        self.dataloader: Optional[DataLoader] = None
        self.generator: Optional[Generator] = None
        self.critic: Optional[Critic] = None
        self.g_optimizer: Optional[torch.optim.Optimizer] = None
        self.d_optimizer: Optional[torch.optim.Optimizer] = None
        
        # Training params
        self._training_params = DEFAULT_WGAN_PARAMS.copy()
        
        # Stats
        self.training_losses = {'generator': [], 'critic': []}
    
    def load_data(
        self,
        high_level_csv: str,
        low_level_zip_path: Optional[str] = None,
        low_level_dir: Optional[str] = None,
        folder_name_in_zip: str = 'separate_low_level_files',
        condition_cols: Optional[List[str]] = None,
        remove_negative_rows: bool = True,
        batch_size: Optional[int] = None,
        num_workers: int = 4
    ) -> Dict[str, Any]:
        print("Loading data...")
        
        if batch_size is not None:
            self._training_params['batch_size'] = batch_size
        
        self.dataset = QuicSequenceDataset(
            high_level_csv=high_level_csv,
            low_level_zip_path=low_level_zip_path,
            low_level_dir=low_level_dir,
            folder_name_in_zip=folder_name_in_zip,
            condition_cols=condition_cols,
            remove_negative_rows=remove_negative_rows
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self._training_params['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers
        )
        
        stats = {
            'num_sequences': len(self.dataset),
            'condition_dim': self.dataset.scaled_conditions.shape[1],
            'sequence_feature_dim': len(self.dataset.sequence_columns),
            'sequence_columns': self.dataset.sequence_columns,
            'batch_size': self._training_params['batch_size']
        }
        
        print(f"  Loaded {stats['num_sequences']} sequences")
        print(f"  Condition dimension: {stats['condition_dim']}")
        print(f"  Sequence feature dimension: {stats['sequence_feature_dim']}")
        print(f"  Batch size: {stats['batch_size']}")
        
        return stats
    
    def create_models(
        self,
        latent_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_layers_generator: Optional[int] = None,
        num_layers_critic: Optional[int] = None,
        g_lr: Optional[float] = None,
        d_lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        betas: Optional[Tuple[float, float]] = None
    ) -> Tuple[Generator, Critic]:
        if self.dataset is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Update params
        if latent_dim is not None:
            self._training_params['latent_dim'] = latent_dim
        if hidden_dim is not None:
            self._training_params['hidden_dim'] = hidden_dim
        if num_layers_generator is not None:
            self._training_params['num_layers_generator'] = num_layers_generator
        if num_layers_critic is not None:
            self._training_params['num_layers_critic'] = num_layers_critic
        if g_lr is not None:
            self._training_params['g_lr'] = g_lr
        if d_lr is not None:
            self._training_params['d_lr'] = d_lr
        if weight_decay is not None:
            self._training_params['weight_decay'] = weight_decay
        if betas is not None:
            self._training_params['betas'] = betas
        
        print("Creating models...")
        
        condition_dim = self.dataset.scaled_conditions.shape[1]
        sequence_feature_dim = len(self.dataset.sequence_columns)
        
        # Create models
        self.generator = Generator(
            latent_dim=self._training_params['latent_dim'],
            condition_dim=condition_dim,
            sequence_feature_dim=sequence_feature_dim,
            hidden_dim=self._training_params['hidden_dim'],
            num_layers=self._training_params['num_layers_generator']
        ).to(self.device)
        
        self.critic = Critic(
            condition_dim=condition_dim,
            sequence_feature_dim=sequence_feature_dim,
            hidden_dim=self._training_params['hidden_dim'],
            num_layers=self._training_params['num_layers_critic']
        ).to(self.device)
        
        # Create optimizers
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self._training_params['g_lr'],
            betas=self._training_params['betas'],
            weight_decay=self._training_params['weight_decay']
        )
        
        self.d_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self._training_params['d_lr'],
            betas=self._training_params['betas'],
            weight_decay=self._training_params['weight_decay']
        )
        
        print(f"  Generator: latent_dim={self._training_params['latent_dim']}, "
              f"hidden_dim={self._training_params['hidden_dim']}, "
              f"num_layers={self._training_params['num_layers_generator']}")
        print(f"  Critic: hidden_dim={self._training_params['hidden_dim']}, "
              f"num_layers={self._training_params['num_layers_critic']}")
        print(f"  Optimizers: G_LR={self._training_params['g_lr']}, "
              f"D_LR={self._training_params['d_lr']}, "
              f"weight_decay={self._training_params['weight_decay']}")
        
        return self.generator, self.critic
    
    def train(
        self,
        epochs: Optional[int] = None,
        critic_iterations: Optional[int] = None,
        lambda_gp: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 200,
        progress_callback: Optional[callable] = None
    ):
        if self.generator is None or self.critic is None:
            raise ValueError("Models not created. Call create_models() first.")
        
        if self.dataloader is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Update params
        if epochs is not None:
            self._training_params['epochs'] = epochs
        if critic_iterations is not None:
            self._training_params['critic_iterations'] = critic_iterations
        if lambda_gp is not None:
            self._training_params['lambda_gp'] = lambda_gp
        if max_grad_norm is not None:
            self._training_params['max_grad_norm'] = max_grad_norm
        
        print(f"Training WGAN-GP for {self._training_params['epochs']} epochs...")
        print(f"  Critic iterations: {self._training_params['critic_iterations']}")
        print(f"  Lambda GP: {self._training_params['lambda_gp']}")
        print(f"  Max grad norm: {self._training_params['max_grad_norm']}")
        
        self.generator.train()
        self.critic.train()
        
        for epoch in range(self._training_params['epochs']):
            epoch_g_losses = []
            epoch_d_losses = []
            
            for i, batch in enumerate(tqdm(
                self.dataloader, 
                desc=f"Epoch {epoch+1}/{self._training_params['epochs']}"
            )):
                real_sequences = batch['sequence'].to(self.device)
                conditions = batch['condition'].to(self.device)
                batch_size, seq_len, _ = real_sequences.shape
                
                # Train Critic
                for _ in range(self._training_params['critic_iterations']):
                    self.d_optimizer.zero_grad()
                    
                    noise = torch.randn(
                        batch_size, 
                        self._training_params['latent_dim'], 
                        device=self.device
                    )
                    fake_sequences = self.generator(noise, conditions, seq_len)
                    
                    real_validity = self.critic(real_sequences, conditions)
                    fake_validity = self.critic(fake_sequences.detach(), conditions)
                    
                    gradient_penalty = compute_gradient_penalty(
                        self.critic, 
                        real_sequences.data, 
                        fake_sequences.data, 
                        conditions.data, 
                        self.device
                    )
                    
                    d_loss = (-torch.mean(real_validity) + 
                             torch.mean(fake_validity) + 
                             self._training_params['lambda_gp'] * gradient_penalty)
                    
                    d_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), 
                        max_norm=self._training_params['max_grad_norm']
                    )
                    
                    self.d_optimizer.step()
                
                # Train Generator
                self.g_optimizer.zero_grad()
                gen_sequences = self.generator(noise, conditions, seq_len)
                g_loss = -torch.mean(self.critic(gen_sequences, conditions))
                g_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(), 
                    max_norm=self._training_params['max_grad_norm']
                )
                
                self.g_optimizer.step()
                
                epoch_g_losses.append(g_loss.item())
                epoch_d_losses.append(d_loss.item())
            
            # Calculate average losses
            avg_g_loss = np.mean(epoch_g_losses)
            avg_d_loss = np.mean(epoch_d_losses)
            
            self.training_losses['generator'].append(avg_g_loss)
            self.training_losses['critic'].append(avg_d_loss)
            
            print(f"Epoch [{epoch+1}/{self._training_params['epochs']}], "
                  f"Critic Loss: {avg_d_loss:.4f}, Generator Loss: {avg_g_loss:.4f}")
            
            # Progress callback
            if progress_callback is not None:
                progress_callback(epoch + 1, avg_g_loss, avg_d_loss)
            
            # Save checkpoint
            if checkpoint_dir and (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_dir, epoch)
        
        print("Training complete!")
    
    def generate_samples(
        self,
        num_samples: int,
        conditions: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
        return_numpy: bool = True
    ) -> np.ndarray:

        if self.generator is None:
            raise ValueError("Generator not created. Call create_models() first.")
        
        if self.dataset is None:
            raise ValueError("No dataset loaded. Cannot generate without scalers.")
        
        self.generator.eval()
        
        print(f"Generating {num_samples} synthetic samples...")
        
        with torch.no_grad():
            # Get conditions
            if conditions is None:
                # Randomly sample conditions from dataset
                indices = np.random.choice(len(self.dataset), num_samples, replace=True)
                conditions = self.dataset.scaled_conditions[indices].to(self.device)
            else:
                conditions = conditions.to(self.device)
            
            # Determine sequence length
            if seq_len is None:
                # Use median sequence length from dataset
                seq_lengths = [len(seq) for seq in self.dataset.preloaded_sequences]
                seq_len = int(np.median(seq_lengths))
            
            # Generate noise
            noise = torch.randn(num_samples, self._training_params['latent_dim'], device=self.device)
            
            # Generate sequences
            generated = self.generator(noise, conditions, seq_len)
            
            # Move to CPU and convert to numpy
            generated = generated.cpu().numpy()
            
            # Reshape for inverse transform
            original_shape = generated.shape
            generated_reshaped = generated.reshape(-1, generated.shape[-1])
            
            # Inverse transform
            generated_unscaled = self.dataset.sequence_scaler.inverse_transform(generated_reshaped)
            generated_unscaled = generated_unscaled.reshape(original_shape)
        
        print(f"  Generated sequences with shape: {generated_unscaled.shape}")
        
        if return_numpy:
            return generated_unscaled
        else:
            return torch.tensor(generated_unscaled)
    
    def save_model(
        self,
        save_dir: str,
        save_optimizers: bool = True
    ) -> None:

        if self.generator is None or self.critic is None:
            raise ValueError("No models to save. Train the model first.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Saving models to {save_dir}...")
        
        # Save models
        torch.save(self.generator.state_dict(), os.path.join(save_dir, "generator.pth"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pth"))
        
        # Save optimizers
        if save_optimizers and self.g_optimizer is not None and self.d_optimizer is not None:
            torch.save(self.g_optimizer.state_dict(), os.path.join(save_dir, "g_optimizer.pth"))
            torch.save(self.d_optimizer.state_dict(), os.path.join(save_dir, "d_optimizer.pth"))
        
        # Save scalers
        if self.dataset is not None:
            joblib.dump(self.dataset.sequence_scaler, os.path.join(save_dir, "sequence_scaler.gz"))
            joblib.dump(self.dataset.condition_scaler, os.path.join(save_dir, "condition_scaler.gz"))
            
            # Save sequence columns
            with open(os.path.join(save_dir, "sequence_columns.json"), 'w') as f:
                json.dump(self.dataset.sequence_columns, f)
        
        # Save training params
        with open(os.path.join(save_dir, "training_params.json"), 'w') as f:
            json.dump(self._training_params, f, indent=4)
        
        # Save training losses
        with open(os.path.join(save_dir, "training_losses.json"), 'w') as f:
            json.dump(self.training_losses, f, indent=4)
        
        print("  Models saved successfully!")
    
    def load_model(
        self,
        save_dir: str,
        load_optimizers: bool = False,
        load_dataset_info: bool = True
    ) -> 'WGANHandler':
        print(f"Loading models from {save_dir}...")
        
        # Load training params
        with open(os.path.join(save_dir, "training_params.json"), 'r') as f:
            self._training_params = json.load(f)
        
        # Load dataset info if requested
        if load_dataset_info:
            # Create a minimal dataset object for scalers
            class MinimalDataset:
                pass
            
            self.dataset = MinimalDataset()
            self.dataset.sequence_scaler = joblib.load(os.path.join(save_dir, "sequence_scaler.gz"))
            self.dataset.condition_scaler = joblib.load(os.path.join(save_dir, "condition_scaler.gz"))
            
            with open(os.path.join(save_dir, "sequence_columns.json"), 'r') as f:
                self.dataset.sequence_columns = json.load(f)
        
        # Determine dimensions from scalers or params
        if load_dataset_info:
            condition_dim = self.dataset.condition_scaler.n_features_in_
            sequence_feature_dim = len(self.dataset.sequence_columns)
        else:
            # Will need to be set manually
            raise ValueError("Cannot determine model dimensions without dataset info")
        
        # Create models
        self.generator = Generator(
            latent_dim=self._training_params['latent_dim'],
            condition_dim=condition_dim,
            sequence_feature_dim=sequence_feature_dim,
            hidden_dim=self._training_params['hidden_dim'],
            num_layers=self._training_params['num_layers_generator']
        ).to(self.device)
        
        self.critic = Critic(
            condition_dim=condition_dim,
            sequence_feature_dim=sequence_feature_dim,
            hidden_dim=self._training_params['hidden_dim'],
            num_layers=self._training_params['num_layers_critic']
        ).to(self.device)
        
        # Load weights
        self.generator.load_state_dict(torch.load(
            os.path.join(save_dir, "generator.pth"),
            map_location=self.device
        ))
        self.critic.load_state_dict(torch.load(
            os.path.join(save_dir, "critic.pth"),
            map_location=self.device
        ))
        
        # Load optimizers if requested
        if load_optimizers:
            self.g_optimizer = torch.optim.Adam(self.generator.parameters())
            self.d_optimizer = torch.optim.Adam(self.critic.parameters())
            
            g_opt_path = os.path.join(save_dir, "g_optimizer.pth")
            d_opt_path = os.path.join(save_dir, "d_optimizer.pth")
            
            if os.path.exists(g_opt_path) and os.path.exists(d_opt_path):
                self.g_optimizer.load_state_dict(torch.load(g_opt_path, map_location=self.device))
                self.d_optimizer.load_state_dict(torch.load(d_opt_path, map_location=self.device))
        
        # Load training losses if available
        losses_path = os.path.join(save_dir, "training_losses.json")
        if os.path.exists(losses_path):
            with open(losses_path, 'r') as f:
                self.training_losses = json.load(f)
        
        print("  Models loaded successfully!")
        return self
    
    def save_checkpoint(self, checkpoint_dir: str, epoch: int) -> None:
        """Save training checkpoint"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'training_losses': self.training_losses,
            'training_params': self._training_params
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved at epoch {epoch+1}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:

        if not os.path.exists(checkpoint_path):
            print("No checkpoint found, starting from scratch")
            return 0
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.training_losses = checkpoint.get('training_losses', {'generator': [], 'critic': []})
        self._training_params = checkpoint.get('training_params', self._training_params)
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"  Resumed from epoch {start_epoch}")
        return start_epoch
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get current training parameters"""
        return self._training_params.copy()
    
    def get_training_losses(self) -> Dict[str, List[float]]:
        """Get training loss history"""
        return self.training_losses.copy()
