import pandas as pd
import numpy as np
import math
import random
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Helper Classes
class AttributeMatrixGenerator:
    """
    Transforms raw multivariate time-series into spatial-temporal correlation matrices 
    and target values for the Hybrid MSCVAE model.
    """
    def __init__(self, window_size=10, step=10):
        self.w = window_size # Time steps captured in a single matrix (temporal context)
        self.step = step     # Sliding window stride (step=w means no overlap)
        self.mean = None
        self.std = None

    def fit_scaler(self, train_dataframes):
        """
        Calculates the mean and standard deviation from the training data.
        These statistics are used to normalize the data (Z-score) before generating matrices.
        """
        if isinstance(train_dataframes, pd.DataFrame):
            train_dataframes = [train_dataframes]
            
        # Concat to calculate global statistics (mean, std)
        # Z-score normalization is crucial so features with larger magnitudes do not dominate the dot product calculations.
        full_train_df = pd.concat(train_dataframes, ignore_index=True)
        self.mean = full_train_df.mean()
        self.std = full_train_df.std() + 1e-6 # Added epsilon to prevent division by zero
        self.min_physical_vals = full_train_df.min()

    def generate(self, df):
        """
        Transforms a dataframe into normalized spatial-temporal matrices and extracts target values.
        Args:
            df (pd.DataFrame): Raw time-series dataframe.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - matrices: Tensor of shape (B, 1, N, N)
                - target_values: Tensor of shape (B, N)
        """
        if self.mean is None:
            raise ValueError("Execute .fit_scaler() first!")

        # Scaling data to mean=0, std=1
        data = (df - self.mean) / self.std
        # NaNs become 0 (which is the mean after scaling), preserving matrix stability
        values = np.nan_to_num(data.values) 
        
        matrices = []
        target_values = [] 

        if len(values) < self.w:
            # Return empty tuple if df is too small (i.e. smaller than window_size)
            return torch.empty(0), torch.empty(0)

        # Sliding window extraction
        for t in range(self.w, len(values), self.step):
            x_segment = values[t-self.w : t] # Shape: (window_size, n_features)
            
            # Matrix (Eq. 1 from paper) - Spatial-temporal inner product
            # Captures correlations between all pairs of sensors within the window
            # Kernel Trick can be applied here but the model already has a non-linear decoder and the anomaly detection get less sensitive.
            x_t = torch.tensor(x_segment, dtype=torch.float32).T # Shape: (n_features, window_size)
            m_t = torch.matmul(x_t, x_t.T) / self.w              # Shape: (n_features, n_features)
            matrices.append(m_t)
            
            # Extracts the exact raw values of the last timestamp in the window.
            # Used as the ground truth for the Hybrid MLP val_decoder.
            last_val = torch.tensor(x_segment[-1], dtype=torch.float32)
            target_values.append(last_val)
            
        if not matrices:
            return torch.empty(0), torch.empty(0)

        # Tuple: (Tensor of Matrices, Tensor of Values)
        # unsqueeze(1) adds a 'Channel' dimension (B, 1, N, N) required by PyTorch Conv2d layers.
        return torch.stack(matrices).unsqueeze(1), torch.stack(target_values)

class TemporalAttention(nn.Module):
    """
    Optimized Temporal Attention.
    Computes similarity scores using highly compressed semantic vectors (GAP),
    making the matrix multiplications hundreds of times faster, and then applies 
    these weights to the full spatial history.
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.scale = 5.0 
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, h_current, h_history):
        if not h_history:
            return h_current
        
        b, c, h, w = h_current.size()
        
        # Compute scores using Pooled Semantics
        # Reduces (B, 64, H, W) to (B, 1, 64)
        pool_curr = self.gap(h_current).view(b, -1).unsqueeze(1) 
        
        pool_hist = []
        for hist_tensor in h_history:
            pool_hist.append(self.gap(hist_tensor).view(b, -1))
        
        # Shape: (B, 64, History_Length)
        pool_hist = torch.stack(pool_hist, dim=-1) 
        
        # Calculate similarity scores (Dot product of size 64 instead of 36,000)
        scores = torch.bmm(pool_curr, pool_hist) / self.scale 
        weights = F.softmax(scores, dim=-1) # Shape: (B, 1, History_Length)
        
        # Apply the calculated weights to the FULL spatial history
        context = torch.stack(h_history, dim=0) # (History_Length, B, C, H, W)
        flat_hist = context.view(context.size(0), b, -1).permute(1, 2, 0) # (B, C*H*W, History_Length)
        
        weighted_hist = torch.bmm(weights, flat_hist.permute(0, 2, 1))
        
        # Reshape back to original spatial dimensions
        h_hat = weighted_hist.view(b, c, h, w)
        
        return h_hat

class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell.
    Purpose: Replaces fully-connected linear layers of a standard LSTM with 2D Convolutions.
    This is crucial because it preserves the 2D spatial structure (the sensor-to-sensor 
    correlations in the matrix) while simultaneously learning temporal dynamics.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Padding ensures the output spatial dimensions (H, W) match the input
        self.padding = kernel_size // 2
        
        # A single convolution layer calculates all 4 LSTM gates at once for efficiency.
        # The output channel size is 4 * hidden_dim to account for i, f, o, and g gates.
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate current input matrix and previous hidden state along the channel dimension
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Apply convolution to generate raw gate values
        combined_conv = self.conv(combined)
        
        # Split the convolved output into the 4 distinct LSTM gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply non-linearities: Sigmoid for gates (0 to 1), Tanh for candidate states (-1 to 1)
        i = torch.sigmoid(cc_i) # Input gate: what new info to keep
        f = torch.sigmoid(cc_f) # Forget gate: what old info to discard
        o = torch.sigmoid(cc_o) # Output gate: what to output
        g = torch.tanh(cc_g)    # Cell candidate: new potential memory
        
        # Update cell state (c_next) and hidden state (h_next)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class MSCVAE_Hybrid(nn.Module):
    """
    Multivariate Spatial-Temporal Convolutional Variational Autoencoder (Hybrid).
    
    Architecture:
    1. CNN Encoder (4 Layers): Compresses spatial correlation matrices deeply to accelerate temporal processing.
    2. VAE Core (mu, logvar): Learns a continuous normal distribution of the system's baseline state.
    3. Temporal Modeling (ConvLSTM + Attention): Captures the temporal evolution (inertia) of the compressed states.
    4. Dual Decoder (Hybrid mechanism): 
       - Route 1 (MLP): Reconstructs exact raw physical values (Optimized with Global Average Pooling).
       - Route 2 (CNN): Reconstructs correlation matrices to detect broken systemic relationships.
    """
    def __init__(self, n_features):
        super(MSCVAE_Hybrid, self).__init__()
        self.n_features = n_features
        # Latent dimension scaled by the square root of features to prevent over-compression
        latent_dim = round(math.sqrt(n_features))
        
        # Deep Spatial Compression 
        # 4-layer Convolutional Encoder. 
        # Reduces the spatial grid dramatically, making the subsequent ConvLSTM operations highly efficient.
        self.enc1 = nn.Conv2d(1, 16, 3, 2, 1)
        self.enc2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.enc3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.enc4 = nn.Conv2d(64, 64, 3, 2, 1) 
        
        # Dummy pass: Automatically calculates the flattened dimension size after convolutions.
        # This makes the architecture dynamic and adaptable to any number of sensors (n_features).
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_features, n_features)
            out = self.enc4(self.enc3(self.enc2(self.enc1(dummy))))
            self.flatten_dim = out.view(1, -1).size(1) 
            self.spatial_shape = out.shape[1:] 

        # Latent Space (VAE Core)
        # Projects the flattened CNN features into probabilistic mean and variance vectors
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder Matrix: Projects the latent vector back to the spatial shape required by ConvTranspose2d
        self.fc_decode_mat = nn.Linear(latent_dim, self.flatten_dim)
        
        # Global Average Pooling
        # Drastically reduces parameter count by converting 2D spatial maps into 1D semantic feature vectors,
        # preventing the MLP linear layers from creating millions of unnecessary weights.
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Decoder of Values (Hybrid Route 1)
        # MLP network designed to reconstruct raw sensor values.
        # Inputs: static latent representation (z) + the purely semantic channels from GAP.
        val_input_dim = latent_dim + 64 
        self.val_decoder = nn.Sequential(
            nn.Linear(val_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, n_features) 
        )

        # Temporal Modeling
        # ConvLSTM retains the 2D spatial correlations while learning temporal dependencies.
        self.clstm = ConvLSTMCell(64, 64, kernel_size=3, bias=True)
        self.attention = TemporalAttention(hidden_dim=64)
        
        # Convolutional Decoder (Hybrid Route 2)
        # 4-layer ConvTranspose to upsample the combined features back to the original N x N matrix size
        self.dec4 = nn.ConvTranspose2d(64, 64, 3, 2, 1, output_padding=1) 
        self.dec3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1)
        self.dec1 = nn.ConvTranspose2d(16, 1, 3, 2, 1, output_padding=1)
        
        # State initialization for temporal modules
        self.h_state = None; self.c_state = None; self.history = []; self.max_history = 5

    def reparameterize(self, mu, logvar):
        """
        Reparameterization Trick: Allows backpropagation through the stochastic sampling process.
        Adds Gaussian noise during training but acts deterministically during evaluation.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        else:
            return mu

    def forward(self, x):
        """
        Forward pass executing the Encoder, Temporal Memory, and Dual Decoders.
        """
        batch_size = x.size(0)
        
        # Encode (Deep Spatial Compression)
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))
        e4 = F.relu(self.enc4(e3))
        
        flat = e4.view(batch_size, -1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        z = self.reparameterize(mu, logvar)
        
        # Temporal Modeling (Runs on a highly compressed spatial footprint)
        if self.h_state is None or self.h_state.size(0) != batch_size:
            h, w = self.spatial_shape[1], self.spatial_shape[2]
            self.h_state = torch.zeros(batch_size, 64, h, w).to(x.device)
            self.c_state = torch.zeros(batch_size, 64, h, w).to(x.device)
            self.history = []

        h_next, c_next = self.clstm(e4, (self.h_state, self.c_state))
        h_att = self.attention(h_next, self.history)
        
        # Detach states to truncate Backpropagation Through Time (BPTT) and save memory
        self.h_state = h_att.detach()
        self.c_state = c_next.detach()
        self.history.append(self.h_state)
        if len(self.history) > self.max_history: self.history.pop(0)

        # Route 1: Raw Values Reconstructions
        # GAP summarizes the (64, H, W) temporal tensor into a purely semantic (64) vector.
        pooled_h_att = self.gap(h_att).view(batch_size, -1)
        z_temporal = torch.cat([z, pooled_h_att], dim=1) 
        recon_values = self.val_decoder(z_temporal)
        
        # Route 2: Matrices Reconstructions
        # Reconstructs matrix by adding current latent projection to the historical attention map
        z_dec = self.fc_decode_mat(z).view(batch_size, *self.spatial_shape)
        combined = z_dec + h_att 
        
        # Decode (Deep Spatial Expansion)
        d4 = F.relu(self.dec4(combined))
        d3 = F.relu(self.dec3(d4))
        d2 = F.relu(self.dec2(d3))
        recon_matrix = self.dec1(d2)
        
        # Bilinear interpolation ensures the output matrix exactly matches the input dimensions (N x N)
        # This acts as a safety net if max-pooling or strides lost dimension parity due to odd feature counts.
        if recon_matrix.shape != x.shape:
            recon_matrix = F.interpolate(recon_matrix, size=x.shape[2:], mode='bilinear', align_corners=False)

        return recon_matrix, recon_values, mu, logvar

    def loss_function(self, recon_matrix, x_matrix, recon_values, x_values, mu, logvar, alpha=5.0, beta=0.1):
        """
        Multitask Loss Function (Beta-VAE framework).
        Combines Matrix Error, Value Error, and Kullback-Leibler Divergence.
        """
        mse_mat_mean = F.mse_loss(recon_matrix, x_matrix, reduction='mean')
        mse_val_mean = F.mse_loss(recon_values, x_values, reduction='mean')
        
        # Dimensionality Balancing: 
        # A matrix has N*N elements, while the value vector has only N.
        # Multiplying by n_elements_matrix simulates a 'sum' reduction, ensuring that the
        # magnitude of the reconstruction loss aligns mathematically with the KLD (which uses sum).
        n_elements_matrix = x_matrix.shape[2] * x_matrix.shape[3] 
        
        MSE_Mat_scaled = mse_mat_mean * n_elements_matrix
        MSE_Val_scaled = mse_val_mean * n_elements_matrix 
        
        # KLD acts as a regularizer, forcing the latent space to approximate a standard normal distribution.
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total Loss Calculation:
        # alpha: Weighs the importance of reconstructing raw values vs matrices.
        # beta: Controls the strength of the KLD regularizer. Lower beta (<1) prevents 'Posterior Collapse'.
        total_loss = MSE_Mat_scaled + (alpha * MSE_Val_scaled) + (beta * KLD)
        
        return total_loss

class SPOT:
    """
    Streaming Peaks-Over-Threshold (SPOT) Algorithm.
    Purpose: Dynamically calculates anomaly thresholds based on Extreme Value Theory (EVT).
    Instead of assuming a normal (Gaussian) distribution of errors, SPOT models the 
    'tail' of the distribution (the extreme reconstruction errors) using a Generalized 
    Pareto Distribution (GPD). This allows for robust, mathematically sound thresholding 
    that adapts to non-linear and heavy-tailed error distributions typical in VAEs.
    """
    def __init__(self, q=1e-4):
        # q (proba): The desired probability of false alarms. A lower 'q' means a 
        # stricter threshold, resulting in fewer anomalies being flagged.
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def fit(self, init_data, data):
        """
        Loads the initial calibration data (from the training phase) and the 
        data to be monitored. Handles multiple data types (list, numpy, pandas).
        """
        if isinstance(data, list): self.data = np.array(data)
        elif isinstance(data, np.ndarray): self.data = data
        elif isinstance(data, pd.Series): self.data = data.values
        else: return
        if isinstance(init_data, list): self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray): self.init_data = init_data
        elif isinstance(init_data, pd.Series): self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) and (init_data < 1) and (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else: return

    def initialize(self, level=0.98, min_extrema=False, verbose=True):
        """
        Calibration Phase: Establishes the base parameters using normal data.
        It sorts the initial data and sets a basic threshold (e.g., the 98th percentile).
        Values above this are considered 'peaks' and are used to fit the GPD.
        """
        if min_extrema:
            self.init_data = -self.init_data
            self.data = -self.data
            self.level = 1 - level
        level = level - math.floor(level)
        n_init = self.init_data.size
        S = np.sort(self.init_data)
        self.init_threshold = S[int(level * n_init)]
        
        # 'peaks' are the extreme reconstruction errors that exceed the initial threshold
        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init
            
        # Fits the Generalized Pareto Distribution to the peaks
        g, s, l = self._grimshaw()
        # Calculates the final strict threshold based on the desired false alarm probability (q)
        self.extreme_quantile = self._quantile(g, s)
        
        if verbose:
            print('Extreme quantile (probability = %s): %s' % (self.proba, self.extreme_quantile))

    def _rootsFinder(fun, jac, bounds, npoints, method):
        """Helper function for Grimshaw's trick: Finds roots of the derivative equation."""
        from scipy.optimize import minimize
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            if step == 0: bounds, step = (0, 1e-4), 1e-5
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)
            
        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j
            
        opt = minimize(lambda X: objFun(X, fun, jac), X0,
                       method='L-BFGS-B',
                       jac=True, bounds=[bounds] * len(X0))
        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        """Calculates the log-likelihood of the GPD given shape (gamma) and scale (sigma) parameters."""
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * math.log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + math.log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        """
        Grimshaw's Trick: An efficient algorithm to find the Maximum Likelihood Estimation (MLE) 
        for the parameters of the Generalized Pareto Distribution (gamma and sigma) 
        based on the observed peaks.
        """
        def u(s): return 1 + np.log(s).mean()
        def v(s): return np.mean(1 / s)
        def w(Y, t):
            s = 1 + t * Y
            us = u(s); vs = v(s)
            return us * vs - 1
        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s); vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us
            
        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()
        a = -1 / YM
        if abs(a) < 2 * epsilon: epsilon = abs(a) / n_points
        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)
        
        left_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                       lambda t: jac_w(self.peaks, t),
                                       (a + epsilon, -epsilon),
                                       n_points, 'regular')
        right_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                        lambda t: jac_w(self.peaks, t),
                                        (b, c),
                                        n_points, 'regular')
        zeros = np.concatenate((left_zeros, right_zeros))
        
        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)
        
        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = SPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll
                
        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        """
        Computes the final anomaly threshold (extreme quantile) using the fitted GPD parameters 
        and the predefined false alarm probability (q).
        """
        r = self.n * self.proba / self.Nt
        if gamma != 0: return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else: return self.init_threshold - sigma * math.log(r)


class MSCVAE:
    """
    Main wrapper class for the MSCVAE anomaly detection pipeline.
    Purpose: Acts as the high-level API orchestrating the entire lifecycle: 
    data preprocessing (matrix generation), model training, dynamic threshold 
    calculation (SPOT).
    """
    def __init__(self, n_features=None, window_size=10, stride=1, device=None, seed=42):
        self.seed = seed
        # Enforce reproducibility right at initialization
        self.set_deterministic(self.seed)
        
        self.n_features = n_features
        self.window_size = window_size
        self.stride = stride
        
        # Hardware selection: automatically defaults to GPU if available for faster tensor operations
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        self.model = None
        # Instantiates the generator that will transform flat series into spatial-temporal inputs
        self.generator = AttributeMatrixGenerator(window_size=self.window_size, step=self.stride)
        
        self.threshold = None
        # Gain acts as a manual sensitivity tuner applied on top of the SPOT statistical threshold
        self.gain = 1.0 
    
    def set_deterministic(self, seed=42):
        """
        Fixes random seeds across all underlying libraries (Python, NumPy, PyTorch).
        Deep learning involves stochastic processes (weight initialization, batch shuffling).
        Locking the seed ensures that experiments, debugging, and anomaly scores are 100% 
        reproducible across different executions on the same machine.
        """
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # Ensures reproducibility in multi-GPU setups
            
        # Forces CuDNN to use deterministic convolution algorithms. 
        # Prevents slight precision variations caused by underlying hardware optimizations.
        torch.backends.cudnn.deterministic = True
        # Disables auto-tuner that searches for the fastest convolution algorithm, 
        # prioritizing consistency over maximum speed.
        torch.backends.cudnn.benchmark = False

    def fit(self, train_data, epochs=50, batch_size=128, lr=1e-3, gain=1.0, verbose=True):
        """
        Main training orchestration pipeline.
        Handles data preprocessing, model weight optimization, and statistical threshold calibration.
        """
        self.gain = gain
        # Standardize input to a list of DataFrames to support training on multiple disjoint periods
        if isinstance(train_data, pd.DataFrame):
            train_data = [train_data]
            
        # Fit scaler
        if verbose: print("Fitting scaler...")
        # Fits Z-score scaler globally on normal data to ensure stable inner products later
        self.generator.fit_scaler(train_data)
        
        # Prepare data
        if verbose: print("Generating training data...")
        train_matrices = []
        train_values = []
        
        # If n_features was not set, infer it dynamically from the dataset
        if self.n_features is None:
            self.n_features = train_data[0].shape[1]
            
        for df in train_data:
            # Transforms flat multivariate series into spatial-temporal matrices and target values
            t_mat, t_val = self.generator.generate(df)
            if t_mat.nelement() > 0:
                train_matrices.append(t_mat)
                train_values.append(t_val)
        
        if not train_matrices:
             raise ValueError("No training data generated. Check window_size and data length!")

        final_train_matrix = torch.cat(train_matrices, dim=0)
        final_train_values = torch.cat(train_values, dim=0)
        
        # DataLoader shuffles the batches. This breaks sequence bias and ensures the VAE 
        # learns robust global representations rather than just memorizing local trends.
        train_loader = DataLoader(
            TensorDataset(final_train_matrix, final_train_values), 
            batch_size=batch_size, 
            shuffle=True
        )
        
        # Initialize Model
        self.model = MSCVAE_Hybrid(n_features=self.n_features).to(self.device)
        # Adam optimizer is used due to its adaptive learning rate, which is highly 
        # effective for training the distinct components of a Hybrid VAE simultaneously.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Training Phase
        self.model.train()
        if verbose: print(f"Starting training on {self.device} for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0
            for batch_matrix, batch_values in train_loader:
                x_mat = batch_matrix.to(self.device)
                x_val = batch_values.to(self.device)
                
                # Standard PyTorch Backprop Loop
                optimizer.zero_grad()
                recon_mat, recon_val, mu, logvar = self.model(x_mat)
                
                # Calculates multitask loss (Matrix MSE + Value MSE + KLD Regularization)
                loss = self.model.loss_function(recon_mat, x_mat, recon_val, x_val, mu, logvar)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            epoch_duration = time.time() - epoch_start_time

            if verbose and (epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 5 == 0):
                print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(train_loader.dataset):.4f} | Time: {epoch_duration:.2f}s")

        # Post-Training: Threshold Calibration (POT)
        if verbose: print("Calculating threshold...")
        # Evaluates the model on the normal training data to establish the baseline error distribution
        train_scores = self._get_anomaly_scores(final_train_matrix)
        
        # Applies Extreme Value Theory (SPOT algorithm) to find the mathematical upper limit 
        # of normal reconstruction errors, providing a strict, unsupervised alarm threshold.
        self.threshold = self._pot_eval(train_scores)
        
        if verbose:
            print(f"Base Threshold (POT): {self.threshold:.6f}")
            print(f"Gain: {self.gain}")
            print(f"Final Threshold: {self.threshold * self.gain:.6f}")
            
        # Gain acts as a manual sensitivity multiplier (e.g., gain=1.2 makes alarms 20% less sensitive)
        self.threshold = self.threshold * self.gain

    def _get_anomaly_scores(self, data_tensor, batch_size=128):
        """
        Calculates the raw anomaly score (phi) for each timestamp based purely on 
        the spatial-temporal matrix reconstruction error.
        The matrix represents the core physical relationships between sensors. 
        If the correlation breaks, the system is fundamentally anomalous.
        """
        self.model.eval()
        # reduction='none' allows us to compute the error for each sample individually 
        # instead of averaging the whole batch.
        mse_loss = nn.MSELoss(reduction='none')
        scores = []
        
        loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            # Reset temporal states to prevent leakage between completely different inferences
            self.model.h_state = None
            self.model.c_state = None
            self.model.history = []
            
            for (batch_input,) in loader:
                x = batch_input.to(self.device).float()
                recon_mat, _, _, _ = self.model(x)
                
                # Sums the squared error over all channels, height, and width (dims 1, 2, 3).
                # The result is a 1D array where each element is the total error of one temporal window.
                loss = mse_loss(recon_mat, x).sum(dim=(1, 2, 3)).cpu().numpy()
                scores.extend(loss)
                
        return np.array(scores)

    def _pot_eval(self, init_score, q=1e-4, level=0.02):
        """
        Wrapper for the SPOT (Peaks-Over-Threshold) algorithm initialization.
        Finds the exact mathematical threshold (extreme_quantile) above which 
        a reconstruction error is considered a true anomaly, rather than just noise.
        """
        lms = level
        while True:
            try:
                s = SPOT(q)
                s.fit(init_score, init_score)
                # Attempts to fit the Pareto distribution. If the data is too smooth or 
                # lacks distinct peaks, the Scipy optimization might fail (raise Exception).
                s.initialize(level=lms, min_extrema=False, verbose=False)
            except Exception:
                # Fallback mechanism: Gradually lowers the definition of what constitutes a 'peak' 
                # until the optimization algorithm converges successfully.
                lms = lms * 0.999
            else:
                break
        return s.extreme_quantile

    def predict(self, df_test, timestamps=None, batch_size=128):
        """
        Inference pipeline for new unseen data.
        Returns a dictionary mapping timestamps to their respective anomaly scores (phi).
        """
        if self.model is None:
            raise ValueError("Model not trained. Call .fit() first!")

        self.model.eval()

        # Filter columns to match the generator's expected features
        df_test = df_test[self.generator.mean.index]
        
        # Transform raw test data into correlation matrices
        try:
            tensor_matrices, _ = self.generator.generate(df_test)
        except ValueError as e:
            print(f"Generation error: {e}")
            return {}
            
        if tensor_matrices.nelement() == 0:
            print(f"Test dataframe too small for window {self.generator.w}.")
            return {}
            
        # Get the raw anomaly score (reconstruction error) for each matrix
        all_scores = self._get_anomaly_scores(tensor_matrices, batch_size=batch_size)
        
        # Time Alignment
        # The generator uses a sliding window (w) and a step size (s).
        # This means the score calculated for matrix M_t actually represents the state 
        # of the system at the *end* of that window. We must map the score back to the 
        # correct original timestamp to avoid time-shift errors in production.
        w = self.generator.w
        s = self.generator.step
        
        if timestamps is not None:
            if hasattr(timestamps, 'values'):
                ts_values = timestamps.values
            else:
                ts_values = np.array(timestamps)
                
            # 'valid_indices' mimics the loop in the generator to find which timestamps 
            # correspond to the end of each generated matrix.
            valid_indices = range(w, len(df_test) + 1, s)
            
            # Truncate to the smallest length to prevent IndexError in case of minor dimension mismatches
            min_len = min(len(all_scores), len(valid_indices))
            final_scores = all_scores[:min_len]
            final_indices = valid_indices[:min_len]
            
            final_timestamps = []
            for idx in final_indices:
                # Basic protection for index out of bounds
                if idx < len(ts_values):
                     final_timestamps.append(ts_values[idx])
                elif idx == len(ts_values):
                    # If index hits exactly the length, grab the last available timestamp
                    final_timestamps.append(ts_values[-1])
            
            final_scores = final_scores[:len(final_timestamps)]
            
            return {
                'timestamp': final_timestamps,
                'phi': final_scores
            }
        else:
            return {
                'timestamp': np.arange(len(all_scores)), # Dummy timestamps if none provided
                'phi': all_scores
            }
