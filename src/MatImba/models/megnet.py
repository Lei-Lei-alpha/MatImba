import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Set2Set
from typing import Optional, Dict, Any

from .fds import FDS
from .layers import MegnetModule, ShiftedSoftplus

ATOMIC_NUMBERS = 95

class MEGNet(nn.Module):
    """
    MEGNet (MatErials Graph Network) implementation with optional FDS integration.
    
    This model performs message passing between nodes (atoms), edges (bonds), 
    and global state attributes to predict material properties.

    Args:
        load_from (str, optional): Path to a checkpoint to load model weights from.
        edge_input_shape (int, optional): Dimensionality of edge features.
        node_input_shape (int, optional): Dimensionality of node features.
        state_input_shape (int, optional): Dimensionality of global state features.
        node_embedding_size (int, optional): Size of atom embeddings (if not using pre-featurized inputs). Defaults to 16.
        embedding_size (int, optional): Hidden dimension size for message passing blocks. Defaults to 32.
        n_blocks (int, optional): Number of MEGNet message passing blocks. Defaults to 3.
        fds (bool, optional): Whether to enable Feature Distribution Smoothing. Defaults to False.
        device (torch.device, optional): Calculation device (CPU/GPU).
        **fds_params: Additional keyword arguments passed to the FDS module.
    """
    def __init__(
        self, 
        load_from: Optional[str] = None,
        edge_input_shape: Optional[int] = None,
        node_input_shape: Optional[int] = None,
        state_input_shape: Optional[int] = None,
        node_embedding_size: int = 16,
        embedding_size: int = 32,
        n_blocks: int = 3, 
        fds: bool = False,
        device: Optional[torch.device] = None,
        **fds_params
    ):
        super().__init__()
        
        # Device Setup
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Initialization
        if load_from:
            self._init_from_checkpoint(load_from)
        else:
            self._init_new_model(
                edge_input_shape, node_input_shape, state_input_shape,
                node_embedding_size, embedding_size, n_blocks, fds, fds_params
            )

        # Build Layers
        self.blocks = nn.ModuleList()
        for _ in range(self.n_blocks - 1):
            self.blocks.append(MegnetModule(
                self.embedding_size, self.embedding_size, self.embedding_size
            ))

        self.se = Set2Set(self.embedding_size, 1)
        self.sv = Set2Set(self.embedding_size, 1)
        
        self.hiddens = nn.Sequential(
            nn.Linear(5 * self.embedding_size, self.embedding_size),
            ShiftedSoftplus(),
            nn.Linear(self.embedding_size, self.embedding_size // 2),
            ShiftedSoftplus(),
        )
        self.linear = nn.Linear(self.embedding_size // 2, 1)

        # FDS Initialization
        if self.fds:
            # Feature dim is input size of the final linear layer
            self.FDS = FDS(
                feature_dim = self.embedding_size // 2,
                **self.fds_params
            )

        self.to(self.device)

    def _init_new_model(self, edge_dim, node_dim, state_dim, node_emb_size, emb_size, n_blocks, fds, fds_params):
        """Initializes a fresh model architecture."""
        self.edge_input_shape = edge_dim
        self.node_input_shape = node_dim
        self.state_input_shape = state_dim
        self.node_embedding_size = node_emb_size
        self.embedding_size = emb_size
        self.n_blocks = n_blocks
        self.fds = fds
        self.fds_params = fds_params or {}

        self.embedded = self.node_input_shape is None
        if self.embedded:
            self.node_input_shape = self.node_embedding_size
            self.emb = nn.Embedding(ATOMIC_NUMBERS, self.node_embedding_size)

        self.m1 = MegnetModule(
            self.edge_input_shape, self.node_input_shape,
            self.state_input_shape, inner_skip=True,
            embed_size=self.embedding_size
        )

    def _init_from_checkpoint(self, path):
        """Restores model state and architecture from a file."""
        checkpoint = torch.load(path, map_location=self.device)["model"]
        # Restore init params
        for key, val in checkpoint['init_params'].items():
            setattr(self, key, val)
            
        # Re-establish architecture based on restored params
        if 'emb.weight' in checkpoint['states']:
            self.embedded = True
            self.node_input_shape = self.node_embedding_size
            self.emb = nn.Embedding(ATOMIC_NUMBERS, self.node_embedding_size)
        else:
            self.embedded = False

        self.m1 = MegnetModule(
            self.edge_input_shape, self.node_input_shape,
            self.state_input_shape, inner_skip=True,
            embed_size=self.embedding_size
        )

        self.fds_params = checkpoint.get('fds_params', {})
        # State dict will be loaded automatically if names match, 
        # but we need to load it explicitly here because __init__ constructs layers.
        # HOWEVER, standard practice is to construct first, then load_state_dict.
        # The logic here assumes layers are created in __init__ after this helper returns.
        # We will load weights at the END of __init__ if load_from was passed.
        self._pending_state_dict = checkpoint['states']

    def _compute_features(self, x, edge_index, edge_attr, state,
                batch, bond_batch):
        '''compute features before last linear layer'''
        if self.embedded:
            x = self.emb(x).squeeze()
        else:
            x = x.float()

        x, edge_attr, state = self.m1(x, edge_index, edge_attr,
                                      state, batch, bond_batch)
        for block in self.blocks:
            x, edge_attr, state = block(x, edge_index, edge_attr,
                                        state, batch, bond_batch)
        x_pool = self.sv(x, batch)
        edge_attr_pool = self.se(edge_attr, bond_batch)

        # Handle potential mismatch if batch size 1 or other edge cases causing padding needs
        # Though Set2Set usually handles batching correctly.
        # Original code had padding logic:
        tmp_shape = x_pool.shape[0] - edge_attr_pool.shape[0]
        if tmp_shape > 0:
            edge_attr_pool = F.pad(edge_attr_pool, (0, 0, 0, tmp_shape), value=0.0)

        tmp = torch.cat((x_pool, edge_attr_pool, state), 1)
        encoding = self.hiddens(tmp)
        return encoding
        
    def forward(
        self, 
        x, 
        edge_index, 
        edge_attr, 
        state,
        batch, 
        bond_batch, 
        targets: Optional[torch.Tensor] = None, 
        epoch: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass of the MEGNet model.

        Args:
            x (torch.Tensor): Node features (or atomic indices if embedded).
            edge_index (torch.Tensor): Graph connectivity (2, num_edges).
            edge_attr (torch.Tensor): Edge features.
            state (torch.Tensor): Global state features.
            batch (torch.Tensor): Batch index for nodes.
            bond_batch (torch.Tensor): Batch index for edges.
            targets (torch.Tensor, optional): Ground truth values (used for FDS smoothing).
            epoch (int, optional): Current training epoch (used for FDS scheduling).

        Returns:
            torch.Tensor: Model prediction (scalar per graph).
        """
        
        features = self._compute_features(
            x, edge_index, edge_attr, state,
            batch, bond_batch
        )
        
        # FDS Smoothing (Dynamics Stage)
        if self.training and self.fds:
            # Note: FDS.smooth requires targets. 
            # Trainer typically passes targets if using specialized loop,
            # otherwise update_fds handles the smoothing statistics globally.
            # If per-batch smoothing is desired during forward pass:
            features = self.FDS(features, targets, epoch)
        
        out = self.linear(features)
        return out.squeeze()

    def update_fds(self, train_loader, epoch: int):
        """
        **Call this ONCE per epoch** after training step.
        Auto-collects features & targets from full train_loader for FDS momentum update.
        """
        if not self.fds:
            return

        self.eval()
        with torch.no_grad():
            all_features = []
            all_targets = []
            for batch in train_loader:
                batch = batch.to(self.device)
                
                batch_features = self._compute_features(
                    batch.x, batch.edge_index, batch.edge_attr,
                    batch.state, batch.batch, batch.bond_batch
                )
                all_features.append(batch_features)
                all_targets.append(batch.y)
            
            # Concatenate all batches
            all_features = torch.cat(all_features, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # Update FDS statistics
            self.FDS.update(all_features, all_targets, epoch)
            
        self.train()