import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from typing import Optional, Tuple


class ShiftedSoftplus(nn.Module):
    """
    Applies the shifted Softplus function:
    
    .. math::
        y = ln(exp(x) + 1) - ln(2)
        
    This ensures the activation starts at 0 for x=0, which helps with convergence 
    in certain graph neural network architectures like MEGNet/SchNet.
    """
    def __init__(self):
        super().__init__()
        self.sp = nn.Softplus()
        self.shift = nn.Parameter(torch.log(torch.tensor([2.])), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the activation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated tensor with shifted baseline.
        """
        return self.sp(x) - self.shift


class MegnetModule(MessagePassing):
    """
    A MEGNet Block that updates edge, node, and global state features sequentially.
    
    This module implements the core message passing logic described in the MEGNet paper.
    It performs three distinct updates in order:
    1. **Edge Update**: Based on current edge attributes, connected node features, and global state.
    2. **Node Update**: Aggregates updated edge messages to update node features.
    3. **State Update**: Aggregates all updated node and edge features to update the global state.

    Args:
        edge_input_shape (int): Dimensionality of input edge features.
        node_input_shape (int): Dimensionality of input node features.
        state_input_shape (int): Dimensionality of input global state features.
        inner_skip (bool, optional): If True, applies the skip connection *after* non-linear projection (DenseNet style). If False, applies it *before* (ResNet style). Defaults to False.
        embed_size (int, optional): The hidden dimension size for all feature updates. Defaults to 32.
    """
    def __init__(self,
                 edge_input_shape: int,
                 node_input_shape: int,
                 state_input_shape: int,
                 inner_skip: bool = False,
                 embed_size: int = 32,
                 ):
        super().__init__(aggr="mean")
        self.inner_skip = inner_skip
        self.embed_size = embed_size

        # Edge update network (Phi_e)
        self.phi_e = nn.Sequential(
            nn.Linear(4 * embed_size, 2 * embed_size),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size, 2 * embed_size),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size),
            ShiftedSoftplus(),
        )

        # Global State update network (Phi_u)
        self.phi_u = nn.Sequential(
            nn.Linear(3 * embed_size, 2 * embed_size),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size, 2 * embed_size),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size),
            ShiftedSoftplus(),
        )

        # Node update network (Phi_v)
        self.phi_v = nn.Sequential(
            nn.Linear(3 * embed_size, 2 * embed_size),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size, 2 * embed_size),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size),
            ShiftedSoftplus(),
        )

        # Preprocessing projections to align dimensions
        self.preprocess_e = nn.Sequential(
            nn.Linear(edge_input_shape, 2 * embed_size),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size),
            ShiftedSoftplus(),
        )

        self.preprocess_v = nn.Sequential(
            nn.Linear(node_input_shape, 2 * embed_size),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size),
            ShiftedSoftplus(),
        )

        self.preprocess_u = nn.Sequential(
            nn.Linear(state_input_shape, 2 * embed_size),
            ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size),
            ShiftedSoftplus(),
        )

    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                edge_attr: torch.Tensor, 
                state: torch.Tensor, 
                batch: torch.Tensor, 
                bond_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes the MEGNet block update cycle.

        Args:
            x (torch.Tensor): Node features of shape (num_nodes, node_input_shape).
            edge_index (torch.Tensor): Graph connectivity of shape (2, num_edges).
            edge_attr (torch.Tensor): Edge features of shape (num_edges, edge_input_shape).
            state (torch.Tensor): Global state features of shape (batch_size, state_input_shape).
            batch (torch.Tensor): Batch vector mapping nodes to graphs (num_nodes,).
            bond_batch (torch.Tensor): Batch vector mapping edges to graphs (num_edges,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the updated 
            (node_features, edge_features, state_features).
        """
        # 1. Feature Preprocessing
        x_proj = self.preprocess_v(x)
        edge_attr_proj = self.preprocess_e(edge_attr)
        state_proj = self.preprocess_u(state)
        
        # 2. Skip Connection Handling
        if self.inner_skip:
            # Skip connection is established AFTER projection (Inner)
            x_skip = x_proj
            edge_attr_skip = edge_attr_proj
            state_skip = state_proj
            
            # Working variables are the projected ones
            x_curr, edge_attr_curr, state_curr = x_proj, edge_attr_proj, state_proj
        else:
            # Skip connection is established BEFORE projection (Outer/Residual)
            x_skip = x
            edge_attr_skip = edge_attr
            state_skip = state
            
            # Working variables are the projected ones
            x_curr, edge_attr_curr, state_curr = x_proj, edge_attr_proj, state_proj

        # 3. Edge Update Phase
        if torch.numel(bond_batch) > 0:
            # edge_updater calls self.edge_update internally with provided args
            edge_attr_curr = self.edge_updater(
                edge_index=edge_index, 
                x=x_curr, 
                edge_attr=edge_attr_curr, 
                state=state_curr, 
                bond_batch=bond_batch
            )
            
        # 4. Node Update Phase (Message Passing)
        # propagate calls self.message -> aggregate -> self.update
        x_curr = self.propagate(
            edge_index=edge_index, 
            x=x_curr, 
            edge_attr=edge_attr_curr, 
            state=state_curr, 
            batch=batch
        )
        
        # 5. Global State Update Phase
        # Aggregate updated nodes and edges to graph level
        u_v = global_mean_pool(x_curr, batch)
        
        # Calculate max batch index safely
        num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1
        u_e = global_mean_pool(edge_attr_curr, bond_batch, size=num_graphs)
        
        # Concatenate mean(edges), mean(nodes), and old state -> Update
        state_curr = self.phi_u(torch.cat((u_e, u_v, state_curr), 1))
        # 6. Apply Residual Add
        return x_curr + x_skip, edge_attr_curr + edge_attr_skip, state_curr + state_skip

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Constructs the message to be aggregated by the target node.
        
        In MEGNet, the message passed to the node is simply the updated edge attribute.
        
        Args:
            x_i (torch.Tensor): Features of target nodes.
            x_j (torch.Tensor): Features of source nodes.
            edge_attr (torch.Tensor): Updated edge attributes.

        Returns:
            torch.Tensor: The message (edge_attr).
        """
        return edge_attr

    def update(self, inputs: torch.Tensor, x: torch.Tensor, state: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Updates node features based on aggregated messages.
        
        Corresponds to Phi_v in the paper. Concatenates (mean_edge_msg, old_node, global_state).

        Args:
            inputs (torch.Tensor): Aggregated messages (mean of edge attributes) from neighbors.
            x (torch.Tensor): Current node features.
            state (torch.Tensor): Current global state features.
            batch (torch.Tensor): Batch vector for nodes.

        Returns:
            torch.Tensor: New node features.
        """
        return self.phi_v(torch.cat((inputs, x, state[batch, :]), 1))

    def edge_update(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor, 
                    state: torch.Tensor, bond_batch: torch.Tensor) -> torch.Tensor:
        """
        Updates edge features.
        
        Corresponds to Phi_e in the paper. Concatenates (src_node, dst_node, old_edge, global_state).

        Args:
            x_i (torch.Tensor): Target node features.
            x_j (torch.Tensor): Source node features.
            edge_attr (torch.Tensor): Current edge features.
            state (torch.Tensor): Global state features.
            bond_batch (torch.Tensor): Batch vector for edges.

        Returns:
            torch.Tensor: New edge features.
        """
        return self.phi_e(torch.cat((x_i, x_j, edge_attr, state[bond_batch, :]), 1))
