import torch
from torch import nn
import torch.nn.functional as F
from typing import List

from src.model.mask_gcn_layer import MaskGCNLayer


class MaskGCNStack(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        cheb_k,
        weight_embed_dim,
        num_nodes,
        config,
        inverse_mask,
        num_layers=1,
    ):
        super(MaskGCNStack, self).__init__()
        assert num_layers >= 1, "At least one GCN layer in the Encoder."
        self.input_dim = dim_in
        self.input_out = dim_out
        self.num_layers = num_layers
        self.weight_embed_dim = weight_embed_dim
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(
            MaskGCNLayer(
                dim_in,
                dim_out,
                cheb_k,
                weight_embed_dim,
                num_nodes,
                config,
                inverse_mask=inverse_mask,
            )
        )
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(
                MaskGCNLayer(
                    dim_out,
                    dim_out,
                    cheb_k,
                    weight_embed_dim,
                    num_nodes,
                    config,
                    inverse_mask=inverse_mask,
                )
            )

    def forward(self, x, init_state, weight_embeddings, enable_explainability, time_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)

        assert x.shape[2] == weight_embeddings.shape[0] and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        i = 0
        all_masks = []
        all_mask_losses = []
        for cell in self.dcrnn_cells:
            state = init_state[i]
            inner_states = []
            masks = torch.jit.annotate(List[torch.Tensor], [])
            mask_losses = torch.jit.annotate(List[torch.Tensor], [])
            for t in range(seq_length):
                # Calculate the state on every timestep
                state, mask, mask_loss = cell(
                    current_inputs[:, t, :, :],
                    state,
                    weight_embeddings,
                    enable_explainability,
                    t,
                    time_embeddings,
                )
                inner_states.append(state)
                masks.append(mask)
                mask_losses.append(mask_loss)
            output_hidden.append(state)
            all_masks.append(torch.stack(masks, dim=1))
            all_mask_losses.append(torch.stack(mask_losses, dim=1))
            current_inputs = torch.stack(inner_states, dim=1)
            i += 1

        return current_inputs, output_hidden, all_masks, all_mask_losses

    def init_hidden(self, batch_size: int, num_nodes: int):
        init_states = []
        for cell in self.dcrnn_cells:
            init_states.append(cell.init_hidden_state(batch_size, num_nodes))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class SEGCRN(nn.Module):
    def __init__(self, config, visualization=False, inverse_mask=False):
        super(SEGCRN, self).__init__()
        self.input_dim = config["target_dim"]
        self.hidden_dim = config["rnn_units"]
        self.output_dim = config["output_dim"]
        self.window = config["window"]
        self.horizon = config["horizon"]
        self.num_layers = config["num_layers"]
        self.weigths_embed_dim = config["weigths_embed_dim"]
        self.cheb_k = config["cheb_k"]
        self.num_nodes = config["num_nodes"]
        self.config = config
        self.visualization = visualization
        self.inverse_mask = inverse_mask

        # Get the embedding using the ID if the graph is static
        self.weight_embedding_layer = nn.Embedding(self.num_nodes, self.weigths_embed_dim)
        self.time_embedding_layer = nn.Embedding(self.num_nodes, self.num_nodes)
        self.attention_layer = nn.TransformerEncoderLayer(d_model=self.num_nodes, nhead=1, batch_first=True)

        self.encoder = MaskGCNStack(
            self.input_dim,
            self.hidden_dim,
            self.cheb_k,
            self.weigths_embed_dim,
            num_layers=self.num_layers,
            num_nodes=self.num_nodes,
            config=config,
            inverse_mask=self.inverse_mask,
        )

        # Output
        self.output_linear = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        # Initialize weigths
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, graph_ids, graph_values, enable_explainability, times):
        # Generate the embeddings for the weights
        weight_embeddings = self.weight_embedding_layer(graph_ids[0])

        time_embeddings = self.time_embedding_layer(times)
        time_embeddings = self.attention_layer(time_embeddings)
        # Generate the initial state
        init_state = self.encoder.init_hidden(graph_values.shape[0], graph_values.shape[2])
        # B, T, N, hidden
        output, _, masks, mask_losses = self.encoder(
            graph_values, init_state, weight_embeddings, enable_explainability, time_embeddings
        )
        # Get the state of the node to predict
        output = output[:, -1:, :, :]  # noqa: E203

        # Process and get the prediction
        output = self.output_linear(output)

        # Resize output
        output = output.squeeze(-1)

        # Calculate the supports
        if self.visualization:
            supports = F.softmax(F.relu(torch.mm(weight_embeddings, weight_embeddings.transpose(0, 1))), dim=1)
            time_supports = F.softmax(F.relu(torch.bmm(time_embeddings.transpose(-2, -1), time_embeddings)), dim=2)
        else:
            supports = None
            time_supports = None

        return output, masks, mask_losses, supports, time_supports
