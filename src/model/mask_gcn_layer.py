import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x, iteration):
        pos_encoding = self.pe[0][iteration]
        x = x + pos_encoding
        return x


class MaskGCNLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        cheb_k,
        weight_embed_dim,
        num_nodes,
        config,
        inverse_mask=False,
    ):
        super(MaskGCNLayer, self).__init__()
        self.hidden_dim = dim_out
        self.reset_gate = MaskGate(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, weight_embed_dim)
        self.update_gate = MaskGate(dim_in + self.hidden_dim, dim_out, cheb_k, weight_embed_dim)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.min_percentage_nodes = config["min_percentage_nodes"]
        self.mask_pool = nn.Parameter(torch.FloatTensor(weight_embed_dim, num_nodes))
        self.time_pool = nn.Parameter(torch.FloatTensor(config["window"], num_nodes))
        self.config = config
        self.num_nodes = num_nodes
        self.mask_batch_norm = nn.BatchNorm1d(num_nodes)
        self.positional_encoding = PositionalEncoding(
            d_model=weight_embed_dim,
            max_len=config["window"],
        )
        # Used to calculate the fidelity
        self.inverse_mask = inverse_mask

    def apply_hard_concrete(self, input, temperature=1 / 3, gamma=-0.2, zeta=1.0, bias=3):
        # Shift the initial values using the bias
        input = input + bias

        if self.training:
            # Calculate the noise from a uniform distribution between 0 and 1
            u = torch.empty_like(input).uniform_(1e-6, 1.0 - 1e-6)
            # Apply a distribution with values between 0 and 1
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + input) / temperature)
            # Calculate the penalty (it needs to use the temperature and the ratio between gamma and zeta
            # because the distribution varies on the basis of this values
            gamma_zeta_ratio = np.math.log(-gamma / zeta)
            penalty = torch.sigmoid(input - temperature * gamma_zeta_ratio)
        else:
            s = torch.sigmoid(input)
            penalty = torch.ones_like(input)

        # Stretch the distribution
        s = s * (zeta - gamma) + gamma
        # Clip the distribution to set as 0 and 1 the values that are lower and greater than this values respectively
        clipped_s = s.clamp(0, 1)
        # Convert values between 0 and 1 in booleans and convert again to float
        hard_concrete = clipped_s.round()

        return hard_concrete, penalty

    def forward(self, x, state, weight_embeddings, enable_explainability, iteration, time_embeddings):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)

        # Get the embeddings of every node to calculate the mask
        if enable_explainability:
            weight_embeddings = self.positional_encoding(weight_embeddings, iteration)
            repeated_embeddings = weight_embeddings.repeat(state.shape[0], 1, 1)

            raw_mask = torch.einsum("bni,im->bnm", repeated_embeddings, self.mask_pool)
            time_mask = torch.einsum("bni,im->bnm", time_embeddings.transpose(-2, -1), self.time_pool)
            raw_mask = raw_mask + time_mask
            raw_mask = self.mask_batch_norm(raw_mask)

            # Apply hard concrete distribution
            mask, penalty = self.apply_hard_concrete(raw_mask)

            # Used when we need to calculate the fidelity
            if self.inverse_mask:
                mask = (1 - mask).round()

            # Get a version of the mask without a percentage of nodes equal to the value of 'min_percentage_nodes'
            # to apply the reduction of loss without being affected by the penalty
            active_nodes = mask.mean()
            prob_removes = torch.clamp(self.min_percentage_nodes / active_nodes + 1e-8, max=1)
            remove_one_probabilities = mask * prob_removes
            remove_random_connect = torch.bernoulli(remove_one_probabilities).to(device="cuda")
            # In this way we indicate the model that going under the percentage of nodes of 'min_percentage_nodes'
            # doesn't decrease the mask loss
            reduced_mask = mask - remove_random_connect

            # Calculate the mask loss using the penalty and the reduced mask
            penalty_norm = torch.sum((mask > 0), (1, 2)).float()
            mask_loss = torch.sum((penalty * (reduced_mask > 0).float()), (1, 2)) / (penalty_norm + 1e-8)

            # Penalize when the percentage of nodes used is under the desired percentage
            remove_penalty = torch.clamp((self.min_percentage_nodes / active_nodes) - 1, min=0)
            mask_loss = mask_loss + remove_penalty
        else:
            mask = torch.ones(state.shape[0], state.shape[1], state.shape[1]).to(device="cuda")
            mask_loss = torch.zeros(state.shape[0]).to(device="cuda")

        z_r = torch.sigmoid(
            self.reset_gate(input_and_state, weight_embeddings, mask, enable_explainability, time_embeddings)
        )
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update_gate(candidate, weight_embeddings, mask, enable_explainability, time_embeddings))
        h = r * state + (1 - r) * hc

        return h, mask, mask_loss

    def init_hidden_state(self, batch_size: int, num_nodes: int):
        return torch.zeros(batch_size, num_nodes, self.hidden_dim)


class MaskGate(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(MaskGate, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings, mask, enable_explainability, time_embeddings):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        # The variable 'supports' is the equivalent of the Laplacian matrix calculated by the network itself.
        # So the mask to the adjacency matrix should be applied to it.
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        time_supports = F.softmax(F.relu(torch.bmm(time_embeddings.transpose(-2, -1), time_embeddings)), dim=2)

        # Adapt torch eye to
        identity_matrix = torch.eye(supports.shape[1]).to(supports.device)

        if enable_explainability:
            # Apply the mask
            supports = supports * mask
            time_supports = time_supports * mask

            # Repeat the identity matrix to equals the batch size
            identity_matrix = identity_matrix.repeat(supports.shape[0], 1, 1)

        supports = (supports + time_supports) / 2

        # N, cheb_k, dim_in, dim_out
        weights = torch.einsum("nd,dkio->nkio", node_embeddings, self.weights_pool)

        bias = torch.matmul(node_embeddings, self.bias_pool)  # B, N, dim_out

        x_a = torch.matmul(identity_matrix, x)  # B, N, dim_in
        x_b = torch.matmul(supports, x)  # B, N, dim_in

        x_g = torch.stack([x_a, x_b], dim=1)
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum("bnki,nkio->bno", x_g, weights) + bias  # b, N, dim_out

        return x_gconv
