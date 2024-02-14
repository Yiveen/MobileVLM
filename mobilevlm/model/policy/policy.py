from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from action_embed import *
from action_decoder import *
class PolicyModel(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
    ):
        super().__init__()

        self.action_encoder = ActionEmbedding(
            output_dim=embed_dim,
            embed_dict={
                "pose0_position": ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=2,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
                "pose0_rotation": ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=4,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
                "pose1_position": ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=2,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
                "pose1_rotation": ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=4,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
            },
        )
        self.action_decoder = ActionDecoder(
            input_dim=embed_dim,
            action_dims={
                "pose0_position": [50, 100],
                "pose0_rotation": [50] * 4,
                "pose1_position": [50, 100],
                "pose1_rotation": [50] * 4,
            },
            hidden_dim=512,
            hidden_depth=2,
            activation="relu",
            norm_type=None,
            last_layer_gain=0.01,
        )

        self._n_discrete_x_bins = 50
        self._n_discrete_y_bins = 100
        self._n_discrete_z_bins = 50
        self._n_discrete_rot_bins = 50

    def forward_action_token(self, action):
        return self.action_encoder(self._de_discretize_actions(action))

    def forward_action_decoder(self, predicted_action_tokens: torch.Tensor):
        return self.action_decoder(predicted_action_tokens)

    def discretize_action(self, action):
        device = action["pose0_position"].device
        boundary_x = torch.linspace(
            start=0, end=1, steps=self._n_discrete_x_bins, device=device
        )
        boundary_y = torch.linspace(
            start=0, end=1, steps=self._n_discrete_y_bins, device=device
        )
        boundary_rot = torch.linspace(
            start=0, end=1, steps=self._n_discrete_rot_bins, device=device
        )

        action["pose0_position"][..., 0] = torch.bucketize(
            action["pose0_position"][..., 0].contiguous(), boundary_x
        )
        action["pose0_position"][..., 1] = torch.bucketize(
            action["pose0_position"][..., 1].contiguous(), boundary_y
        )
        action["pose0_rotation"] = torch.bucketize(
            action["pose0_rotation"].contiguous(), boundary_rot
        )

        action["pose1_position"][..., 0] = torch.bucketize(
            action["pose1_position"][..., 0].contiguous(), boundary_x
        )
        action["pose1_position"][..., 1] = torch.bucketize(
            action["pose1_position"][..., 1].contiguous(), boundary_y
        )
        action["pose1_rotation"] = torch.bucketize(
            action["pose1_rotation"].contiguous(), boundary_rot
        )
        action = {k: v.long() for k, v in action.items()}
        return action

    def _de_discretize_actions(self, actions):
        actions = {k: v.float() for k, v in actions.items()}
        actions["pose0_position"][..., 0] = (
            actions["pose0_position"][..., 0] / self._n_discrete_x_bins
        )
        actions["pose0_position"][..., 1] = (
            actions["pose0_position"][..., 1] / self._n_discrete_y_bins
        )
        actions["pose0_rotation"] = (
            actions["pose0_rotation"] / self._n_discrete_rot_bins
        )

        actions["pose1_position"][..., 0] = (
            actions["pose1_position"][..., 0] / self._n_discrete_x_bins
        )
        actions["pose1_position"][..., 1] = (
            actions["pose1_position"][..., 1] / self._n_discrete_y_bins
        )
        actions["pose1_rotation"] = (
            actions["pose1_rotation"] / self._n_discrete_rot_bins
        )
        return actions
