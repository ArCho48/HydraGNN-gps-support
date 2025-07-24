##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ModuleList
from torch_geometric.nn import GINEConv, BatchNorm, Sequential

from .Base import Base


class GINEStack(Base):
    def __init__(self, input_args, conv_args, edge_dim: int, *args, **kwargs):
        self.edge_dim = edge_dim
        self.is_edge_model = True  # specify that mpnn can handle edge features
        super().__init__(input_args, conv_args, *args, **kwargs)

    def get_conv(self, input_dim, output_dim, edge_dim=None):
        gine = GINEConv(
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
            ),
            eps=100.0,
            train_eps=True,
            edge_dim=edge_dim
        )

        return Sequential(
            self.input_args,
            [
                (gine, self.conv_args + " -> inv_node_feat"),
                (
                    lambda x, equiv_node_feat: [x, equiv_node_feat],
                    "inv_node_feat, equiv_node_feat -> inv_node_feat, equiv_node_feat",
                ),
            ],
        )

    def __str__(self):
        return "GINStack"
