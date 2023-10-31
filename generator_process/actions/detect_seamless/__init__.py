from enum import Enum

import numpy as np
from numpy.typing import NDArray

from ....api.models.seamless_axes import SeamlessAxes
from .... import image_utils

def detect_seamless(self, image: image_utils.ImageOrPath) -> SeamlessAxes:
    import os
    import torch
    from torch import nn

    if image.shape[0] < 8 or image.shape[1] < 8:
        return SeamlessAxes.OFF

    model = getattr(self, 'detect_seamless_model', None)
    if model is None:
        state_npz = np.load(os.path.join(os.path.dirname(__file__), 'model.npz'))
        state = {k: torch.tensor(v) for k, v in state_npz.items()}

        class SeamlessModel(nn.Module):
            def __init__(self):
                super(SeamlessModel, self).__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                    nn.Dropout(.2),
                    nn.PReLU(64),
                    nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
                    nn.Dropout(.2),
                    nn.PReLU(16),
                    nn.Conv2d(16, 64, kernel_size=8, stride=4, padding=0),
                    nn.Dropout(.2),
                    nn.PReLU(64),
                    nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=0),
                    nn.Dropout(.2)
                )
                self.gru = nn.GRU(64, 32, batch_first=True)
                self.fc = nn.Linear(32, 1)

            def forward(self, x: torch.Tensor):
                if len(x.size()) == 3:
                    x = x.unsqueeze(0)
                x = self.conv(x)
                h = torch.zeros(self.gru.num_layers, x.size()[0], self.gru.hidden_size,
                                dtype=x.dtype, device=x.device)
                x, h = self.gru(x.squeeze(3).transpose(2, 1), h)
                return torch.tanh(self.fc(x[:, -1]))

        model = SeamlessModel()
        model.load_state_dict(state)
        model.eval()
        setattr(self, 'detect_seamless_model', model)

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'cpu'
    else:
        device = 'cpu'

    image = image_utils.image_to_np(image, mode="RGB")

    # slice 8 pixels off each edge and combine opposing sides where the seam/seamless portion is in the middle
    # may trim up to 3 pixels off the length of each edge to make them a multiple of 4
    # expects pixel values to be between 0-1 before this step
    edge_x = np.zeros((image.shape[0], 16, 3), dtype=np.float32)
    edge_x[:, :8] = image[:, -8:]
    edge_x[:, 8:] = image[:, :8]
    edge_x *= 2
    edge_x -= 1
    edge_x = edge_x[:image.shape[0] // 4 * 4].transpose(2, 0, 1)

    edge_y = np.zeros((16, image.shape[1], 3), dtype=np.float32)
    edge_y[:8] = image[-8:]
    edge_y[8:] = image[:8]
    edge_y *= 2
    edge_y -= 1
    edge_y = edge_y[:, :image.shape[1] // 4 * 4].transpose(2, 1, 0)

    @torch.no_grad()
    def infer(*inputs):
        try:
            model.to(device)
            results = []
            for tensor in inputs:
                results.append(model(tensor))
            return results
        finally:
            # swap model in and out of device rather than reloading from file
            model.to('cpu')

    if edge_x.shape == edge_y.shape:
        # both edges batched together
        edges = torch.tensor(np.array([edge_x, edge_y]), dtype=torch.float32, device=device)
        res = infer(edges)
        return SeamlessAxes((res[0][0].item() > 0, res[0][1].item() > 0))
    else:
        edge_x = torch.tensor(edge_x, dtype=torch.float32, device=device)
        edge_y = torch.tensor(edge_y, dtype=torch.float32, device=device)
        res = infer(edge_x, edge_y)
        return SeamlessAxes((res[0].item() > 0, res[1].item() > 0))
