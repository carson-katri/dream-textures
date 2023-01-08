from enum import Enum

import numpy as np
from numpy.typing import NDArray


class SeamlessAxes(Enum):
    """Unified handling of seamless axes.
    Can be converted from str (id or text) or bool tuple/list (x, y).
    Each enum is equal to their respective convertible values.
    Special cases:
        AUTO: None
        OFF: False, empty str
        BOTH: True
    """

    AUTO =       'auto', 'Auto-detect', None,  None
    OFF =        'off',  'Off',         False, False
    HORIZONTAL = 'x',    'X',           True,  False
    VERTICAL =   'y',    'Y',           False, True
    BOTH =       'xy',   'Both',        True,  True

    def __init__(self, id, text, x, y):
        self.id = id
        self.text = text
        self.x = x
        self.y = y

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self is other
        if isinstance(other, str):
            return self.id == other or self.text == other or (other == '' and self is self.OFF)
        if isinstance(other, (tuple, list)) and len(other) == 2:
            return self.x == other[0] and self.y == other[1]
        if other is True and self is self.BOTH:
            return True
        if other is False and self is self.OFF:
            return True
        if other is None and self is self.AUTO:
            return True
        return False

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            if value == '':
                return cls.OFF
            for e in cls:
                if e.id == value or e.text == value:
                    return e
            raise ValueError(f'no {cls.__name__} with id {repr(id)}')
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            for e in cls:
                if e.x == value[0] and e.y == value[1]:
                    return e
            raise ValueError(f'no {cls.__name__} with x {value[0]} and y {value[1]}')
        elif value is True:
            return cls.BOTH
        elif value is False:
            return cls.OFF
        elif value is None:
            return cls.AUTO
        raise TypeError(f'expected str, bool, tuple[bool, bool], or None, got {repr(value)}')

    def bpy_enum(self, *args):
        return self.id, self.text, *args


def detect_seamless(self, image: NDArray) -> SeamlessAxes:
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

    if image.shape[2] == 4:
        # only trained on RGB channels, not alpha
        image = image[:, :, :3]

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
