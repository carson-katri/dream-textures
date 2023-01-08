"""
It's recommended to copy this script to its own project folder to
keep it with your own image samples and trained models.

Each dataset should have images of the same square dimensions for batched training and validation.
You can train with multiple datasets in the same session.

dataset_layout/
    imagesNone/
        [sample_images]
    imagesX/
        [sample_images]
    imagesY/
        [sample_images]
    imagesXY/
        [sample_images]
"""

# if torch, numpy, and cv2 are not installed to site-packages
# import site
# site.addsitedir(r"path/to/dream_textures/.python_dependencies")

import itertools
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from numpy._typing import NDArray
from torch import nn
from torch.utils.data import Dataset, DataLoader

EDGE_SLICE = 8
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'


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
            nn.Dropout(.2),
        )
        self.gru = nn.GRU(64, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor):
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        # x[batch, channels, height, EDGE_SLICE*2]
        x = self.conv(x)
        # x[batch, features, height/4, 1]
        h = torch.zeros(self.gru.num_layers, x.size()[0], self.gru.hidden_size,
                        dtype=x.dtype, device=x.device)
        x = x.squeeze(3).transpose(2, 1)
        # x[batch, height/4, features]
        x, h = self.gru(x, h)
        return torch.tanh(self.fc(x[:, -1]))


def image_edges(path):
    image: NDArray = cv2.imread(path)
    # Pretty sure loading images is a bottleneck and makes the first epoch incredibly slow until fully in RAM.
    # Might be worth caching the edges in an easier to deserialize format with np.savez()

    edge_x = np.zeros((image.shape[0], EDGE_SLICE * 2, 3), dtype=np.float32)
    edge_x[:, :EDGE_SLICE] = image[:, -EDGE_SLICE:]
    edge_x[:, EDGE_SLICE:] = image[:, :EDGE_SLICE]

    edge_y = np.zeros((EDGE_SLICE * 2, image.shape[1], 3), dtype=np.float32)
    edge_y[:EDGE_SLICE] = image[-EDGE_SLICE:]
    edge_y[EDGE_SLICE:] = image[:EDGE_SLICE]

    return edge_x, edge_y


def prepare_edge(edge: NDArray, axis: str) -> torch.Tensor:
    edge = (edge * 2 / 255 - 1)
    if axis == 'x':
        edge = edge.transpose(2, 0, 1)
    elif axis == 'y':
        edge = edge.transpose(2, 1, 0)
    else:
        raise ValueError('axis should be "x" or "y"')
    return torch.as_tensor(edge, dtype=torch.float32)


def prepare_edges(edge_x: NDArray, edge_y: NDArray) -> tuple[torch.Tensor, torch.Tensor]:
    edge_x = edge_x * 2 / 255 - 1
    edge_y = edge_y * 2 / 255 - 1
    edge_x = edge_x.transpose(2, 0, 1)
    edge_y = edge_y.transpose(2, 1, 0)
    return torch.as_tensor(edge_x, dtype=torch.float32), torch.as_tensor(edge_y, dtype=torch.float32)


def seamless_tensor(seamless):
    return torch.tensor([1 if seamless else -1], dtype=torch.float32)


class EdgeDataset(Dataset):
    def __init__(self, path):
        self.data = []
        self._load_dir(os.path.join(path, 'imagesNone'), (False, False))
        self._load_dir(os.path.join(path, 'imagesX'), (True, False))
        self._load_dir(os.path.join(path, 'imagesY'), (False, True))
        self._load_dir(os.path.join(path, 'imagesXY'), (True, True))
        print(f'dataset loaded {path} contains {len(self)}')

    def _load_dir(self, imdir, seamless):
        if not os.path.exists(imdir):
            print(f'skipping {imdir}, does not exist')
            return
        if not os.path.isdir(imdir):
            print(f'skipping {imdir}, not a directory')
            return
        print(f'loading {imdir}')

        for image in sorted(os.listdir(imdir)):
            path = os.path.join(imdir, image)
            if not os.path.isfile(path):
                continue
            self.data.append((seamless_tensor(seamless[0]), None, 'x', path))
            self.data.append((seamless_tensor(seamless[1]), None, 'y', path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        ret = self.data[idx]
        if ret[1] is not None:
            return ret
        path = ret[3]
        edge_x, edge_y = prepare_edges(*image_edges(path))
        # Edges will be cached in cpu when first requested. Might not be desirable with a large enough dataset.
        if idx % 2 == 0:
            ret = (ret[0], edge_x, 'x', path)
            self.data[idx] = ret
            self.data[idx + 1] = (self.data[idx + 1][0], edge_y, 'y', path)
        else:
            self.data[idx - 1] = (self.data[idx - 1][0], edge_x, 'x', path)
            ret = (ret[0], edge_y, 'y', path)
            self.data[idx] = ret
        return ret


CHANNEL_PERMUTATIONS = [*itertools.permutations((0, 1, 2))]


class PermutedEdgeDataset(Dataset):
    """Permutes the channels to better generalize color data."""

    def __init__(self, dataset: EdgeDataset | str):
        if isinstance(dataset, str):
            dataset = EdgeDataset(dataset)
        self.base_dataset = dataset

    def __len__(self):
        return len(self.base_dataset) * len(CHANNEL_PERMUTATIONS)

    def __getitem__(self, idx):
        perm = CHANNEL_PERMUTATIONS[idx % len(CHANNEL_PERMUTATIONS)]
        result, edge, edge_type, path = self.base_dataset[idx // len(CHANNEL_PERMUTATIONS)]
        edge_perm = torch.zeros(edge.size(), dtype=edge.dtype)
        edge_perm[0] = edge[perm[0]]
        edge_perm[1] = edge[perm[1]]
        edge_perm[2] = edge[perm[2]]
        return result, edge_perm, edge_type, path, perm


def mix_iter(*iterables):
    """Iterates through multiple objects while attempting to balance
    by yielding from which one has the highest of remaining/length"""
    iterables = [x for x in iterables if len(x) > 0]
    lengths = [len(x) for x in iterables]
    counts = lengths.copy()
    ratios = [1.] * len(iterables)
    iters = [x.__iter__() for x in iterables]
    while True:
        idx = -1
        max_ratio = 0
        for i, ratio in enumerate(ratios):
            if ratio > max_ratio:
                idx = i
                max_ratio = ratio
        if idx == -1:
            return
        c = counts[idx] - 1
        counts[idx] = c
        ratios[idx] = c / lengths[idx]
        yield next(iters[idx])


def train(model: nn.Module, train_datasets, valid_datasets, epochs=1000, training_rate=0.0001, batch=50):
    train_loaders = [DataLoader(PermutedEdgeDataset(ds), batch_size=batch, shuffle=True, num_workers=0, pin_memory=True)
                     for ds in train_datasets]
    valid_loaders = [DataLoader(ds, batch_size=batch, num_workers=0, pin_memory=True)
                     for ds in valid_datasets]

    criterion = nn.MSELoss()
    criterion.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), training_rate, .9)

    def train_one_epoch():
        running_loss = 0.
        print_rate = 5000
        print_after = print_rate
        for i, data in enumerate(mix_iter(*train_loaders)):
            seamless = data[0].to(DEVICE)
            edge = data[1].to(DEVICE)

            optimizer.zero_grad()

            output = model(edge)

            loss: torch.Tensor = criterion(output, seamless)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            running_loss += loss.item()

            if i * batch > print_after:
                print_after += print_rate
                print(f"LOSS train {running_loss / (i + 1)}")

        return running_loss / (i + 1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    best_vloss = 1_000_000.

    for epoch in range(epochs):
        print(f'EPOCH {epoch}:')

        model.train(True)
        avg_loss = train_one_epoch()

        model.train(False)

        running_vloss = 0.0
        with torch.no_grad():
            for i, vdata in enumerate(mix_iter(valid_loaders)):
                expected_results = vdata[0].to(DEVICE)
                inputs = vdata[1].to(DEVICE)
                outputs = model(inputs)
                vloss = criterion(outputs, expected_results)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'model/model_{timestamp}_{epoch}_{int(avg_vloss * 1000)}.pt'
            torch.save(model.state_dict(), model_path)


@torch.no_grad()
def validate(model, datasets):
    # datasets here do not need images of equal sizes or to be square as there is no batching
    passes = 0
    fails = 0
    print_limit = 100
    print_count = 0

    def do_print(result, path, axis):
        nonlocal print_count
        if print_count < print_limit:
            print(f'{path} {axis} {result}')
        print_count += 1

    for valid_dataset in datasets:
        for data in valid_dataset:
            expected_result = data[0]
            tensor = data[1]
            axis = data[2]
            path = data[3]
            result = model(tensor.to(DEVICE))[0].item()
            if expected_result.item() == 1:
                if result >= 0:
                    passes += 1
                else:
                    fails += 1
                    do_print(result, path, axis)
            elif expected_result.item() == -1:
                if result < 0:
                    passes += 1
                else:
                    fails += 1
                    do_print(result, path, axis)
            else:
                raise RuntimeError(f'Unexpected result target {expected_result.item()}')
    if print_count > print_limit:
        print(f"{print_count - print_limit} more")
    total = passes + fails
    print(f"Passed: {passes} | {passes / total * 100:.2f}%")  # edge accuracy
    print(f"Failed: {fails} | {fails / total * 100:.2f}%")
    print(f"Passed²: {(passes / total) ** 2 * 100:.2f}%")  # image accuracy


# I prefer to not perpetuate the public distribution of torch.save() pickled files.
def save_npz(path, state_dict):
    np.savez(path, **state_dict)


def load_npz(path):
    state_dict_np: dict[str, NDArray] = np.load(path, allow_pickle=False)
    state_dict_torch = dict()
    for name, arr in state_dict_np.items():
        state_dict_torch[name] = torch.from_numpy(arr)
    return state_dict_torch


def main():
    model = SeamlessModel()

    # resume training or validate a saved model
    # model.load_state_dict(load_npz("model.npz"))
    # model.load_state_dict(torch.load("model/model_20221203_162623_26_10.pt"))

    model.to(DEVICE)

    datasets = [
        (EdgeDataset('train/samples'), EdgeDataset('valid/samples')),
        (EdgeDataset('train/samples2x'), EdgeDataset('valid/samples2x')),
        (EdgeDataset('train/samples4x'), EdgeDataset('valid/samples4x'))
    ]

    # Though it's possible to keep training and validation samples in the same dataset, you really shouldn't.
    # If you add new samples to a dataset that's being used like this DO NOT resume a previously trained model.
    # Training and validation samples will get reshuffled and your validation samples will likely be overfit.
    # gen = torch.Generator().manual_seed(132)
    # datasets = [
    #     torch.utils.data.random_split(EdgeDataset('samples'), [.8, .2], gen),
    #     torch.utils.data.random_split(EdgeDataset('samples2x'), [.8, .2], gen),
    #     torch.utils.data.random_split(EdgeDataset('samples4x'), [.8, .2], gen)
    # ]

    # If you're generating new samples it can be useful to modify generator_process/actions/prompt_to_image.py
    # to automatically save images to the dataset. It's best to keep them separate at first and run a previously
    # trained model on them to help find bad samples. Stable diffusion can at times add solid colored borders to
    # the edges of images that are not meant to be seamless. I recommend deleting all samples where an edge
    # appears seamless with scrutiny but was not generated to be, don't move it to another folder in the dataset.
    # datasets = [
    #     (None, EdgeDataset('tmp'))
    # ]

    # If you only want to validate a saved model.
    # datasets = [(None, valid) for _, valid in datasets]

    train_datasets = []
    valid_datasets = []
    for t, v in datasets:
        if t is not None:
            train_datasets.append(t)
        if v is not None:
            valid_datasets.append(v)

    try:
        if len(train_datasets) > 0:
            train(model, train_datasets, valid_datasets, epochs=50, training_rate=0.001)
            # It should easily converge in under 50 epochs.
            # Training rate is a little high, but I've never managed better
            # results with a lower rate and several times more epochs.
    except KeyboardInterrupt:
        pass

    # As long as your images have meaningful names you can get feedback on
    # what kind of images aren't detecting well to add similar samples.
    model.train(False)
    validate(model, valid_datasets)


if __name__ == '__main__':
    main()
