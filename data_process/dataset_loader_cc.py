"""PRCC dataset loader.

Loads (img, gray, pid, clothid, camid) tuples. The `gray` channel is
produced by averaging RGB channels; it is present in the batch output for
API compatibility with older code but the IAHL model uses only `img`.
"""
from __future__ import absolute_import

from torch.utils.data import DataLoader, Dataset

from utils.util import read_image
from data_process.samplers import RandomIdentitySampler
from data_process.transform import get_transform


class ImageClothDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, clothid, camid = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        # gray channel kept for loader compatibility; IAHL model ignores it.
        gray = img.mean(dim=0, keepdim=True)
        return img, gray, pid, clothid, camid


def get_prcc_dataset_loader(dataset, args, use_gpu=True):
    transform_train, transform_test = get_transform(args)

    sampler = RandomIdentitySampler(dataset.train,
                                     batch_size=args.train_batch,
                                     num_instances=args.num_instances)

    train_loader = DataLoader(
        ImageClothDataset(dataset.train, transform=transform_train),
        sampler=sampler, batch_size=args.train_batch,
        num_workers=args.num_workers, pin_memory=False, drop_last=True,
    )
    query_sc_loader = DataLoader(
        ImageClothDataset(dataset.query_cloth_unchanged, transform=transform_test),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=False, drop_last=False,
    )
    query_cc_loader = DataLoader(
        ImageClothDataset(dataset.query_cloth_changed, transform=transform_test),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=False, drop_last=False,
    )
    gallery_loader = DataLoader(
        ImageClothDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=False, drop_last=False,
    )

    return train_loader, query_sc_loader, query_cc_loader, gallery_loader
