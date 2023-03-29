from __future__ import annotations
import os
import sys
import shutil
import atexit
import importlib
import subprocess
from pathlib import Path
from typing import Sequence, Callable

import h5py
import numpy as np
from tqdm import tqdm
import torch
from torchvision.datasets import VisionDataset
# from multi_object_datasets import [?]: we do this dynamically in MultiObjectDataset.convert


class MultiObjectDataset(VisionDataset):

    multi_object_datasets = ['cater_with_masks', 'clevr_with_masks', 'multi_dsprites', 'objects_room', 'tetrominoes']

    def __init__(
            self,
            root: str,
            dataset: str,
            version: str | None,
            split: str,
            ttv: Sequence[int],
            features: Sequence[str],
            transforms: dict[str, Callable],
            tf_files: Sequence[str],
            tf_max_size: int,
            h5_file: str,
            download: bool,
            convert: bool
            # TODO: transforms: dict
    ) -> None:
        assert dataset in self.multi_object_datasets, f"Unknown dataset: {dataset}. " \
                                                      f"Available options are: {self.multi_object_datasets}"
        split = split.capitalize()
        assert split in ['Train', 'Test', 'Val'],     f"Unknown split: '{split}'. " \
                                                      f"Available options are: {['Train', 'Test', 'Val']}"
        assert len(ttv) == 3,                         f"ttv has to be len 3. Given: {ttv} which is len {len(ttv)}"
        assert sum(ttv) <= tf_max_size,               f"The requested [train,test,val] size is larger than the " \
                                                      f"available data: {ttv}={sum(ttv):,} > {tf_max_size:,} (max. " \
                                                      f"datapoints in tfrecord files)"
        for k in transforms.keys():
            assert k in features,                     f"Transforms key '{k}' not in available features: {features}"

        root = str(Path(root).expanduser().resolve() / "multi-object-datasets" / dataset)
        super().__init__(root)

        self.version = version
        self.split = split
        self.features = features
        self.transforms = transforms
        self.ttv = ttv
        self.ttv_size = sum(ttv)
        self.indices = self.calc_split_indices()
        assert len(self.indices) > 0, f"You requested an empty '{self.split}' split of '{dataset}' " \
                                      f"with ttv={self.ttv}"

        self.tf_files = [str(Path(self.root) / f) for f in tf_files]
        self.tf_module = dataset
        self.tf_max_size = tf_max_size

        self.h5_file = Path(self.root) / h5_file
        self.h5_data = None
        self.h5_size = None

        # new shapes to convert channel format in self.convert()
        # image:   HxWxC ->   CxHxW (cater has additional sequence dimension)
        # masks: MxHxWxC -> MxCxHxW (cater has additional sequence dimension)
        self._image_T =   (0, 3, 1, 2) if dataset == 'cater_with_masks' else    (2, 0, 1)
        self._mask_T = (0, 1, 4, 2, 3) if dataset == 'cater_with_masks' else (0, 3, 1, 2)

        if self.h5_exists() and self.ttv_size <= self.get_h5_size():
            return

        if download:
            self.download()

        if convert:
            self.convert()

        if not self.h5_exists():
            raise RuntimeError("Dataset not found. You can use download=True and convert=True to 'build' it")

        if self.ttv_size > self.get_h5_size():
            raise AssertionError(f"The requested [train,test,val] size is larger than the converted data: "
                                 f"{self.ttv}={self.ttv_size} > {self.get_h5_size()} ({self.h5_file.name}). "
                                 f"You can use convert=True to convert more data")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data = self.get_h5_data()

        # for every (feature) key, read data from h5 with an index lookup, convert it to a torch tensor and
        # create a list of (key, value) tuples that are finally converted to a dict
        d = dict([(k, torch.from_numpy(data[k][self.indices[idx]])) for k in data.keys()])

        for k, transform in self.transforms.items():
            d[k] = transform(d[k])

        return d

    def extra_repr(self) -> str:

        # TODO: allow a verbose print with h5 and ttv info?
        v_repr = f"Version: {self.version}" if self.version is not None else ""
        h_repr = "" # f"H5 file: {self.h5_file.name} (converted {self.get_h5_size():,}/{self.tf_max_size:,} max)"
        t_repr = "" # f"Train,Test,Val: {self.ttv}"
        s_repr = f"Split: {self.split}"
        f_repr = f"Features: {self.features}"

        return "\n".join([r for r in [v_repr, h_repr, t_repr, s_repr, f_repr] if r])

    def calc_split_indices(self) -> Sequence:

        # TODO: allow random splits? multi-object-datasets seem random enough though and, the tf implementation
        #       reads the data sequentially as well (with a read buffer from which random batches a drawn).
        train, test, val = self.ttv
        if self.split == 'Train': start, end = 0, train
        if self.split == 'Test':  start, end = train, (train + test)
        if self.split == 'Val':   start, end = (train + test), (train + test + val)

        return list(range(start, end))

    def get_h5_size(self) -> int:
        if self.h5_size is None:
            with h5py.File(self.h5_file, 'r', libver='latest') as h5_data:
                self.h5_size = h5_data['image'].len()
        return self.h5_size

    def h5_exists(self) -> bool:
        return self.h5_file.exists() and self.h5_file.is_file()

    def get_h5_data(self):
        """
        Lazy load the h5_file in __getitem__ instead of __init__ to support multiple workers in
        torch.utils.data.DataLoader. This is because:

        'Read-only parallel access to HDF5 files works with no special preparation:
         each process should open the file independently and read data normally
         (avoid opening the file and then forking).' - docs.h5py.org

        See: https://docs.h5py.org/en/stable/mpi.html
        See: https://discuss.pytorch.org/t/hdf5-multi-threaded-alternative/6189/18
        See: https://discuss.pytorch.org/t/what-s-the-best-way-to-load-large-hdf5-data/11044/4
        """
        if self.h5_data is None:
            self.h5_data = h5py.File(self.h5_file, 'r', libver='latest')  # open  the file once
            atexit.register(lambda: self.h5_data.close())                 # close the file on exit
        return self.h5_data

    def tf_exist(self) -> bool:
        return all([(Path(f).exists() and Path(f).is_file()) for f in self.tf_files])

    def download(self) -> None:

        # 1. skipp if TFRecord files exist
        if self.tf_exist():
            return

        # 2. ensure the dataset directory exists
        dataset = Path(self.root)
        dataset.mkdir(parents=True, exist_ok=True)

        # 3. check if gsutil is installed
        gsutil = shutil.which('gsutil')
        if gsutil is None:
            raise RuntimeError(f"gsutil is not available. Please install it, "
                               f"e.g. with: conda install -c conda-forge gsutil")

        # 4. download the dataset (all files/versions)
        subprocess.run([gsutil, "-m", "cp", "-n", "-r",
                        f"gs://multi-object-datasets/{dataset.name}", f"{dataset.parent}"],
                       check=True)

    def convert(self) -> None:

        # 1. check if TFRecord files exist
        if not self.tf_exist():
            raise RuntimeError(f"TFRecord files not found or missing in: {self.root}/. "
                               f"You can use download=True to download them.")

        # 2. print some info what is going to happen
        print(f"Convert: \033[1m{self.h5_file.name}\033[0m")
        if self.h5_exists():
            print(f"Found:   {self.h5_file.name} with size={self.get_h5_size():,}")
        print(f"Request: ttv={self.ttv} -> {self.ttv_size:,}/{self.tf_max_size:,} max")

        # 3. on first time use: convert ttv_size by default (ask the user what to do otherwise)
        if not self.h5_exists():
            size = self.ttv_size
        else:
            print("\nWhat do you want to do?\n")
            print(f"  ttv (s): convert the requested ttv size: {self.ttv_size:>{len(str(self.tf_max_size))}}")
            print(f"  max (m): convert the full tfrecord file: {self.tf_max_size}")
            try:
                choice = input("\nSelect: 's' or 'm' (Press CTRL-C to abort): ").strip()
            except KeyboardInterrupt:
                print(f"KeyboardInterrupt: Abort conversion")
                sys.exit(1)

            assert choice in ['s', 'm'], f"Invalid option: '{choice}'. Please select one of: ['s', 'm']"
            if choice == 's': size = self.ttv_size
            if choice == 'm': size = self.tf_max_size

        # 4. dynamically import one of: cater_with_masks, clevr_with_masks, multi_dsprites, objects_room, tetrominoes
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # set log level WARN
        tf_module = importlib.import_module(f'multi_object_datasets.{self.tf_module}')

        # 5. initialize the TFRecord dataset (some need a version string)
        if self.version is None:
            numpy_iterator = tf_module.dataset(self.tf_files).as_numpy_iterator()
        else:
            numpy_iterator = tf_module.dataset(self.tf_files, self.version).as_numpy_iterator()

        # 6. convert to hdf5 dataset
        tmp_h5_file = self.h5_file.parent / f"tmp_{self.h5_file.name}"
        try:
            with tqdm(total=size, desc=f'Convert: {self.h5_file.name}') as pbar:
                with h5py.File(tmp_h5_file, 'w', libver='latest') as h5_data:
                    for d in numpy_iterator:
                        for k, v in d.items():

                            # arrays are read only when returned from the tf numpy iterator,
                            # so we copy to ensure that the flags OWNDATA & WRITEABLE = True
                            # TODO: not sure if this is actually necessary as we write to h5
                            v = v.copy()

                            # convert channel format from HxWxC to CxHxW (see __init__ for details)
                            if k == 'image': v = np.transpose(v, axes=self._image_T)
                            if k == 'mask':  v = np.transpose(v, axes=self._mask_T)

                            h5_data.require_dataset(k, shape=(size,) + v.shape, dtype=v.dtype,
                                                       chunks=(1,)   + v.shape, compression="gzip")
                            h5_data[k][pbar.n] = v
                        pbar.update(1)
                        if pbar.n == size:
                            break
                    self.h5_size = h5_data['image'].len()
            print("Flushing write buffer. This may take a few moments to complete...")
        except Exception as e:
            tmp_h5_file.unlink(missing_ok=True)
            raise e
        except KeyboardInterrupt:
            print(f"KeyboardInterrupt: Abort conversion")
            tmp_h5_file.unlink(missing_ok=True)
            sys.exit(1)

        # 7. if all went well, replace old file with new one
        self.h5_file.unlink(missing_ok=True)
        self.h5_file = tmp_h5_file.rename(self.h5_file)

        # TODO: Remove tfrecord files afterwards to save disk space?


class CaterWithMasks(MultiObjectDataset):

    def __init__(self, root, split='Train', ttv=[39364, 17100, 0], transforms={}, download=True, convert=True) -> None:

        if ttv != [39364, 17100, 0]:
            print(f"WARNING: The intended train,test,val split is: [39364, 17100, 0] but you requested {ttv}")

        super().__init__(root=root, dataset="cater_with_masks", version=None, split=split, ttv=ttv,
                         features = ['camera_matrix', 'image', 'mask', 'object_positions'],
                         transforms=transforms,
                         tf_files=[f"cater_with_masks_{t}.tfrecords-{str(i).rjust(5, '0')}-of-00100" for
                                   t in ("train", "test") for i in range(0, 100)],
                         tf_max_size=56464,
                         h5_file="cater_with_masks.hdf5",
                         download=download, convert=convert)


class ClevrWithMasks(MultiObjectDataset):

    def __init__(self, root, split='Train', ttv=[90000, 5000, 5000], transforms={}, download=True, convert=True) -> None:

        super().__init__(root=root, dataset="clevr_with_masks", version=None, split=split, ttv=ttv,
                         features=['color', 'image', 'mask', 'material', 'pixel_coords', 'rotation', 'shape', 'size',
                                   'visibility', 'x', 'y', 'z'],
                         transforms=transforms,
                         tf_files=["clevr_with_masks_train.tfrecords"],
                         tf_max_size=100000,  # 100k
                         h5_file="clevr_with_masks.hdf5",
                         download=download, convert=convert)


class MultiDSprites(MultiObjectDataset):

    def __init__(self, root, split='Train', ttv=[90000, 5000, 5000], transforms={}, version='colored_on_grayscale',
                 download=True, convert=True) -> None:

        versions = ['binarized', 'colored_on_colored', 'colored_on_grayscale']
        assert version in versions, f"Unknown version: {version}. Available options are: {versions}"

        super().__init__(root=root, dataset="multi_dsprites", version=version, split=split, ttv=ttv,
                         features=['color', 'image', 'mask', 'orientation', 'scale', 'shape', 'visibility', 'x', 'y'],
                         transforms=transforms,
                         tf_files=[f"multi_dsprites_{version}.tfrecords"],
                         tf_max_size=1000000,  # 1m
                         h5_file=f"multi_dsprites_{version}.hdf5",
                         download=download, convert=convert)


class ObjectsRoom(MultiObjectDataset):

    def __init__(self, root, split='Train', ttv=[90000, 5000, 5000], transforms={}, download=True, convert=True) -> None:

        norm_splits = ['Train', 'Test', 'Val']
        ood_splits = ['empty_room', 'identical_color', 'six_objects']
        assert split.capitalize() in norm_splits or split in ood_splits, f"Unknown split: {split}. Available options " \
                                                                         f"are: {norm_splits} or {ood_splits}"

        if split in ['empty_room', 'identical_color', 'six_objects']:
            print(f"INFO: '{split}' is a special out-of-distribution 'Test' split and will allways return as "
                  f"ttv=[0, 922, 0], independent of the requested ttv.")
            version = split
            split, ttv, tf_max_size = "Test", [0, 922, 0], 922  # almost 1k
            tf_files = [f"objects_room_test_{version}.tfrecords"]
            h5_file = f"objects_room_test_{version}.hdf5"
        else:
            version = None
            tf_max_size = 1000000  # 1m
            tf_files = ["objects_room_train.tfrecords"]
            h5_file = "objects_room.hdf5"

        super().__init__(root=root, dataset="objects_room", version=version, split=split, ttv=ttv,
                         features=['image', 'mask'], transforms=transforms,
                         tf_files=tf_files,
                         tf_max_size=tf_max_size,
                         h5_file=h5_file,
                         download=download, convert=convert)


class Tetrominoes(MultiObjectDataset):

    def __init__(self, root, split='Train', ttv=[90000, 5000, 5000], transforms={}, download=True, convert=True) -> None:

        super().__init__(root=root, dataset="tetrominoes", version=None, split=split, ttv=ttv,
                         features=['color', 'image', 'mask', 'shape', 'visibility', 'x', 'y'],
                         transforms=transforms,
                         tf_files=["tetrominoes_train.tfrecords"],
                         tf_max_size=1000000,  # 1m
                         h5_file="tetrominoes.hdf5",
                         download=download, convert=convert)
