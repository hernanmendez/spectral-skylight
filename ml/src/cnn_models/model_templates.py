import pickle
import random
import time
from sqlalchemy import true
import torch
import math

import pandas as pd
import torch.nn as nn

from dataclasses import dataclass
from os.path import exists
from collections import OrderedDict
from typing import FrozenSet, List

from multiprocessing.dummy import Pool

from IPython.display import display


def cache_dict_to_path(dict, name):
    path = f"cache-{name}"
    for key in dict:
        escaped = str(dict[key]).replace("/", "+")
        path += f"[{key}]{escaped}"
    return path


def cache_path_to_dict(path):
    dict = {}
    pairs = path.split("[")[1:]
    for pair in pairs:
        key, value = pair.split("]")
        dict[key] = value.replace("+", "/")
    return dict


@dataclass
class DataAggregateObject:
    data_mean: torch.Tensor
    data_range: torch.Tensor
    normalized_data: torch.Tensor

    def __init__(self, src_data: torch.Tensor):
        # mean of all measurements for each frequency
        self.data_mean = torch.mean(src_data, dim=0)

        # range of all measurements for each frequency
        self.data_range = (
            torch.max(src_data, dim=0).values - torch.min(src_data, dim=0).values
        )

        self.normalized_data = (src_data - self.data_mean) / self.data_range


@dataclass
class DataBundle:
    df: pd.DataFrame
    x_image_branch: torch.Tensor
    x_feature_branch: torch.Tensor
    y: torch.Tensor


@dataclass
class DevelopmentDataObject:
    training_ratio: float
    # excluded_skies: set
    src_skies_csv_filepath: str
    data_aggregates: DataAggregateObject
    train_data: DataBundle
    test_data: DataBundle
    # exceptions_data: DataBundle


@dataclass
class SkyIndex:
    datetime: str  # TODO: change to datetime object
    sky_start_index: int
    number_of_samples: int

    def convert_to_file_path(self, src_folder_path):
        pass

    def get_tensor_locations(self):
        return list(
            range(self.sky_start_index, self.sky_start_index + self.number_of_samples)
        )


@dataclass
class SkylightData:
    spectrum_beginning = 350
    spectrum_end = 1780
    wavelengths = [
        wavelength for wavelength in range(spectrum_beginning, spectrum_end + 1)
    ]
    sky_data_df = pd.DataFrame
    data_aggregates: DataAggregateObject
    skies_csv_filepath: str

    @classmethod
    def cached_init(cls, skies_csv_filepath, get_samples_per_sky):
        cache_path = "other_data/" + cache_dict_to_path(
            {"skies_csv_filepath": skies_csv_filepath}, "skylightdata"
        )
        cache_exists = exists(cache_path)

        if cache_exists:
            return cls.load(cache_path)
        else:
            obj = cls(skies_csv_filepath)
            obj.save(cache_path)
            return obj

    def __init__(self, skies_csv_filepath, get_samples_per_sky):
        self.get_samples_per_sky = get_samples_per_sky
        self.skies_csv_filepath = skies_csv_filepath
        df = pd.read_csv(skies_csv_filepath)
        df.insert(0, "TensorIndex", list(range(df.shape[0])), False)

        def time_to_seconds(l):
            l_split = map(lambda x: x.split(":"), list(l["Time"]))
            return list(
                map(
                    lambda x: float(int(x[0]) * 60 * 60 + int(x[1]) * 60 + int(x[2])),
                    l_split,
                )
            )

        df.insert(0, "TimeInSeconds", time_to_seconds(df), False)

        self.sky_data_df = df

        features = torch.tensor(
            df[
                [
                    "SunAltitude",
                    "SampleAzimuth",
                    "SampleAltitude",
                    "SunPointAngle",
                    "TimeInSeconds",
                ]
            ].values
        ).float()

        self.normalized_features = (features - torch.mean(features, dim=0)) / (
            torch.max(features, dim=0).values - torch.min(features, dim=0).values
        )

        measurements = df[map(str, self.wavelengths)]
        measurements_t = torch.tensor(measurements.values).float()

        self.data_aggregates = DataAggregateObject(measurements_t)

    def get_image_data(self):
        # TODO: read from disk and cache, make serialize the file with the data in its name separated by _'s

        dfs = [
            self.sky_data_df.iloc[index.get_tensor_locations()]
            for index in self.get_sky_indexes()
        ]

        dfs_from_indexes = list(map(self.get_samples_per_sky, dfs))

        return torch.cat(dfs_from_indexes, 0)

        # return torch.load("other_data\sky_samples_not_random")

    def create_development_data(
        self, training_ratio, excluded_skies: FrozenSet
    ) -> DevelopmentDataObject:
        """
        Training ratio is only an approximate
        - The skies will be divided such that training_data is roughly equal to the training_ratio (although the ratio may be slightly higher) and testing_data will be ~(1-training_ratio) after exclusions
        """

        sky_indexes = self.get_sky_indexes()

        filtered_indexes = list(
            filter(lambda x: x.datetime not in excluded_skies, sky_indexes)
        )

        excluded_indexes = list(
            filter(lambda x: x.datetime in excluded_skies, sky_indexes)
        )

        random.seed(0)
        # random.shuffle(filtered_indexes)

        total_length = sum([index.number_of_samples for index in filtered_indexes])

        length_sum = 0
        train_indexes = []
        test_indexes = []
        for index in filtered_indexes:
            length = index.number_of_samples

            if length_sum / total_length < training_ratio:
                train_indexes.append(index)
            else:
                test_indexes.append(index)

            length_sum += length

        images = self.get_image_data()

        def get_data_bundle(indexes):
            locations = []
            for index in indexes:
                locations += index.get_tensor_locations()

            df = self.sky_data_df.iloc[locations].sample(frac=1, random_state=0)
            tensor_locations = list(df["TensorIndex"])

            return DataBundle(
                df=df,
                x_image_branch=images[tensor_locations],
                x_feature_branch=self.normalized_features[tensor_locations],
                y=self.data_aggregates.normalized_data[tensor_locations],
            )

        return DevelopmentDataObject(
            training_ratio=training_ratio,
            # excluded_skies=excluded_skies,
            data_aggregates=self.data_aggregates,
            src_skies_csv_filepath=self.skies_csv_filepath,
            train_data=get_data_bundle(train_indexes),
            test_data=get_data_bundle(test_indexes),
            # exceptions_data=get_data_bundle(excluded_indexes),
        )

    def normalize(self, raw_spectral_randiance_tensor):
        return (raw_spectral_randiance_tensor - self.data_mean) / self.data_range

    def denormalize(self, normalized_randiance_tensor):
        return (normalized_randiance_tensor * self.data_range) + self.data_mean

    def get_sky_indexes(self):
        prev_date_time = self.get_datetime_str(self.sky_data_df.iloc[0])
        sky_start_index = 0
        skies_indexes = []  # (start_index, length)

        for index, series in self.sky_data_df.iterrows():
            current_date_time = self.get_datetime_str(series)

            # different datetimes means the next set of samples has started to be read
            if prev_date_time != current_date_time:
                skies_indexes.append(
                    SkyIndex(
                        datetime=current_date_time,
                        sky_start_index=sky_start_index,
                        number_of_samples=index - sky_start_index,
                    )
                )

                prev_date_time = current_date_time
                sky_start_index = index

        last_sky_size = self.sky_data_df.shape[0] - sky_start_index
        skies_indexes.append(
            SkyIndex(
                datetime=current_date_time,
                sky_start_index=sky_start_index,
                number_of_samples=last_sky_size,
            )
        )

        return skies_indexes

    def save(self, path: str):
        with open(path, "wb+") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def get_datetime_str(cls, series):
        return f"{series['Date']} {series['Time'][:5]}"


class ModelParent:
    def __init__(self, training_data: SkylightData, device="cuda", save_location=None):
        self.data = training_data

        self.device = device

        self.model = self.create_model(device)
        if save_location:
            self.model.load_state_dict(torch.load(save_location))

    def predict_spectral_radiance(self, img_tensor):
        raise NotImplementedError()

    def save(self, path: str):
        raise NotImplementedError()

    @classmethod
    def create_model(cls):
        raise NotImplementedError()

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError()


@dataclass
class PerformanceSnapshot:
    training_loss: float
    testing_loss: float


@dataclass
class TrainingReport:
    progression_history: List[PerformanceSnapshot]
    best_model_dict: OrderedDict
    number_of_epochs: int
    training_ratio: float
    excluded_skies: set
    src_skies_csv_filepath: str
    batch_size: int
    start_time: float
    end_time: float
    lr: float
    model_used: str  # TODO: change to class, get the class of the model through the model
    # TODO: add rmsd stuff

    def save(self, folder):
        save_path = f"{folder}/training_report__{int(self.end_time)}"
        with open(save_path, "wb+") as f:
            pickle.dump(self, f)
        print(f"Saved at: {save_path}")

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


class TrainingHelper:
    progression_history: List[PerformanceSnapshot] = []
    data: DevelopmentDataObject
    model: nn.Module
    current_best_model_dict: OrderedDict

    def __init__(self, data, model, optimizer, criterion, device):
        self.data = data
        model.eval()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def epoch(self, batch_size) -> PerformanceSnapshot:
        total_training_loss = 0
        total_testing_loss = 0

        train_data: DataBundle = self.data.train_data

        train_n_batches = math.ceil(train_data.df.shape[0] / batch_size)

        self.model.train()
        for i in range(train_n_batches):
            start = i * batch_size
            pred = self.model(
                train_data.x_image_branch[start : start + batch_size].to(self.device),
                train_data.x_feature_branch[start : start + batch_size].to(self.device),
            )
            loss = self.criterion(
                pred, train_data.y[start : start + batch_size].to(self.device)
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_training_loss += float(loss)

        test_data = self.data.test_data
        test_n_batches = math.ceil(test_data.df.shape[0] / batch_size)

        self.model.eval()
        for i in range(test_n_batches):
            start = i * batch_size

            pred = self.model(
                test_data.x_image_branch[start : start + batch_size].to(self.device),
                test_data.x_feature_branch[start : start + batch_size].to(self.device),
            )
            loss = self.criterion(
                pred, test_data.y[start : start + batch_size].to(self.device)
            )

            total_testing_loss += float(loss)

        return PerformanceSnapshot(total_training_loss, total_testing_loss)

    def training_loop(
        self, number_of_epochs, batch_size, model_used, lr, print_progress=True
    ) -> TrainingReport:
        first_snapshot: PerformanceSnapshot
        best_snapshot_so_far: PerformanceSnapshot
        training_start_time = time.time()

        for epoch_number in range(number_of_epochs):
            current_snapshot = self.epoch(batch_size)

            if epoch_number == 0:
                first_snapshot = current_snapshot
                best_snapshot_so_far = current_snapshot
                self.current_best_model_dict = self.model.state_dict()

            if current_snapshot.testing_loss < best_snapshot_so_far.testing_loss:
                self.current_best_model_dict = self.model.state_dict()
                best_snapshot_so_far = current_snapshot

            self.progression_history.append(current_snapshot)
            if print_progress:
                self.print_loss(current_snapshot, beginning=f"{epoch_number}\t")
                self.compare_snapshots(
                    first_snapshot, current_snapshot, training_start_time
                )

        return TrainingReport(
            self.progression_history,
            self.current_best_model_dict,
            number_of_epochs,
            self.data.training_ratio,
            self.data.excluded_skies,
            self.data.src_skies_csv_filepath,
            batch_size,
            training_start_time,
            time.time(),
            lr,
            model_used="Sixth Genth",
        )

    @classmethod
    def print_loss(cls, snapshot: PerformanceSnapshot, beginning=""):
        tab = "\t"
        training_loss_str = f"{snapshot.training_loss:.2f}\t{tab if snapshot.training_loss < 100000 else ''}"
        print(
            f"{beginning}Training Loss: {training_loss_str}Testing Loss: {snapshot.testing_loss:.2f}"
        )

    @classmethod
    def compare_snapshots(
        cls,
        first_snapshot: PerformanceSnapshot,
        second_snapshot: PerformanceSnapshot,
        training_start_time,
    ):
        change_in_training = (
            second_snapshot.training_loss - first_snapshot.training_loss
        )
        change_in_training = change_in_training * 100 / first_snapshot.training_loss
        change_in_training = (
            " " * 15
            + f'{"+" if change_in_training >= 0 else ""}{change_in_training:.2f}'
        )

        change_in_testing = second_snapshot.testing_loss - first_snapshot.testing_loss
        change_in_testing = change_in_testing * 100 / first_snapshot.testing_loss
        change_in_testing = (
            " " * 22 + f'{"+" if change_in_testing >= 0 else ""}{change_in_testing:.2f}'
        )

        print(f"\t{change_in_training}%\t{change_in_testing}%")

        minutes_elapsed_since_start = round((time.time() - training_start_time) / 60)
        print(f"Minutes Training: {minutes_elapsed_since_start}\n")

