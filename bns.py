"""
Module implementation of beyond neural scaling laws beating power scaling laws through data pruning
"""
import os
import shutil
from typing import Union
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


class PruneMode(str, Enum):
    EASY = 'easy'
    HARD = 'Hard'
    BOTH = 'both'


@dataclass
class ExtractedFeatures:
    features: Union[np.ndarray, list]
    image_names: list[str]


@dataclass
class DistanceSpace:
    cluster_distance: list[list]


@dataclass
class SortedDistanceSpace:
    easy: list
    hard: list


@dataclass
class SelfSupervisedLearningMetric:
    distance_space: list[list]
    cluster_labels: list
    image_names: list[str]
    number_clusters: int


def get_cluster_distance_space(
        distance_space: list[list],
        image_names: list[str]
) -> DistanceSpace:
    subspace_information_list, subspace_list, counter = [], [], []
    label_basis, sorted_subspace = [], []
    nearest_sort, cluster_distance_space = [], []

    distance_space_information = [
        [image_names[sub_space_index], np.argmin(distance_subspace),
         np.min(distance_subspace)]
        for sub_space_index, distance_subspace in enumerate(distance_space)
    ]

    number_cluster = len(np.unique(
        [i[1] for i in distance_space_information])
    )

    for label in range(number_cluster):
        count = 0
        for index, distance_subspace_information in \
                enumerate(distance_space_information):
            if distance_subspace_information[1] == label:
                count += 1
                subspace_list.append(
                    distance_subspace_information
                )
        counter.append(count)

    # get data subspace information on label basis
    get_sorted_space(
        counter,
        subspace_information_list,
        subspace_list
    )

    for subspace_information in subspace_information_list:
        for index, subspace in enumerate(subspace_information):
            label_basis.append(subspace[2])

    #  get the distance subspace information based on labels
    get_sorted_space(
        counter,
        sorted_subspace,
        label_basis
    )

    # get sorted subspace index based on closest prototype
    for i in sorted_subspace:
        nearest_sort.append(np.argsort(i))

    # get the sort files base on subspace information
    for i in range(len(nearest_sort)):
        cluster_distance_space.append(
            [subspace_information_list[i][v][0]
             for j, v in enumerate(nearest_sort[i])]
        )

    return DistanceSpace(
        cluster_distance=cluster_distance_space
    )


def get_sorted_space(x, y, z):
    init_count = 0
    for count in x:
        count = count + init_count
        y.append(z[init_count:count])
        init_count = count


def get_pruned_easy_hard_examples(
        sorted_distance_space: list[int],
        prune_fraction: float) -> SortedDistanceSpace:
    maximum_prune = int(
        len(sorted_distance_space) -
        np.ceil((1 - prune_fraction) * len(sorted_distance_space))
    )
    return SortedDistanceSpace(
        easy=sorted_distance_space[:maximum_prune],
        hard=sorted_distance_space[maximum_prune:]
    )


def get_prune_set(
        pruned_indexes: SortedDistanceSpace,
        prune_mode: str
):
    if prune_mode == PruneMode.EASY.value:
        return pruned_indexes.easy
    if prune_mode == PruneMode.HARD.value:
        return pruned_indexes.hard
    if prune_mode == PruneMode.BOTH.value:
        return pruned_indexes.easy, pruned_indexes.hard
    return None


def get_prune_set_folder(
        distance_space: list[list],
        image_names: list[str],
        prune_fraction: float,
        prune_labels):
    clustered_distance_space = get_cluster_distance_space(
        distance_space,
        image_names
    ).cluster_distance

    for i in range(len(clustered_distance_space)):
        pruned = get_prune_set(get_pruned_easy_hard_examples(
            clustered_distance_space[i],
            prune_fraction),
            'both'
        )
        for prune_index, prune_list in enumerate(pruned):
            for index, file in enumerate(prune_list):
                shutil.copy(
                    f'cluster/{file}',
                    f'prune_{i}_{prune_labels[prune_index]}')


def check_directory(
        number_clusters: int,
        folder_name: str
):
    directory = [
        f"./{folder_name}_{folder_index}"
        for folder_index in range(number_clusters)
    ]
    for folder in directory:
        if os.path.exists(folder) and os.path.isdir(folder):
            shutil.rmtree(folder, ignore_errors=True)
        if not os.path.exists(folder):
            os.makedirs(folder)


def check_directory_prune(
        number_clusters: int,
        folder_name: str
):
    directory = [
        [f"./{folder_name}_{cluster_label}_{prune_type}"
         for prune_type in ['easy', 'hard']] for cluster_label
        in range(number_clusters)
    ]

    directory_prune = [
        folder for prune_folders in directory for folder in prune_folders
    ]

    for folder in directory_prune:
        if os.path.exists(folder) and os.path.isdir(folder):
            shutil.rmtree(folder, ignore_errors=True)
        if not os.path.exists(folder):
            os.makedirs(folder)


@dataclass
class Prune:
    ssl_type: SelfSupervisedLearningMetric
    directory: str
    data_frame_clustered: bool
    random_state: Union[int, None]
    prune_fraction: float
    prune_folder_name: str = 'prune'
    prune_mode: list[str] = field(
        default_factory=lambda: ['easy', 'hard']
    )

    @property
    def get_cluster_results(self) -> Union[pd.DataFrame, None]:

        check_directory(
            number_clusters=self.ssl_type.number_clusters,
            folder_name=self.directory
        )

        image_cluster_df = pd.DataFrame(
            self.ssl_type.image_names,
            columns=['image_names']
        )

        image_cluster_df["cluster_label"] = self.ssl_type.cluster_labels
        for cluster_index in range(self.ssl_type.number_clusters):
            for index in range(len(image_cluster_df)):
                if image_cluster_df['cluster_label'][index] == cluster_index:
                    shutil.copy(os.path.join(
                        self.directory,
                        image_cluster_df['image_names'][index]),
                        f'{self.directory}_{cluster_index}'
                    )
        if self.data_frame_clustered:
            return image_cluster_df
        return None

    @property
    def prune(self):
        check_directory_prune(
            number_clusters=self.ssl_type.number_clusters,
            folder_name=self.prune_folder_name
        )

        return get_prune_set_folder(
            distance_space=self.ssl_type.distance_space,
            image_names=self.ssl_type.image_names,
            prune_fraction=self.prune_fraction,
            prune_labels=self.prune_mode
        )
