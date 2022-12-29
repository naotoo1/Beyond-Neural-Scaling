"""
Module implementation of beyond neural scaling laws beating power scaling laws through data pruning
"""
import argparse
import os
import shutil

from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing import image
from keras.utils import img_to_array
from skfuzzy import cmeans_predict
from sklearn.cluster import KMeans
import skfuzzy as fuzz
import pandas as pd
import numpy as np
from tqdm import tqdm
from gng import GrowingNeuralGas


def get_extracted_image_features(directory: str):
    """

    :param directory: str: path/directory with to the images
    :return: array-like: extracted features and corresponding image names
    """
    model = MobileNetV2(weights='imagenet', include_top=False)
    extracted_features, image_name = [], []
    for image_index in tqdm(os.listdir(directory)):
        image_file = "cluster" + "/" + image_index
        img = image.image_utils.load_img(image_file, target_size=(96, 96))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = model.predict(img)
        features = features.flatten()
        extracted_features.append(features)
        image_name.append(image_index)
    return extracted_features, image_name


def get_cluster_distance_space(distance_space, image_names):
    """


    :param distance_space: array-like: 2D list containing the distance space
        wrt to the centroids
    :param image_names: list: images names
    :return: clustered distance space for a given cluster label.
    """

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

    return cluster_distance_space


def get_sorted_space(x, y, z):
    init_count = 0
    for count in x:
        count = count + init_count
        y.append(z[init_count:count])
        init_count = count


def get_pruned_easy_hard_examples(sorted_distance_space, prune_fraction):
    """

    :param sorted_distance_space: list: list with indexes
        of sorted distanced space
    :param prune_fraction: float: prune percentage or fraction
    :return: array-like: pruned list with indices of sorted distance space
    """

    maximum_prune = int(
        len(sorted_distance_space) -
        np.ceil((1 - prune_fraction) * len(sorted_distance_space))
    )
    return sorted_distance_space[:maximum_prune], \
        sorted_distance_space[maximum_prune:]


def get_prune_set(pruned_indexes, prune_mode):
    """

    :param pruned_indexes:list: list with pruned indexes
    :param prune_mode:str: easy , hard and both. If none default is both
    :return: array-like: list with file names of pruned data set.
    """
    if prune_mode == 'easy':
        return pruned_indexes[0]
    if prune_mode == 'hard':
        return pruned_indexes[1]
    if prune_mode == 'both':
        return pruned_indexes
    return None


def get_prune_set_folder(distance_space, image_names, prune_fraction, prune_labels):
    """

    :param prune_labels:
    :param prune_fraction:float: fraction of the prune
    :param distance_space:computed distances from the nearest prototype
    :param image_names:
    :return:
    """
    clustered_distance_space = get_cluster_distance_space(
        distance_space,
        image_names
    )

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


def check_directory(number_clusters, folder_name):
    """

    :param number_clusters: int: number of clusters
    :param folder_name: str: folder name
    :return: cleans the directory for new runs regarding new runs
    """
    directory = [
        f"./{folder_name}_{folder_index}"
        for folder_index in range(number_clusters)
    ]
    for folder in directory:
        if os.path.exists(folder) and os.path.isdir(folder):
            shutil.rmtree(folder, ignore_errors=True)
        if not os.path.exists(folder):
            os.makedirs(folder)


def check_directory_prune(number_clusters, folder_name):
    """

    :param number_clusters: int: number of clusters
    :param folder_name: str: folder name
    :return: cleans the directory for new runs regarding new runs
    """

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


def ssl_growing_neural_gas(x, y=2):
    gng = GrowingNeuralGas(np.array(x).transpose())
    gng.fit_network(
        e_b=0.1,
        e_n=0.006,
        a_max=10,
        l=200,
        a=0.5,
        d=0.995,
        passes=y,
        plot_evolution=True
    )
    return gng.number_of_clusters()


class SSL:
    def __init__(self, number_cluster, random_state, directory='cluster'):
        self.number_cluster = number_cluster
        self.random_state = random_state
        self.directory = directory
        self.number_topology = []

    def get_embedded_space(self):
        image_features, image_names = get_extracted_image_features(
            directory=self.directory
        )
        return image_features, image_names

    def ssl_kmeans(self, init='k-means++', n_init='auto'):
        embedded_space, image_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)  #
        self.number_topology.append(number_of_clusters)  #
        self_supervised_learning_model = KMeans(
            n_clusters=number_of_clusters,
            random_state=self.random_state,
            init=init,
            n_init=n_init
        )
        self_supervised_learning_model.fit(embedded_space)

        prototype_responsibilities = self_supervised_learning_model.fit_transform(
            embedded_space
        )

        cluster_labels = self_supervised_learning_model.labels_
        cluster_centers = self_supervised_learning_model.cluster_centers_

        return prototype_responsibilities, cluster_labels, \
            image_names, cluster_centers, embedded_space

    def ssl_fcm(self):
        embedded_space, image_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)  #
        self.number_topology.append(number_of_clusters)  #
        cntr, u_matrix, u_matrix_init, distance_space, \
            objective_function_history, num_inter_run, \
            fuzzy_partition_coefficient = fuzz.cmeans(
            data=np.array(embedded_space).transpose(),
            c=number_of_clusters,
            m=2,
            error=0.001,
            maxiter=1000,
            init=None,
            seed=self.random_state
        )
        cluster_labels = [np.argmax(i) for i in u_matrix.transpose()]

        return distance_space.transpose(), cluster_labels, image_names

    def ssl_fcm_init(self):
        prototype_responsibilities, cluster_labels, \
            image_names, cluster_centers, embedded_space = self.ssl_kmeans()
        u_matrix, u_matrix_init, distance_space, object_function_history, \
            num_iter_run, fuzzy_partition_coefficient = cmeans_predict(
            test_data=np.array(embedded_space).transpose(),
            cntr_trained=cluster_centers,
            m=2,
            error=0.001,
            maxiter=1000,
            init=None,
            seed=self.random_state
        )

        cluster_labels = [np.argmax(i) for i in u_matrix.transpose()]
        return distance_space.transpose(), cluster_labels, image_names

    def get_number_clusters(self, x):
        """

        :param x: embeded space
        :return: number of clusters.
        """
        if isinstance(self.number_cluster, int):
            return self.number_cluster
        if self.number_cluster == 'default':
            number_cluster = 2
            return number_cluster
        if self.number_cluster == 'auto':
            learned_topologies = ssl_growing_neural_gas(x)
            if learned_topologies < 2:
                number_cluster = 2
                return number_cluster
            return learned_topologies
        return None


class Prune(SSL):
    """
    Prune
    params:

    random_state: int:
        Random seed
    prune_fraction: float:
        fraction or percentage of dataset to prune
    prune_mode: str:
        easy , hard and both
    prune_type: bool:
        True or False .Indicating the ceiling or floor of the prune results.


    """

    def __init__(self, prune_type, ssl_type, data_frame_clustered,
                 number_cluster, random_state, directory):
        super().__init__(number_cluster, random_state, directory)
        self.prune_mode = ['easy', 'hard']
        self.prune_type = prune_type
        self.cluster_folder_name = self.directory
        self.prune_folder_name = 'prune'
        self.ssl_type = ssl_type
        self.date_frame_clustered = data_frame_clustered

        if self.ssl_type == 'kmeans':
            self.distance_space, self.clustered_labels, \
                self.clustered_image_names, self.cluster_centers, \
                self.embedded_space = self.ssl_kmeans()

        if self.ssl_type == "fcm":
            self.distance_space, self.clustered_labels, \
                self.clustered_image_names = self.ssl_fcm()

        if self.ssl_type == "fcm_init":
            self.distance_space, self.clustered_labels, \
                self.clustered_image_names = self.ssl_fcm_init()

    def get_number_topologies(self):
        """

        :return: The learned topologies.
        """
        return self.number_topology

    def get_cluster_results(self):
        """

        :return: panda dataframe and populate folders wit the clustering results.
        """
        if isinstance(self.number_cluster, str):
            self.number_cluster = self.get_number_topologies()[0]

        check_directory(
            number_clusters=self.number_cluster,
            folder_name=self.cluster_folder_name
        )

        image_cluster_df = pd.DataFrame(
            self.clustered_image_names,
            columns=['image_names']
        )

        image_cluster_df["cluster_label"] = self.clustered_labels
        for cluster_index in range(self.number_cluster):
            for index in range(len(image_cluster_df)):
                if image_cluster_df['cluster_label'][index] == cluster_index:
                    shutil.copy(os.path.join(
                        self.directory,
                        image_cluster_df['image_names'][index]),
                        f'{self.directory}_{cluster_index}'
                    )
        if self.date_frame_clustered:
            return image_cluster_df
        return None

    def prune(self, prune_fraction):

        if isinstance(self.number_cluster, str):
            self.number_cluster = self.get_number_topologies()[0]

        check_directory_prune(
            number_clusters=self.number_cluster,
            folder_name=self.prune_folder_name
        )

        get_prune_set_folder(
            distance_space=self.distance_space,
            image_names=self.clustered_image_names,
            prune_fraction=prune_fraction,
            prune_labels=self.prune_mode
        )


def self_supervised_learning_metric():
    parser = argparse.ArgumentParser(
        description='Executes self supervised learning metric for data pruning '
    )

    parser.add_argument(
        "-m",
        "--ssl_model",
        type=str,
        metavar='',
        required=True,
        help=" self supervised learning metric model type"
    )

    parser.add_argument(
        "-n",
        "--number_of_clusters",
        metavar='', required=True,
        help=" number of cluster under consideration"
    )

    parser.add_argument(
        "-x",
        "--prune_fraction",
        type=float, metavar='',
        required=True,
        help=" fraction for pruning the dataset"
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument('-p',
                       '--prune',
                       help='prune data set ',
                       action="store_true")

    group.add_argument('-b',
                       '--get_cluster_results',
                       help=' populates cluster folders with clustering results',
                       action="store_true")
    group.add_argument('-a',
                       '--all',
                       help=' populates cluster folders with clustering results'
                            'prune data set for all specifications',
                       action="store_true")

    args = parser.parse_args()

    prune: Prune = Prune(
        prune_type=True,
        ssl_type=args.ssl_model,
        number_cluster=args.number_of_clusters,
        random_state=40,
        directory='cluster',
        data_frame_clustered=args.get_cluster_results
    )

    if args.prune:
        prune.prune(prune_fraction=args.prune_fraction)

    if args.get_cluster_results:
        prune.get_cluster_results()

    if args.all:
        prune.get_cluster_results()
        prune.prune(prune_fraction=args.prune_fraction)


if __name__ == '__main__':
    self_supervised_learning_metric()
