"""
Data pruning example using keras mobilenet for beyond neural scaling  with learned data topologies
"""

import os
import keras
import numpy as np
from keras.preprocessing import image
from keras.utils import img_to_array
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
import prosemble.models as ps
from tqdm import tqdm
from bns import ExtractedFeatures, SelfSupervisedLearningMetric, Prune
from cluster_analysis.gng import GrowingNeuralGas

# setup a feature embedder
model = MobileNetV2(weights='imagenet', include_top=False)


# set up feature extractor
def get_extracted_image_features(
        image_directory: str,
        feature_extractor: keras.Model,
        target_size: tuple,
        preprocess_input

) -> ExtractedFeatures:
    extracted_features, image_name = [], []
    for image_index in tqdm(os.listdir(image_directory)):
        image_file = "cluster" + "/" + image_index
        img = image.load_img(
            image_file,
            target_size=target_size
        )
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = feature_extractor.predict(img)
        features = features.flatten()
        extracted_features.append(features)
        image_name.append(image_index)
    return ExtractedFeatures(
        features=extracted_features,
        image_names=image_name
    )


# set up the model to learn inherent topologies
def get_number_of_cluster(ssl):
    gng = GrowingNeuralGas(ssl)
    gng.fit_network(
        e_b=0.1,
        e_n=0.006,
        a_max=10,
        l=200,
        a=0.5,
        d=0.995,
        passes=2,
        plot_evolution=True
    )
    learned_topologies = gng.number_of_clusters()
    if learned_topologies < 2:
        return 2
    return learned_topologies


# set up a prototype-based self supervised metric
def ssl_hcm() -> SelfSupervisedLearningMetric:
    # get embedded space
    embedded_space = get_extracted_image_features(
        feature_extractor=model,
        image_directory='cluster',
        target_size=(96, 96),
        preprocess_input=preprocess_input
    )

    # automate the number of clusters selections
    number_of_clusters = get_number_of_cluster(
        np.array(embedded_space.features).transpose()
    )

    self_supervised_learning_model = ps.hcm.Kmeans(
        data=np.array(embedded_space.features),
        c=number_of_clusters,
        epsilon=0.001,
        num_inter=1000,
        ord=None,
        set_prototypes=None,
        plot_steps=False
    )
    self_supervised_learning_model.fit()
    cluster_labels = self_supervised_learning_model.predict()
    distance_space = self_supervised_learning_model.get_distance_space(
        embedded_space.features
    )

    return SelfSupervisedLearningMetric(
        distance_space=distance_space,
        cluster_labels=cluster_labels,
        image_names=embedded_space.image_names,
        number_clusters=number_of_clusters

    )


# set up the data pruning class for pruning.
prune = Prune(
    ssl_type=ssl_hcm(),
    directory='cluster',
    data_frame_clustered=True,
    random_state=42,
    prune_fraction=0.2
)

# summary of pruning results
prune_data = prune.prune
