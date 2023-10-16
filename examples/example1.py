"""
Data pruning example using keras mobilenet for beyond neural scaling
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
from bns import (
    ExtractedFeatures,
    SelfSupervisedLearningMetric,
    Prune
)

# setup a feature embedder
model = MobileNetV2(weights='imagenet', include_top=False)


def get_extracted_image_features(
        image_directory: str,
        feature_extractor: keras.Model,
        target_size: tuple,
        preprocess_input

) -> ExtractedFeatures:
    extracted_features, image_name = [], []
    for image_index in tqdm(os.listdir(image_directory)):
        image_file = "cluster" + "/" + image_index
        img = image.load_img(image_file, target_size=target_size)
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


def ssl_hcm() -> SelfSupervisedLearningMetric:
    # get embedded space
    embedded_space = get_extracted_image_features(
        feature_extractor=model,
        image_directory='cluster',
        target_size=(96, 96),
        preprocess_input=preprocess_input
    )

    self_supervised_learning_model = ps.hcm.Kmeans(
        data=np.array(embedded_space.features),
        c=2,
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
        number_clusters=self_supervised_learning_model.num_clusters
    )


# set up the SSL class for pruning
prune = Prune(
    ssl_type=ssl_hcm(),
    directory='cluster',
    data_frame_clustered=True,
    random_state=4,
    prune_fraction=0.1
)

# summary of pruning results
prune_data = prune.prune
