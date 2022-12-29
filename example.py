"""
Beyond neural scaling example for prototype-based models using Iris Data
"""
import matplotlib
import numpy as np
import torch
from torch.utils.data import dataloader
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import prototorch as pt
import prototorch.models
import pytorch_lightning as pl
from dataprune1 import Prune
import matplotlib.pyplot as plt

matplotlib.use('QtAgg')


class Vis2D:

    def __init__(self, model, data):
        self.model = model
        self.x_train, self.y_train = pt.utils.parse_data_arg(data)
        self.title = f"Components Visualization with data pruning"
        self.fig = plt.figure(self.title)
        self.cmap = "viridis"
        self.models = ['GLVQ', 'CELVQ', 'CBC', 'GMLVQ', 'LGMLVQ']
        self.colors = ['tab:purple', 'tab:purple', 'g', 'g', 'y', 'y']

    def vis(self, index):
        x_train, y_train = self.x_train, self.y_train
        prototypes = self.model[index].components_layer._components.detach().cpu().numpy() \
            if index == 2 else self.model[index].prototypes

        ax = self.fig.gca()
        ax.cla()
        ax.set_title(f'{self.models[index]} {self.title}')
        ax.axis("off")
        ax.scatter(
            x_train[:, 0],
            x_train[:, 1],
            c=y_train,
            cmap=self.cmap,
            edgecolor="k",
            marker="o",
            s=30,
        )
        ax.scatter(
            prototypes[:, 0],
            prototypes[:, 1],
            c=self.colors,
            cmap=self.cmap,
            edgecolor="k",
            marker="D",
            s=50,
        )
        x_min, x_max = x_train[:, 0].min() - 1, \
                       x_train[:, 0].max() + 1
        y_min, y_max = x_train[:, 1].min() - 1, \
                       x_train[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.1),
            np.arange(y_min, y_max, 0.1))
        z = self.model[index].predict(
            torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
        )
        y_pred = z.reshape(xx.shape)
        ax.contourf(
            xx,
            yy, y_pred, cmap=self.cmap, alpha=0.35)

        plt.pause(0.1)


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


X, y = load_iris(return_X_y=True)
X_real, y_real = X[:, [0, 2]], y

prune: Prune = Prune(
    prune_type=True,
    ssl_type='fcm_init',
    number_cluster=3,
    random_state=40,
    dataset=X
)

print(a := prune.get_prune(
    prune_fraction=0.9,
    pruned_mode='easy',
    prune_type='both')
      )

# get access to pruned dataset
X, y = X_real[a[0]], y_real[a[0]]

# get access to pruned out dataset for evaluating purposes
X_b, y_b = X_real[a[1]], y_real[a[1]]

X = torch.from_numpy(X).to(torch.float32)
y = torch.from_numpy(y).to(torch.float32)

mean_accuracy_cv, mean_accuracy_test = [], []
skf = KFold(n_splits=5, shuffle=True)

for i in range(5):
    accuracy, accuracy_test = [], []
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f'fold {fold}')
        train_ds = torch.utils.data.TensorDataset(
            X[train_index],
            y[train_index]
        )

        test_ds = torch.utils.data.TensorDataset(
            X[test_index],
            y[test_index]
        )

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=124
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=124,
        )

        glvq = pt.models.GLVQ(
            hparams=dict(
                distribution=[2, 2, 2],
                proto_lr=0.00001
            ),
            prototypes_initializer=pt.initializers.SMCI(
                train_ds, noise=0.1
            ),
            optimizer=torch.optim.Adam
        )

        celvq = pt.models.CELVQ(
            hparams=dict(
                distribution=[2, 2, 2],
                proto_lr=0.00001
            ),
            prototypes_initializer=pt.initializers.SMCI(
                train_ds,
                noise=0.1
            ),
            optimizer=torch.optim.Adam
        )
        cbc = pt.models.cbc.CBC(
            hparams=dict(
                distribution=[2, 2, 2],
                margin=0.1,
                proto_lr=0.01,
                optimizer=torch.optim.Adam
            ),
            components_initializer=pt.initializers.SMCI(
                train_ds,
                noise=0.1
            ),
            reasonings_initializer=pt.initializers.PPRI(
                components_first=True
            )
        )

        gmlvq = pt.models.glvq.GMLVQ(
            hparams=dict(
                input_dim=2,
                latent_dim=2,
                distribution=[2, 2, 2],
                proto_lr=0.01,
                bb_lr=0.01
            ),
            prototypes_initializer=pt.initializers.SMCI(
                train_ds,
                noise=0.1
            )
        )

        lgmlvq = pt.models.glvq.LGMLVQ(
            hparams=dict(
                input_dim=2,
                latent_dim=2,
                distribution=[2, 2, 2],
                proto_lr=0.01,
                bb_lr=0.01
            ),
            prototypes_initializer=pt.initializers.SMCI(
                train_ds,
                noise=0.1
            )
        )

        model = [glvq, celvq, cbc, gmlvq, lgmlvq]

        model[i].apply(reset_weights)

        trainer = pl.Trainer(max_epochs=1000)
        trainer.fit(model[i], train_loader)

        outputs = model[i].predict(torch.Tensor(X[test_index]))

        accuracy.append(accuracy_score(y[test_index], outputs))

        outputs_t = model[i].predict(torch.Tensor(X_b))

        accuracy_test.append(accuracy_score(y_b, outputs_t))

        vis2d = Vis2D(model=model, data=train_ds)
        vis2d.vis(index=i)

    mean_accuracy_cv.append(np.mean(accuracy))
    mean_accuracy_test.append(np.mean(accuracy_test))

print(mean_accuracy_cv)
print(mean_accuracy_test)

