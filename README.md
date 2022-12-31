[![Python: 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Pytorch: 1.11](https://img.shields.io/badge/pytorch-1.11-orange.svg)](https://pytorch.org/blog/pytorch-1.11-released/)
[![Prototorch: 0.7.5](https://img.shields.io/badge/prototorch-0.7.5-blue.svg)](https://pypi.org/project/prototorch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)



# Beyond-Neural-Scaling Laws
 Implementation of beyond neural scaling beating power laws through data pruning.

The implementation covers deep learning and prototype-based models and shows how the optimal pruning algorithm can transition from power law scaling to exponential law scaling.

We exemplify practical use cases for TensorFlow object detection for mobile/edge devices and prototype-based models with learning vector quantization.

 
 ## How to use
The implementation covers deep models with illustrations for computer vision in the dataprune.py module.

There is also an extension of the implementation to cover ML practitioners in the area of prototype-based models with illustrations for LVQ(s) in the dataprune1.py module.

```python
usage: dataprune.py [-h] -m  -n  -x  [-p | -b | -a]

Executes self supervised learning metric for data pruning

options:
  -h, --help            show this help message and exit
  -m , --ssl_model      self supervised model type
  -n , --number_of_clusters 
                        number of cluster under consideration
  -x , --prune_fraction 
                        fraction for pruning the dataset
  -p, --prune           prune data set
  -b, --get_cluster_results
                        populates cluster folders with clustering results
  -a, --all             populates cluster folders with clustering results and pruned data set for all specifications
```

## References

<a id="1">[1]</a> 
Sorscher, Ben, et al.
Beyond neural scaling laws: beating power law scaling via data pruning.
arXiv preprint arXiv:2206.14486 (2022).

