# Beyond-Neural-Scaling
 Implementation of beyond nueral scaling beating power laws for both deep models and prototype-based model.
 
 
 ## How to use
The implementation covers deep models with illustration for computer vision (object detection for edge devices) in the ```dataprune.py``` module. 

There is also an extension of the implementation to cover ML practioners in the area of prototype-based models with illustraion for LVQ(s) in the ```dataprune1.py``` module.

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

