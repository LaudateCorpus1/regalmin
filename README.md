# `Regularized PCA`
This code aims at estimating multiple principal components of a observed matrix given constraints (such as sparsity). Two alternative approaches consist of
    **(1) Deflation.** Compute a single principal component given the constraints, substract the resulting rank-1 matrix from the observation and iterate.
    **(2) Regularization.** Formulate the problem as a single optimization problem on the set of principal components and estimate them jointly.

We refer to the [Section 5.2](https://github.com/AdRoll/regalmin/tree/first_commit/techreport) and references therein for a detailed description of the algorithm. Note that we are not running the algorithm on the sample covariance matrix but on the observation matrix that has size `number of observations` x `number of variables`. This problem is sometimes referred to as the Canonical Component Analysis. 

## Usage

Make sure `numpy`, `scipy` and also `fbpca` are installed, otherwise install them using `sudo pip install numpy` etc.

### Deflation
First set a dictionary of parameters
```
    deflation_parameters = {
        'r': 5,
        'q': 20,
        'k': 100, 
        'right_regularizer': 'k_support',
        'regularization': .2, 
        'max_iter': 50
        }
```
Then define the deflated PCA object and run the algorithm:
```
    deflated = DeflatedStructuredPCA(observed_matrix, deflation_parameters)
    deflated.compute()
```

### Regularization

Given a dictionary of parameters
```
    regularized_parameters = {
        'r': 5,
        'q': 20,
        'k': 100, 
        'right_regularizer': 'k_support',
        'regularization': .2, 
        'max_iter': 50
        }
```
define the regularized PCA object and optimize:
```
    regularized = RegularizedLeastSquares(observed_matrix, regularized_parameters)
    regularized.optimize()
```


## Examples
 
Generate synthetic data using ```get_synthetic_data(n=100, m=150, signal_to_noise_ratio=2., r=5, q=10)```
 from `sparse_pca_comparisons.py`
### Optimization
Visualize the objective function and the factors gradient norms (used as stopping criteria) by setting 
```
    regularized_parameters['display_objective'] = True 
``` 
![alt tag](https://github.com/AdRoll/regalmin/objective.png)
Note that the objective function being non-convex the gradient norms have no guarantee to decrease. The objective value decreases at every iteration though. 
### Performance
An example on gene-expression data from http://ccb.nki.nl/data/. Use ```load_gene_expression_data.py``` to pre-process the data and run ```sparse_pca_comparison_witten_kqtrace.py```.  This computes the regularized sparse PCA for a range of regularization parameters. Higher parameters result in sparser principal components (right factors). Hence we get a set of `(sparsity, distance to observation)` values that we can represent as a plot. Note that when the number of factors computed is larger than 1, evaluating the results using the explained variance is meaningful only if the factors are constrained to be orthogonal.

<img align="center" src="https://github.com/AdRoll/regalmin/tree/first_commit/reg_vs_deflation_4k_iter.png" width=450/>

We also compare this algorithm with an Active Set algorithm proposed in http://arxiv.org/abs/1407.5158 and show that it performs better in the sense that for the same rank or the same number of nunzeros the obtained low-rank approximation has a lower reconstruction error. This comes with the caveat that the factors obtained using the Active Set algorithm have the exact desired sparsity, whereas the factors obtained using this algorithm have a larger number of nonzeros usually. This comes from the behavior of the k-support norm's proximal operator.

<img align="center" src="https://github.com/AdRoll/regalmin/tree/first_commit/alternate_min_active_set_comparison_4.png" width=450/>

## Future work / TODO

### Other loss functions


### Other regularizers

### Other algorithms

## References
Emile Richard, Guillaume Obozinski, Jean-Philippe Vert. **Tight convex relaxations for sparse matrix factorization** in *Advances in Neural Information Processing Systems 27 (NIPS 2014)*

Daniela Witten, Robert Tibshirani, and Trevor Hastie. **A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis** in *Biostatistics (2009)*

