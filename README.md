 # Active Coalition of Variables (ACV):

Active Coalition of Variables (ACV) is a Python Package that aims to explain any machine learning model or data. 
It implemented the SDP explanations approaches of the Paper: **Consistent Sufficient Explanations and Minimal Local Rules for
explaining the decision of any classifier or regressor**.
 
## Requirements
Python 3.7+ 

**OSX**: ACV uses Cython extensions that need to be compiled with multi-threading support enabled. 
The default Apple Clang compiler does not support OpenMP.
To solve this issue, obtain the lastest gcc version with Homebrew that has multi-threading enabled: 
see for example [pysteps installation for OSX.](https://pypi.org/project/pysteps/1.0.0/)

**Windows**: Install MinGW (a Windows distribution of gcc) or Microsoft’s Visual C

Install the required packages:

```
$ pip install -r requirements.txt
```

## Installation

Clone the repo and run the following command in the main directory
```
$ python setup.py install
```


 
## Agnostic explanations
The Agnostic approaches explain any data (**X**, **Y**) or model (**X**, **f(X)**) using the following 
explanation methods:

* Same Decision Probability (SDP) and **Sufficient Explanations**
* **Sufficient Rules**

See the paper for more details.

**I. First, we need to fit our explainer (ACXplainers) to input-output of the data **(X, Y)** or model
**(X, f(X))** if we want to explain the data or the model respectively.**

```python
from acv_explainers import ACXplainer

# It has the same params as a Random Forest, and it should be tuned to maximize the performance.  
acv_xplainer = ACXplainer(classifier=True, n_estimators=50, max_depth=5)
acv_xplainer.fit(X_train, y_train)

roc = roc_auc_score(acv_xplainer.predict(X_test), y_test)
```

**II. Then, we can load all the explanations in a webApp as follow:**

```python 
import acv_app
import os

# compile the ACXplainer
acv_app.compile_ACXplainers(acv_xplainer, X_train, y_train, X_test, y_test, path=os.getcwd())

# Launch the webApp
acv_app.run_webapp(pickle_path=os.getcwd())
```
![Capture d’écran de 2021-11-03 19-50-12](https://user-images.githubusercontent.com/40361886/140174581-4c5bf018-05ad-49e0-b005-2a65453626e1.png)



**III. Or we can compute each explanation separately as follow:**

#### Same Decision Probability (SDP)
The main tool of our explanations is the Same Decision Probability (SDP). Given <img src="https://latex.codecogs.com/gif.latex?x%20%3D%20%28x_S%2C%20x_%7B%5Cbar%7BS%7D%7D%29" />, the same decision probability <img src="https://latex.codecogs.com/gif.latex?SDP_S%28x%2C%20f%29" /> of variables <img src="https://latex.codecogs.com/gif.latex?x_S" />  is the probabilty that the prediction remains the same when we fixed variables 
<img src="https://latex.codecogs.com/gif.latex?X_S=x_S" /> or when the variables <img src="https://latex.codecogs.com/gif.latex?X_{\bar{S}}" /> are missing.
* **How to compute <img src="https://latex.codecogs.com/gif.latex?SDP_S%28x%2C%20f%29" />  ?**

```python
sdp = acv_xplainer.compute_sdp_rf(X, S, data_bground) # data_bground is the background dataset that is used for the estimation. It should be the training samples.
```
#### Minimal Sufficient Explanations
The Sufficient Explanations is the Minimal Subset S such that fixing the values <img src="https://latex.codecogs.com/gif.latex?X_S=x_S" /> 
permit to maintain the prediction with high probability <img src="https://latex.codecogs.com/gif.latex?\pi" />.
See the paper for more details. 

* **How to compute the Minimal Sufficient Explanation <img src="https://latex.codecogs.com/gif.latex?S^\star" /> ?**
    
    The following code return the Sufficient Explanation with minimal cardinality. 
```python
sdp_importance, min_sufficient_expl, size, sdp = acv_xplainer.importance_sdp_rf(X, y, X_train, y_train, pi_level=0.9)
```

* **How to compute all the Sufficient Explanations  ?**

    Since the Minimal Sufficient Explanation may not be unique for a given instance, we can compute all of them.
```python
sufficient_expl, sdp_expl, sdp_global = acv_xplainer.sufficient_expl_rf(X, y, X_train, y_train, pi_level=0.9)
```

#### Local Explanatory Importance
For a given instance, the local explanatory importance of each variable corresponds to the frequency of 
apparition of the given variable in the Sufficient Explanations. See the paper for more details. 

* **How to compute the Local Explanatory Importance ?**

```python
lximp = acv_xplainer.compute_local_sdp(d=X_train.shape[1], sufficient_expl)
```

#### Local rule-based explanations
For a given instance **(x, y)** and its Sufficient Explanation S such that <img src="https://latex.codecogs.com/gif.latex?SDP_S(\boldsymbol{x};&space;y)&space;\geq&space;\pi" title="SDP_S(\boldsymbol{x}; y) \geq \pi" />, we compute a local minimal rule which contains **x** such 
that every observation **z** that satisfies this rule has <img src="https://latex.codecogs.com/gif.latex?SDP_S(\boldsymbol{z};&space;y)&space;\geq&space;\pi" title="SDP_S(\boldsymbol{z}; y) \geq \pi" />. See the paper for more details

* **How to compute the local rule explanations ?**

```python
sdp, rules, _, _, _ = acv_xplainer.compute_sdp_maxrules(X, y, data_bground, y_bground, S) # data_bground is the background dataset that is used for the estimation. It should be the training samples.
```

#### Global rule-based explanations

We can uses the Sufficient Rules to construct a global rule-based models as follow:

```python
# step 1: compute the Sufficient Explanations
sdp_importance, sdp_index, size, sdp = acv_xplainer.importance_sdp_rf(X, y, X_train, y_train, stop=False, 
                                                                      pi_level=0.9)
# step 2: compute the Sufficient Rules
sdp, rules, sdp_all, rules_data, w = acv_xplainer.compute_sdp_maxrules(X_train.iloc[:max_size], y_train[:max_size].astype(np.double),
                                                         X_train, y_train.astype(np.double), S_star, verbose=True)

# step 3:  give the Sufficient Rules to acv_xplainer to build the global model
acv_xplainer.fit_global_rules(X_train, y_train, rules, S_star)
```

## Notebooks

You can find the experiments of the paper in the [notebook directory](https://github.com/icml2022consistentexpl/ConsistentExplanationsICML/tree/main/notebooks).
