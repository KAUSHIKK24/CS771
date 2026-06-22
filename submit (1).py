import numpy as np
import sklearn

# You are allowed to import any submodules of numpy or sklearn e.g. sklearn.metrics.accuracy_score to calculate accuracy of a learnt model
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_map, my_params etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For a Hamming PUF with 32-bit challenges, the response is determined by:
#   r(c) = 1  if  h_odd(d) * h_even(d) > t,  else 0
# where d = s XOR c.  Expanding h_odd * h_even yields a degree-2 polynomial
# in the challenge bits c_i, with cross terms only between odd-indexed and
# even-indexed positions.  The feature map below captures exactly this
# structure: 32 original bits + 16*16 = 256 odd x even cross products = 288 dims.

################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Map 32-bit raw challenges to 288-dimensional feature vectors.
	# Features: [c_1,...,c_32, {c_i * c_j for i odd, j even}]
	# Odd positions  (1-indexed: 1,3,...,31) -> 0-indexed: 0,2,...,30
	# Even positions (1-indexed: 2,4,...,32) -> 0-indexed: 1,3,...,31

	n = X.shape[0]
	X_odd  = X[:, 0::2]                                       # n x 16
	X_even = X[:, 1::2]                                       # n x 16
	cross  = (X_odd[:, :, None] * X_even[:, None, :]).reshape(n, -1)  # n x 256

	X_map = np.hstack([X, cross])                              # n x 288
	return X_map

################################
# Non Editable Region Starting #
################################
def my_params( X_map, X_raw, y ):
################################
#  Non Editable Region Ending  #
################################

	# Return fixed hyperparameters for sklearn.svm.LinearSVC.
	# These values were selected after experimentation on the public dataset:
	#   C = 1.0        — good regularisation balance
	#   loss           — squared_hinge (default, stable)
	#   tol  = 1e-4    — default convergence tolerance
	#   max_iter       — generous iteration budget for convergence
	#   dual = True    — efficient when n_samples > n_features is not large

	my_params = {
		'C': 1.0,
		'loss': 'squared_hinge',
		'penalty': 'l2',
		'tol': 1e-4,
		'max_iter': 1000,
		'dual':False

	}
	return my_params
