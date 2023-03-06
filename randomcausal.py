import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error

np.random.seed(42)

# Number of samples
n_samples = 1000

# Continuous covariate X1 (e.g., age) - normally distributed
X1_mean = 40
X1_std = 10
X1 = np.random.normal(X1_mean, X1_std, n_samples)

# Discrete covariate X2 (e.g., education level) - categorical with 3 categories
X2_categories = [0, 1, 2]
X2_prob = [0.5, 0.3, 0.2]
X2 = np.random.choice(X2_categories, n_samples, p=X2_prob)

# Treatment variable T - binary treatment (0 or 1) with varying probability depending on X1 and X2
def treatment_probability(x1, x2):
    return 1 / (1 + np.exp(-(0.1 * x1 + 0.5 * x2 - 3)))

T_prob = treatment_probability(X1, X2)
T = np.random.binomial(1, T_prob)

# Outcome variable Y - generated using the treatment and covariates
def outcome_function(t, x1, x2):
    confounder = 0.2 * x1 + 0.8 * x2
    causal_effect = 5 * t
    error = np.random.normal(0, 2, len(t))
    return confounder + causal_effect + error

Y = outcome_function(T, X1, X2)

# Combine the generated data into a DataFrame
data = pd.DataFrame({"X1": X1, "X2": X2, "T": T, "Y": Y})
