import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load and prepare data
data = pd.read_csv('heart.csv')
subset_data = data[['age', 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'target']]
print(subset_data.head())

# Define and fit the Bayesian Network
model = BayesianNetwork([('age', 'target'), ('sex', 'target'), ('cp', 'target'), 
                         ('thalach', 'target'), ('exang', 'target'), ('oldpeak', 'target')])
model.fit(subset_data, estimator=MaximumLikelihoodEstimator)

# Perform inference
inference = VariableElimination(model)
evidence = {'age': 63, 'sex': 1, 'cp': 1, 'thalach': 150, 'exang': 0, 'oldpeak': 2.3}
result = inference.query(variables=['target'], evidence=evidence)

print(result)
