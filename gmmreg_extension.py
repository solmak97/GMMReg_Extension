import numpy as np
import pre_processing
import transforms
import L2_objective
import time
import math
# Load data
data_model = np.loadtxt('fish.csv')  # Model
# Scene : is a transformed version of the model
data_scene = np.loadtxt('fish.csv')
d = 2  # data points are in 2D
# In case of having mising points in the scene
missing_points = len(data_model)-len(data_scene)

# Pre-processing : Normilization
data_model = pre_processing.z_score(data_model)
data_scene = pre_processing.z_score(data_scene)

# Ground Thruth : TG
theta_g = -2  # rotation
t_x_g = 0  # translation in x direction
t_y_g = 0  # translation in y direction
param_g = [t_x_g, t_y_g, theta_g]

# Method selection : GMMreg and GMMreg_ext

# method = 'GMMreg_ext'
method = 'GMMreg'
# Augment the datapoints with class score vectors.
m = pre_processing.aug_class_score_perfect(data_model, method)
model = m[:, :2]
s = pre_processing.aug_class_score_perfect(
    transforms.transformed_model(data_scene, param_g, d), method)
scene = s[:, :2]
# Class score matrixes
class_model = m[:, 2:len(s)+2]  # slice : in case of missing data in the scene
class_scene = s[:, 2:]
# initial parameters T
param = [0, 0, 0]
# Hyperarameters
scale_class = 0.2
scale = 2
# Point set registration
# The regsitration is an optimization poblem to minimize L2 distance betweeen
# model point sets and the the scene.The algorithm estimates the trasnform
# parameters ( rotation, translations in x and y direction) while mimizing the L2 distance.

t = time.time()
parameters, final_scale, iteration = L2_objective.optimization_algorithm(
    model, scene, class_model, class_scene, scale, scale_class, param, d, missing_points, method)
elapsed = time.time()

# Error : The distance between estimated T and the ground truth TG
e = math.sqrt((parameters[0]-param_g[0])**2 + (parameters[1] -
              param_g[1])**2 + (parameters[2]-param_g[2]) ** 2)

### Diplay results###

print('estimated parameters', parameters, '\n', 'Ground Truth', param_g, '\n',
      'Error', e, '\n', 'final sigma', final_scale, '\n', 'elapsed time', elapsed-t, '\n', 'iterations', iteration)
