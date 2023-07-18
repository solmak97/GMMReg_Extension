import transforms
import numpy as np
from scipy.optimize import minimize


def gauss_transform(A, B, wm, ws, scale, scale_class, missing_points, method):
    """The inner product between two spherical Gaussian mixtures computed using
    the Gauss Transform. The centers of the two mixtures are given in terms of
    two point sets A and B (of same dimension d) represented.
    It is assumed that all the components have the same covariance matrix 
    represented by a scale parameter (scale). 

    Args:
        A : model data points
        B : scene data points
        wm : model class scores
        ws : scene class scores
        scale : sigma 
        scale_class : sigma_c
        missing_points : number of missing points in the scene , if any
        method : choose ( 'GMMreg', 'GMMreg_ext')

    Returns:
        cost: cross_term / (m * n)
        adj : the matrix of class score weights distance
    """
    m, dim = A.shape
    n = B.shape[0]
    adj = np.zeros((m, n))
    cross_term = 0
    if method == 'GMMreg':
        # in 'GMMreg' we want to ignore the class scores term.The w_ij is always 0.
        scale_class = 1e10

    for i in range(m):
        for j in range(n-missing_points):
            dist_ij = np.linalg.norm(A[i] - B[j])**2
            w_ij = np.linalg.norm(wm[i] - ws[j])**2
            cost_ij = np.exp(-dist_ij / (4*scale**2)) * \
                np.exp(-w_ij/(4*scale_class**2))
            adj[i][j] = np.exp(-w_ij/(4*scale_class**2))
            cross_term += cost_ij
    return cross_term / (m * n), adj


def L2_distance(model, scene, class_m, class_s, scale, scale_class, param, d,
                missing_points, method):
    # model update based on the parameters
    model = transforms.transform_model(model, param, d)
    # inner product of the model and the scene as L2 distance
    # for non-rigid transform
    fg, adj = gauss_transform(model, scene, class_m, class_s, scale,
                              scale_class, missing_points, method)
    # formula from the (jian 2011 paper, ignoring f*f term due to non-rigid transform)
    f = - 2*fg
    return f, adj


def objective_function(model, scene, class_m, class_s, scale,
                       scale_class, param, d, missing_points, method):
    # Code to compute the L2 distance between the Gaussian mixtures
    # constructed from the transformed model and the scene with a scale
    cost, adj = L2_distance(
        model, scene, class_m, class_s, scale, scale_class, param,
        d, missing_points, method)
    # print(cost)
    return cost


def optimization_algorithm(model, scene, class_m, class_s, scale, scale_class,
                           param, d, missing_points, method):
    """optimization_algorithm

    Args:
        model : model data points (source)
        scene : scene data points (target)
        class_m : class score vectors
        class_s : class score vectors
        scale : initial scale value (sigma)
        scale_class : class score class (sigma_c)
        param : initial parameters
        d : data points dimentions
        missing_points : Number of missing point between the model and scene, if any
        method : 'GMMreg' or ' GMMreg_ext'

    Returns:
        param : estimated parameters
        scale : final scale value
        i : number of iterations
    """
    prev_param = None  # Variable to store previous parameter values
    i = 1
    while scale > 0:
        """
        Simulated annealing optimization with temperature rate of 0.97
        """
        # Set up the objective function
        def objective(x): return objective_function(model, scene,
                                                    class_m,
                                                    class_s,
                                                    scale,
                                                    scale_class, x,
                                                    d, missing_points,
                                                    method)

        # Optimize the objective function
        result = minimize(objective, param, method='SLSQP')

        # get the estimated parameter
        param = result.x

        # Check stopping criterion
        if prev_param is not None and np.allclose(param, prev_param):
            break

        # Update the previous parameter values
        prev_param = param.copy()

        # Decrease the scale ( simulated annealing)
        scale *= 0.9
        # Update iteration number
        i = i+1
    return param, scale, i
