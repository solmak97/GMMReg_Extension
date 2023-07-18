import numpy as np


def z_score(values):
    """Data normalization with z-score method

    Args:
        values : data points 

    Returns:
        zdcores : normalized values for datapoints 
    """
    mean = sum(values) / len(values)
    differences = [(value - mean)**2 for value in values]
    sum_of_differences = sum(differences)
    standard_deviation = (sum_of_differences / (len(values) - 1)) ** 0.5
    zscores = [(value - mean) / standard_deviation for value in values]

    return zscores


def aug_class_score_perfect(model, method):
    """Augment the class score vectors based on the method.
      For GMMreg we dont need class score,so it fills with zeros. 
      For GMMreg_ext, we assume that all the class score prediction are 
      perfect, so each probelity of calss for each point is equal one.

    Args:
        model : data points of the model
        method : registration method : GMMreg or GMMreg_ext

    Returns:
        model_aug: datapoint of the model along with the class score vector
        of each point
    """
    n = len(model)
    if method == 'GMMreg':
        print('Registration method is GMMreg')
        class_score = np.zeros(n)
    elif method == 'GMMreg_ext':
        print('Registration method is GMMreg_ext')
        class_score = np.identity(n)
    else:
        print('Error: choose a method either GMMreg or GMMreg_ext)')
    model_aug = np.column_stack((model, class_score))
    return model_aug
