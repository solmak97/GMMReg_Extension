import numpy as np

# Non-rigid transforms


def ConvertParamToRotationAndTranslation(param, d):
    translation = np.zeros((d, 1))
    translation[0][0] = param[0]
    translation[1][0] = param[1]
    theta = param[2]
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    return rotation, translation


def ConvertRigidParamToMatrix(param, d):
    rotation, translation = ConvertParamToRotationAndTranslation(param, d)
    matrix = np.zeros((d + 1, d + 1))
    matrix[:d, :d] = rotation
    matrix[:d, d] = translation.flatten()
    matrix[d, d] = 1
    return matrix


def transform_model(model, param, d):
    n = len(model)
    r, t = ConvertParamToRotationAndTranslation(param, d)
    transform_model = np.dot(model, r) + np.ones([n, 1])*t.transpose()
    return transform_model


def transformed_model(model, par, d):
    transform = transform_model(model, par, d)
    return transform
