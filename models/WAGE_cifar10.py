from nnUtils_WAGE import *
import WAGE_Initializer

model = Sequential([
    WAGEWeightOnlySpatialConvolution(128,3,3, padding='SAME', bias=False, scale = 2.),
    ReLU(),
    WAGESpatialConvolution(128,3,3, padding='SAME', bias=False, scale = 16.),
    SpatialMaxPooling(2,2,2,2),
    ReLU(),
    WAGESpatialConvolution(256,3,3, padding='SAME', bias=False, scale = 16.),
    ReLU(),
    WAGESpatialConvolution(256,3,3, padding='SAME', bias=False, scale = 16.),
    SpatialMaxPooling(2,2,2,2),
    ReLU(),
    WAGESpatialConvolution(512,3,3, padding='SAME', bias=False, scale = 16.),
    ReLU(),
    WAGESpatialConvolution(512,3,3, padding='SAME', bias=False, scale = 32.),
    SpatialMaxPooling(2,2,2,2),
    ReLU(),
    WAGEAffine(1024, bias=False, scale = 32.),
    ReLU(),
    WAGEAffine(10, bias = False, scale = 16.),
])
