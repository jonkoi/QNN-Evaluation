from nnUtils import *

model = Sequential([
    BinarizedWeightOnlySpatialConvolution(128,3,3, padding='SAME', bias=False, stochastic=True),
    BatchNormalization(),
    ReLU(),
    BinarizedWeightOnlySpatialConvolution(128,3,3, padding='SAME', bias=False, stochastic=True),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(),
    ReLU(),
    BinarizedWeightOnlySpatialConvolution(256,3,3, padding='SAME', bias=False, stochastic=True),
    BatchNormalization(),
    ReLU(),
    BinarizedWeightOnlySpatialConvolution(256,3,3, padding='SAME', bias=False, stochastic=True),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(),
    ReLU(),
    BinarizedWeightOnlySpatialConvolution(512,3,3, padding='SAME', bias=False, stochastic=True),
    BatchNormalization(),
    ReLU(),
    BinarizedWeightOnlySpatialConvolution(512,3,3, padding='SAME', bias=False, stochastic=True),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(),
    ReLU(),
    BinarizedWeightOnlyAffine(1024, bias=False, stochastic=True),
    BatchNormalization(),
    ReLU(),
    BinarizedWeightOnlyAffine(10, stochastic=True),
    # BatchNormalization()
])
