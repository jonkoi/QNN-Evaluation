from nnUtils import *

model = Sequential([
    SpatialConvolution(128,3,3,1,1, padding='VALID'),
    BatchNormalization(),
    ReLU(),
    SpatialConvolution(128,3,3, padding='SAME'),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(),
    ReLU(),
    SpatialConvolution(256,3,3, padding='SAME'),
    BatchNormalization(),
    ReLU(),
    SpatialConvolution(256,3,3, padding='SAME'),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(),
    ReLU(),
    SpatialConvolution(512,3,3, padding='SAME'),
    BatchNormalization(),
    ReLU(),
    SpatialConvolution(512,3,3, padding='SAME'),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(),
    ReLU(),
    Affine(1024, bias=False),
    BatchNormalization(),
    ReLU(),
    Affine(10)
])
