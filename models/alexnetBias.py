from nnUtils import *

model = Sequential([
    SpatialConvolution(64,11,11,4,4, padding='VALID'),
    SpatialMaxPooling(3,3,2,2),
    BatchNormalization(),
    ReLU(),
    SpatialConvolution(192,5,5, padding='SAME'),
    SpatialMaxPooling(3,3,2,2),
    BatchNormalization(),
    ReLU(),
    SpatialConvolution(384,3,3, padding='SAME'),
    BatchNormalization(),
    ReLU(),
    SpatialConvolution(256,3,3, padding='SAME'),
    BatchNormalization(),
    ReLU(),
    SpatialConvolution(256,3,3, padding='SAME'),
    SpatialMaxPooling(3,3,2,2),
    BatchNormalization(),
    ReLU(),
    Affine(4096, bias=False),
    BatchNormalization(),
    ReLU(),
    Dropout(0.5),
    Affine(4096, bias=False),
    BatchNormalization(),
    ReLU(),
    Dropout(0.5),
    Affine(1001) #tensorflow adding 1 category for unused background class
])