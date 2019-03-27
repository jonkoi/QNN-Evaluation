from nnUtils_BWN_layer import *

model = Sequential([
    BinaryNWSpatialConvolution(128,3,3, padding='SAME', bias=False),
    BatchNormalization(),
    ReLU(),
    BinaryNWSpatialConvolution(128,3,3, padding='SAME', bias=False),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(),
    ReLU(),
    BinaryNWSpatialConvolution(256,3,3, padding='SAME', bias=False),
    BatchNormalization(),
    ReLU(),
    BinaryNWSpatialConvolution(256,3,3, padding='SAME', bias=False),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(),
    ReLU(),
    BinaryNWSpatialConvolution(512,3,3, padding='SAME', bias=False),
    BatchNormalization(),
    ReLU(),
    BinaryNWSpatialConvolution(512,3,3, padding='SAME', bias=False),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(),
    ReLU(),
    BinarizedOnlyWeightAffine(1024, bias=False),
    BatchNormalization(),
    ReLU(),
    BinarizedOnlyWeightAffine(10),
    BatchNormalization(),
])
