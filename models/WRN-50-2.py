from nnUtils import *

# WRN50-2 bottleNeck for alexnet
## for WRN50-2: K=2.N=(50-4)/6=7
K=10
N=4
model = Sequential([
    SpatialConvolution(64,7,7,padding='SAME'),
    Group(16,3,3,K=K,N=N,padding='SAME'),
    SpatialMaxPooling(3,3,2,2),
    Group(32,3,3,K=K,N=N,padding='SAME'),
    SpatialMaxPooling(3,3,2,2),
    Group(64,3,3,K=K,N=N,padding='SAME'),
    SpatialAveragePooling(7,7,1,1),
    BatchNormalization(),
    ReLU(),
    Affine(10)
])
