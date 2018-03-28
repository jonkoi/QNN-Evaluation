from nnUtils import *

# WRN28-10 for cifar10
## for WRN28-10: K=10.N=(28-4)/6=4
K=10
N=4
model = Sequential([
    SpatialConvolution(16,3,3,padding='SAME'),
    BatchNormalization(),
    ReLU(),
    Group(16,3,3,K=K,N=N,padding='SAME',fixPaddingFilters=16*K-16),
    Group(32,3,3,2,2,K=K,N=N,padding='SAME',fixPaddingFilters=16*K), #32-16
    Group(64,3,3,2,2,K=K,N=N,padding='SAME',fixPaddingFilters=32*K), #64-32
    SpatialAveragePooling(8,8,1,1),
    Affine(10)
])
