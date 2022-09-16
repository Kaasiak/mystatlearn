import numpy as np

def get_boundaries(model, xlim, ylim, n):
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], n),
                        np.linspace(ylim[0], ylim[1], n))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape)
    return xx, yy, Z