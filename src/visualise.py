import numpy as np
import plotly.graph_objs as go
import matplotlib as mpl
import matplotlib_inline
import matplotlib.pyplot as plt

def setup_notebook():
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
    rcParams = {
        'font.family' : 'Avenir',
        'font.size': 12,
        'axes.linewidth' : 1.5
    }
    mpl.rcParams.update(rcParams)

def get_boundaries(model, xlim, ylim, n):
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], n),
                        np.linspace(ylim[0], ylim[1], n))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape)
    return xx, yy, Z

def plotly_widgets(X, y, xs, title=None, scatter_name=None):
    fig = go.FigureWidget()
    fig.add_trace(go.Scatter(name=scatter_name))
    colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

    fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='data'))
    fig.data[1].marker.color = colors[0]
    fig.data[0].line.color = colors[1]
    fig.update_xaxes(
        range=[xs.min() - 0.1, xs.max() + 0.1], gridcolor='lightgrey', zeroline=False)
    fig.update_yaxes(
        range=[y.min() - 2, y.max() + 2], gridcolor='lightgrey', zeroline=False)
    fig.update_layout(
        paper_bgcolor='white', 
        plot_bgcolor='#f5f5f5', 
        title=title, 
        title_x=0.5, font_family='Avenir'
    )
    return fig

def plot_svm(X, labels, w, b):
    xx = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1)
    a = -w[0]/w[1]
    yy = a*xx - (b)/w[1]
    margin = 1 / np.sqrt(np.sum(w**2))
    yy_neg = yy - np.sqrt(1 + a**2) * margin
    yy_pos = yy + np.sqrt(1 + a**2) * margin
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xx, yy, "k-")
    ax.plot(xx, yy_neg, "m--")
    ax.plot(xx, yy_pos, "m--")
    ax.scatter(X[:, 0], X[:, 1], c=labels, alpha=0.5, edgecolors="black")
    return ax