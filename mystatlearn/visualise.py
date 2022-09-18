import numpy as np
import plotly.graph_objs as go
import matplotlib as plt

def get_boundaries(model, xlim, ylim, n):
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], n),
                        np.linspace(ylim[0], ylim[1], n))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape)
    return xx, yy, Z

def plotly_widgets(X, y, xs, title=None, scatter_name=None):
    # create plotly figure with widgets
    fig = go.FigureWidget()
    fig.add_trace(go.Scatter(name=scatter_name))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

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