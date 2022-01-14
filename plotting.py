from os.path import join
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def matplotlibify(fig, size=24, width_inches=3.5, height_inches=3.5, dpi=142):
    # make it look more like matplotlib
    # modified from: https://medium.com/swlh/formatting-a-plotly-figure-with-matplotlib-style-fa56ddd97539)
    font_dict = dict(family="Arial", size=size, color="black")

    # app = QApplication(sys.argv)
    # screen = app.screens()[0]
    # dpi = screen.physicalDotsPerInch()
    # app.quit()

    fig.update_layout(
        font=font_dict,
        plot_bgcolor="white",
        width=width_inches * dpi,
        height=height_inches * dpi,
        margin=dict(r=40, t=20, b=10),
    )

    fig.update_yaxes(
        showline=True,  # add line at x=0
        linecolor="black",  # line color
        linewidth=2.4,  # line size
        ticks="inside",  # ticks outside axis
        tickfont=font_dict,  # tick label font
        mirror="allticks",  # add ticks to top/right axes
        tickwidth=2.4,  # tick width
        tickcolor="black",  # tick color
    )

    fig.update_xaxes(
        showline=True,
        showticklabels=True,
        linecolor="black",
        linewidth=2.4,
        ticks="inside",
        tickfont=font_dict,
        mirror="allticks",
        tickwidth=2.4,
        tickcolor="black",
    )
    fig.update(layout_coloraxis_showscale=False)

    width_default_px = fig.layout.width
    targ_dpi = 300
    scale = width_inches / (width_default_px / dpi) * (targ_dpi / dpi)

    return fig, scale


def parity_with_err(
    result_df,
    x="actual_hardness",
    y="predicted_hardness",
    error_y="y_std_calib",
    size="load",
    hover_data=["composition"],
    figfolder="figures",
    fname="parity_err_calib",
    auto_open=True,
    **pxscatter_kwargs,
):
    """Scatter plot with error bars.

    Parameters
    ----------
    result_df : DataFrame
        Must contain at minimum `actual_hardness`, `predicted_hardness`, `y_std_calib`,
        `load`, and `composition`, unless custom values are set for `x`, `y`, `error_y`,
        and `hover_data`, respectively. If any custom kwargs are given to `px.scatter`,
        those must be present as well.
    x : str, optional
        x-axis name used to access column from `result_df`, by default "actual_hardness"
    y : str, optional
        y-axis name used to access column from `result_df`, by default "predicted_hardness"
    error_y : str, optional
        error in y values name used to access column from `result_df`, by default "y_std_calib"
    size : str, optional
        name of variable from `result_df` that will control size of markers, by default "load"
    hover_data : list, optional
        list of names of additional data to include when point is hovered on by mouse, by default ["composition"]
    figfolder : str, optional
        [description], by default "figures"
    fname : str, optional
        [description], by default "parity_err"

    Returns
    -------
    [type]
        [description]
    """
    figpath = join(figfolder, fname)
    Path(figfolder).mkdir(exist_ok=True, parents=True)
    fig = px.scatter(
        result_df,
        x=x,
        y=y,
        error_y=error_y,
        size=size,
        hover_data=hover_data,
        **pxscatter_kwargs,
    )

    fig.update_traces(marker=dict(color="red"))

    # parity line
    mx = np.nanmax([result_df["predicted_hardness"], result_df["actual_hardness"]])
    mx2 = mx  # max of both
    # mx, mx2 = np.nanmax(proxy), np.nanmax(target) # max of each
    fig.add_trace(
        go.Line(x=[0, mx], y=[0, mx2], name="parity", line=dict(color="black"))
    )

    fig.write_html(figpath + ".html", auto_open=auto_open)

    fig2, scale = matplotlibify(fig, size=48, dpi=300)

    fig2.update_layout(
        legend=dict(
            font=dict(size=32),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0)",
        )
    )

    fig2.write_image(figpath + ".png")

    return fig, fig2

