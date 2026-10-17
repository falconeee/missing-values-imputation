import pandas as pd
import plotly.graph_objects as go


def graph_predict(phi, eixo_x, threshold, freq="1min", list_periods=None):
    df = pd.DataFrame({"phi": phi, "timestamp": eixo_x})

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    df = df.set_index("timestamp")
    df = df.resample(freq).asfreq()
    df = df["phi"].fillna(0)
    df = df.reset_index()

    df["threshold"] = threshold

    layout = go.Layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    
    fig = go.Figure(layout=layout)
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["phi"], mode="lines", name="Índice", fill="tozeroy", line_color="#0F293A")
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["threshold"], mode="lines", name="Limiar", line_color="#FB8102")
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#CECFD1')
    fig.update_layout(hovermode='x unified')
    fig.update_layout(legend=dict(orientation="h"))
    fig.update_layout(yaxis_range=[0, 4 * threshold])

    if list_periods and list_periods != []:
        ### Add a single dummy shape plot for the legend ###
        fig.add_shape(
                showlegend=True,
                type="rect",
                x0=str(list_periods[0]["date_ini"]),
                y0=0,
                x1=str(list_periods[0]["date_ini"]),
                y1=0,
                fillcolor='#68cbf8',
                opacity=1,
                line_width=0,
                layer="below",
                name="Desligado",
            )
        fig.add_shape(
                showlegend=True,
                type="rect",
                x0=str(list_periods[0]["date_ini"]),
                y0=0,
                x1=str(list_periods[0]["date_ini"]),
                y1=0,
                fillcolor='#b3e5fc',
                opacity=1,
                line_width=0,
                layer="below",
                name="Transitório",
            )
        for periodo in list_periods:
            if periodo["type"] == "desligado":
                color = '#68cbf8'
            elif periodo["type"] == "transitorio":
                color = '#b3e5fc'
            else:
                color = "red"

            fig.add_shape(
                type="rect",
                x0=str(periodo["date_ini"]),
                y0=0,
                x1=str(periodo["date_end"]),
                y1=df["phi"].max()*4,
                fillcolor=color,
                opacity=1,
                line_width=0,
                layer="below"
            )

    return fig

