"""Interactive 3D scatter plot for function 3."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go


# Replace these sample rows with your own (x0, x1, x2, y0) points.
POINTS = [
    (0.171525, 0.343917, 0.248737, -0.112122),
    (0.492581, 0.611593, 0.340176, -0.034835),
    (0.480298, 0.491555, 0.475934, -0.004284),
    (0.622420, 0.406443, 0.433030, -0.006712),
]

DEFAULT_OUT_PATH = Path(__file__).resolve().parent / "images" / "function_3_interactive_from_points.html"


def build_function_3_points_figure(points: list[tuple[float, float, float, float]]) -> go.Figure:
    x0 = [point[0] for point in points]
    x1 = [point[1] for point in points]
    x2 = [point[2] for point in points]
    y0 = [point[3] for point in points]
    customdata = [[idx, value] for idx, value in enumerate(y0, start=1)]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x0,
                y=x1,
                z=x2,
                mode="markers",
                customdata=customdata,
                marker={
                    "size": 7,
                    "color": y0,
                    "colorscale": "geyser",
                    "line": {"color": "#1a1a1a", "width": 1},
                    "colorbar": {"title": "y0"},
                },
                hovertemplate=(
                    "point=%{customdata[0]}<br>"
                    "x0=%{x:.6f}<br>"
                    "x1=%{y:.6f}<br>"
                    "x2=%{z:.6f}<br>"
                    "y0=%{customdata[1]:.6f}<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title="Function 3 Interactive 3D Scatter",
        scene={
            "xaxis_title": "x0",
            "yaxis_title": "x1",
            "zaxis_title": "x2",
        },
        template="plotly_white",
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
    )
    return fig


def write_function_3_points_plot(
    points: list[tuple[float, float, float, float]],
    out_path: Path,
) -> None:
    fig = build_function_3_points_figure(points)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), full_html=True, include_plotlyjs=True)


def main() -> None:
    write_function_3_points_plot(points=POINTS, out_path=DEFAULT_OUT_PATH)
    print(f"Saved: {DEFAULT_OUT_PATH}")


if __name__ == "__main__":
    main()
