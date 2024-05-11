from dataclasses import dataclass
import math
from typing import Optional
from coloraide import Color
import plotly.graph_objects as go
import networkx as nx


@dataclass(frozen=True)
class NodeInfo:
    name: str
    layer: Optional[int] = None
    module: Optional[str] = None
    pos: Optional[int] = None
    activation: int = 1

    def __hash__(self) -> int:
        return hash((self.name, self.layer, self.module, self.pos))
    
    def __repr__(self) -> str:
        return ".".join(map(str, filter(lambda x: x is not None, [self.pos, self.module, self.name])))
    
    def __eq__(self, other: "NodeInfo") -> bool:
        return self.name == other.name and self.layer == other.layer and self.module == other.module and self.pos == other.pos
    
@dataclass
class PathInfo:
    head: Optional[int] = None

    def __hash__(self) -> int:
        return hash((self.head))

    def __repr__(self) -> str:
        return f"head={self.head}"
    
    def __eq__(self, other: "PathInfo") -> bool:
        return self.head == other.head
    
padding = (NodeInfo("padding"), None)

Meta = tuple[NodeInfo, PathInfo]

def draw_graph(G: nx.MultiDiGraph):
    colors_positive = {
        '50': '#fff4ed',
        '100': '#ffe6d5',
        '200': '#feccaa',
        '300': '#fdac74',
        '400': '#fb8a3c',
        '500': '#f97316',
        '600': '#ea670c',
        '700': '#c2570c',
        '800': '#9a4a12',
        '900': '#7c3d12',
        '950': '#432007',
    }

    colors_negative = {
        '50': '#eff5ff',
        '100': '#dbe8fe',
        '200': '#bfd7fe',
        '300': '#93bbfd',
        '400': '#609afa',
        '500': '#3b82f6',
        '600': '#2570eb',
        '700': '#1d64d8',
        '800': '#1e55af',
        '900': '#1e478a',
        '950': '#172e54',
    }

    def interpolate(colors, weight):
        colors = list(map(lambda x: Color(x), colors.values()))
        interpolation = Color.interpolate(colors, space='oklab', method='continuous')
        return interpolation(weight).convert("srgb").to_string(hex=True)

    edge_traces = []
    for x, y, data in G.edges(data=True):
        if 'pos' not in G.nodes[x]:
            print("No pos for x:", x)
            continue
        x0, y0 = G.nodes[x]['pos']
        if 'pos' not in G.nodes[y]:
            print("No pos for y:", y)
            continue
        x1, y1 = G.nodes[y]['pos']
        sign = 1 if data['weight'] > 0 else -1
        weight = max(0, min(1, (math.log(abs(data['weight'])) + 4) / 10))
        edge_trace = go.Scatter(
            x=[x0, x1], y=[y0, y1],
            line=dict(width=weight * 4 + 0.5, color=interpolate(colors_positive if sign > 0 else colors_negative, weight)),
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)

    node_x = []
    node_y = []
    node_colors = []
    node_labels = []
    for node, data in G.nodes(data=True):
        if 'pos' not in data:
            print("No pos for ", node)
            continue
        x, y = data['pos']
        node_x.append(x)
        node_y.append(y)
        node_colors.append(math.log(data['weight']))
        node_labels.append(str(node))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_labels,
        textposition='bottom center',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale=list(colors_positive.values()),
            color=node_colors,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Ln Node Weights',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )

    fig = go.Figure(
        data=[*edge_traces, node_trace],
        layout=go.Layout(
            title='Network graph made with Python',
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    return fig.to_image(format="png", width=2560, height=1440)