# ----- импорт нужных библиотек для графов -----
import pandas as pd
import pyvis
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.collections import LineCollection
import matplotlib as mpl
from pyvis.network import Network
import colorsys, math, json, os
from jinja2 import Environment, FileSystemLoader
import sys

CLASS_MAP = {
    "вит": "Витамины, витаминоподобные вещества и коферменты",
    "элементы": "Макро- и микроэлементы",
    "пнжк": "Жиры, жироподобные вещества и их производные",
    "стеран": "Жиры, жироподобные вещества и их производные",
    "аминокислоты": "Белки, пептиды, аминокислоты, нуклеиновые кислоты",
    "фенольн": "Фенольные соединения",
    "алкалоиды": "Алкалоиды",
    "пробиотики": "Пробиотические микроорганизмы",
    "полисахариды": "Углеводы и продукты их переработки",
    "сапонины": "Сапонины",
    "терпен": "Терпеноиды",
    "ест": "Естественные метаболиты и стимуляторы метаболизма",
    "гидроксикор": "Гидроксикоричные кислоты",
    "ферменты": "Ферменты",
    "дуб": "Дубильные вещества",
    "цеолиты": "Цеолиты и гуминовые кислоты",
}

def parse_items(cell, sep=",", mapper=None):
    if pd.isna(cell):
        return []

    raw_items = [x.strip() for x in str(cell).split(sep)]

    items = []
    for x in raw_items:
        if not x:
            continue
        if mapper is not None:
            x = mapper(x)
        items.append(x)

    return sorted(set(items))

def count_pairs(df, col, sep=",", mapper=None):
    pair_counts = {}

    for cell in df[col]:
        items = parse_items(cell, sep=sep, mapper=mapper)
        if len(items) < 2:
            continue

        for a, b in combinations(items, 2):
            key = (a, b)
            pair_counts[key] = pair_counts.get(key, 0) + 1

    return pair_counts

def filter_dictionary_by_value(dict, threshold):
    filtered_dict = {
        key: value
        for key, value in dict.items()
        if value >= threshold
    }
    return filtered_dict

def create_interactive_graph(pairs, output_path="interactive_graph.html"):
    if not pairs:
        print("Dictionary is empty")
        return

    min_w = min(pairs.values())
    max_w = max(pairs.values())

    nodes_set = set()
    for (u, v) in pairs.keys():
        nodes_set.add(u)
        nodes_set.add(v)

    num_nodes = len(nodes_set)
    num_edges = len(pairs)

    nodes_sorted = sorted(nodes_set)
    node_colors = {}

    if num_nodes > 0:
        for idx, node in enumerate(nodes_sorted):
            hue = idx / float(num_nodes)
            r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.8)
            color_hex = "#{:02x}{:02x}{:02x}".format(
                int(r * 255),
                int(g * 255),
                int(b * 255),
            )
            node_colors[node] = color_hex

    net = Network(
        height="1000px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        cdn_resources="in_line",
    )

    max_w_for_width = max_w if max_w > 0 else 1.0

    for (u, v), w in pairs.items():

        norm = max(w, 0) / max_w_for_width
        width = 2.0 + 3.0 * math.sqrt(norm)

        if u not in node_colors:
            node_colors[u] = "#8ab4f8"
        if v not in node_colors:
            node_colors[v] = "#8ab4f8"

        net.add_node(u, label=u, color=node_colors[u])
        net.add_node(v, label=v, color=node_colors[v])

        net.add_edge(
            u,
            v,
            title=f"совместно в {w} БАДах",
            width=width,
            value=w,
            color={"inherit": "both"},
        )

    options = {
        "interaction": {
            "hover": True,
            "hoverConnectedEdges": True,
            "selectConnectedEdges": True,
        },
        "nodes": {
            "shape": "dot",
            "scaling": {
                "min": 10,
                "max": 30,
            },
            "font": {
                "size": 16,
            },
        },
        "edges": {
            "smooth": {
                "enabled": True,
                "type": "dynamic",
                "roundness": 0.4,
            },
            "color": {
                "inherit": "both",
            },
        },
        "physics": {
            "enabled": True,
            "barnesHut": {
                "gravitationalConstant": -30000,
                "centralGravity": 0.01,
                "springLength": 350,
                "springConstant": 0.01,
                "damping": 0.09,
                "avoidOverlap": 0.7,
            },
            "stabilization": {
                "iterations": 300,
            },
        },
    }

    net.set_options(json.dumps(options))

    def get_template_path():
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            # Running as PyInstaller bundle
            return os.path.join(sys._MEIPASS, 'templates')
        else:
            # Running in development
            return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

    # Используйте это в create_interactive_graph:
    templates_path = get_template_path()
    print(f"Loading templates from: {templates_path}")  # Для отладки
    env = Environment(loader=FileSystemLoader(templates_path))
    net.template = env.get_template("template.html")

    html = net.generate_html()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    add_weight_form_to_html(
        output_path,
        min_w=min_w,
        max_w=max_w,
        num_nodes=num_nodes,
        num_edges=num_edges,
    )

    print("HTML saved in", output_path)

def add_weight_form_to_html(html_path, min_w, max_w, num_nodes, num_edges):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    placeholder = '<div id="mynetwork" class="card-body"></div>'

    controls_html = f"""
<div id="top-panel"
     style="
        position:fixed;
        top:0; left:0; right:0;
        z-index:9999;
        background:rgba(20,20,20,0.95);
        border-bottom:1px solid #444;
        padding:8px 20px;
        display:flex;
        align-items:center;
        gap:16px;
        font-family:Arial, sans-serif;
        font-size:14px;
        color:#eee;
     ">
  <div style="font-weight:600; white-space:nowrap;">
    Bads class graph
  </div>
  <div style="opacity:0.8; white-space:nowrap;">
    Top: {num_nodes} · Edges: {num_edges} · Max. edge weight: {max_w}
  </div>
  <div style="margin-left:auto; display:flex; align-items:center; gap:8px;">
    <label style="white-space:nowrap;">
      Порог веса:
      <input type="number"
             id="minWeightInput"
             value="{min_w}"
             min="{min_w}"
             max="{max_w}"
             step="1"
             style="
                width:70px;
                margin-left:4px;
                padding:2px 4px;
                background:#111;
                border:1px solid #555;
                color:#eee;
                border-radius:4px;
             ">
    </label>
    <button id="applyWeightBtn"
            style="
                padding:3px 12px;
                border-radius:4px;
                border:1px solid #666;
                background:#2d6cdf;
                color:#fff;
                cursor:pointer;
            ">
      Apply
    </button>
    <span id="minWeightInfo"
          style="margin-left:4px; font-size:12px; opacity:0.85;">
      Threshold: ≥ {min_w}
    </span>
  </div>
</div>
<!-- offset, to graph don't move under panel -->
<div style="height:48px;"></div>
""".strip()

    if placeholder in html:
        html = html.replace(placeholder, controls_html + "\n\n" + placeholder, 1)
    else:
        print("Not found div с id='mynetwork' class='card-body' — Panel don't apear")

    js_block = f"""
<script type="text/javascript">
window.addEventListener("load", function () {{
    if (typeof edges === "undefined") {{
        console.warn("edges DataSet not found");
        return;
    }}

    var allEdges = edges.get();
    var input = document.getElementById("minWeightInput");
    var btn   = document.getElementById("applyWeightBtn");
    var info  = document.getElementById("minWeightInfo");

    if (!input || !btn) {{
        console.warn("weight controls not found");
        return;
    }}

    function applyThreshold() {{
        var v = parseInt(input.value);
        if (isNaN(v)) {{
            v = {min_w};
            input.value = v;
        }}
        if (info) {{
            info.textContent = "Порог: ≥ " + v;
        }}

        var updates = [];
        for (var i = 0; i < allEdges.length; i++) {{
            var e = allEdges[i];
            var hide = e.value < v;
            updates.push({{id: e.id, hidden: hide}});
        }}
        edges.update(updates);
    }}

    btn.addEventListener("click", applyThreshold);
    input.addEventListener("keyup", function(e) {{
        if (e.key === "Enter") {{
            applyThreshold();
        }}
    }});

    applyThreshold();
}});
</script>
"""

    html = html.replace("</body>", js_block + "\n</body>", 1)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

def export_graph_png(
    pairs,
    output_path="bad_graph_colored.png",
    min_weight=1,
    label_min_weight=1,
    cmap_name="viridis",
    curve_scale=0.25,
    label_t_ranges=((0.2, 0.4), (0.6, 0.8)),  # зоны для лейблов
):
    num_segments = 20

    G = nx.Graph()
    for (u, v), w in pairs.items():
        if w < min_weight:
            continue
        G.add_edge(u, v, weight=w)

    if G.number_of_edges() == 0:
        print("There is not edges after threshold, PNG haven't anything for draw")
        return

    pos_nodes = {n: np.array(p) for n, p in nx.circular_layout(G).items()}

    fig, ax = plt.subplots(figsize=(16, 12))

    nodes = list(G.nodes())
    n_nodes = len(nodes)
    node_index = {n: i for i, n in enumerate(nodes)}

    base_cmap = mpl.colormaps.get_cmap(cmap_name)
    colors_for_nodes = base_cmap(np.linspace(0, 1, max(n_nodes, 1)))

    node_color_dict = {
        node: colors_for_nodes[i % colors_for_nodes.shape[0]]
        for i, node in enumerate(nodes)
    }
    node_colors = [node_color_dict[n] for n in nodes]

    weights = [w for (_, _, w) in G.edges(data="weight")]
    max_w = max(weights) if weights else 1.0

    segments = []
    segment_colors = []
    segment_widths = []
    label_infos = []

    center = np.array([0.0, 0.0])

    for (u, v, w) in G.edges(data="weight"):
        p0 = np.array(pos_nodes[u])
        p1 = np.array(pos_nodes[v])

        dir_vec = p1 - p0
        dist = np.linalg.norm(dir_vec)
        if dist == 0:
            continue

        mid = 0.5 * (p0 + p1)

        perp = np.array([-dir_vec[1], dir_vec[0]]) / dist
        c1 = mid + perp * curve_scale * dist
        c2 = mid - perp * curve_scale * dist

        control = c1 if np.linalg.norm(c1 - center) < np.linalg.norm(c2 - center) else c2

        t_vals = np.linspace(0.0, 1.0, num_segments + 1)
        a = ((1 - t_vals) ** 2)[:, None]
        b = (2 * (1 - t_vals) * t_vals)[:, None]
        c = (t_vals ** 2)[:, None]
        points = a * p0 + b * control + c * p1

        cu = np.array(node_color_dict[u])
        cv = np.array(node_color_dict[v])

        edge_width = 1.0 + 4.0 * (w / max_w)

        for i in range(num_segments):
            p_start = points[i]
            p_end = points[i + 1]
            segments.append([p_start, p_end])

            t_mid = (t_vals[i] + t_vals[i + 1]) / 2.0
            c_mid = cu * (1 - t_mid) + cv * t_mid
            segment_colors.append(c_mid)
            segment_widths.append(edge_width)

        iu, iv = node_index[u], node_index[v]
        r_idx = (iu + iv) % len(label_t_ranges)
        t_lo, t_hi = label_t_ranges[r_idx]
        t_label = 0.5 * (t_lo + t_hi)

        aL = (1 - t_label) ** 2
        bL = 2 * (1 - t_label) * t_label
        cL = t_label ** 2
        label_point = aL * p0 + bL * control + cL * p1
        mx, my = label_point

        label_color = cu * (1 - t_label) + cv * t_label
        label_infos.append((mx, my, w, label_color))

    lc = LineCollection(
        segments,
        colors=segment_colors,
        linewidths=segment_widths,
        alpha=0.9,
        capstyle="round",
        joinstyle="round",
    )
    lc.set_zorder(1)
    ax.add_collection(lc)

    node_collection = nx.draw_networkx_nodes(
        G,
        pos_nodes,
        node_size=900,
        node_color=node_colors,
        ax=ax,
    )
    node_collection.set_zorder(2)

    label_dict = nx.draw_networkx_labels(
        G,
        pos_nodes,
        font_size=10,
        ax=ax,
    )
    for text in label_dict.values():
        text.set_clip_on(False)
        text.set_zorder(3)

    for (mx, my, w, label_color) in label_infos:
        if w < label_min_weight:
            continue

        txt = ax.text(
            mx,
            my,
            str(w),
            fontsize=7,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.18",
                fc=label_color,
                ec="none",
                alpha=0.9,
            ),
            color="black",
        )
        txt.set_clip_on(False)
        txt.set_zorder(3)

    ax.set_axis_off()
    ax.set_aspect("equal")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("PNG saved in", output_path)
