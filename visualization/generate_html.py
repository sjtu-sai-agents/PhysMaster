import sys
from agent.MCTS.mcts_tree import MCTSTree
from agent.MCTS.mcts_tree import MCTSNode

import json
import textwrap
from pathlib import Path
import numpy as np
from igraph import Graph
import logging

logger = logging.getLogger("aide")

def generate_vis(output_path, mct_tree):
    # output_path = cfg.get("output_path")
    # output_path = Path(cfg.get("output_path"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mct_nodes = mct_tree.nodes.values()

    edges = list(get_edges(mct_nodes))
    layout = normalize_layout(generate_layout(len(mct_nodes), edges))

    def dict_to_lines(d, indent=0):
        if isinstance(d, str):
            try:
                d = json.loads(d)
            except:
                return d.replace('\n', '\n')

        space = " " * (indent * 2)

        if isinstance(d, dict):
            lines = []
            for k, v in d.items():
                if isinstance(v, (dict, list)):
                    nested = dict_to_lines(v, indent + 1).rstrip("\n")
                    lines.append(f"{space}{k}:\n{nested}")
                else:
                    lines.append(f"{space}{k}: {v}")
            return "\n\n".join(lines) + "\n"

        elif isinstance(d, list):
            lines = []
            for item in d:
                if isinstance(item, (dict, list)):
                    nested = dict_to_lines(item, indent + 1).rstrip("\n")
                    lines.append(nested)
                else:
                    lines.append(f"{space}- {item}")
            return "\n\n".join(lines) + "\n"

        else:
            return f"{space}{d}\n"

    def ensure_dict(x):
        if isinstance(x, dict):
            return x
        
        if isinstance(x, str):
            return json.loads(x)
    
    res_info =  dict(
        subtask_id = [str(n.subtask_id) for n in mct_nodes],
        node_index = [str(n.node_index) for n in mct_nodes],
        node_type = [str(n.node_type) for n in mct_nodes],
        status = [str(n.status) for n in mct_nodes],
        average_reward = [str(n.average_reward) for n in mct_nodes],

        task_query = [str(n.subtask_description) for n in mct_nodes],
        
        edges = edges,
        layout = layout.tolist(),
        
        code = [str(n.evaluation.get("code", "")) if hasattr(n, 'evaluation') 
                and n.evaluation is not None and isinstance(n.evaluation, dict) 
                else "" for n in mct_nodes],
        
        core_results = [
            dict_to_lines(ensure_dict(n.result).get("core_results", ""))
            if hasattr(n, 'result') and n.result is not None
            else "" 
            for n in mct_nodes
        ],

        analysis = [str(ensure_dict(n.result).get("analysis", "")) if hasattr(n, 'result') 
                and n.result is not None 
                else "" for n in mct_nodes],

        files = [
            dict_to_lines(ensure_dict(n.result).get("files", []))
            if hasattr(n, 'result') and n.result is not None
            else "" 
            for n in mct_nodes
        ],

        decision = [str(n.evaluation.get("decision", "")) if hasattr(n, 'evaluation') 
                and n.evaluation is not None and isinstance(n.evaluation, dict) 
                else "" for n in mct_nodes],

        score = [str(n.evaluation.get("score", "")) if hasattr(n, 'evaluation') 
                and n.evaluation is not None and isinstance(n.evaluation, dict) 
                else "" for n in mct_nodes],

        summary = [str(n.evaluation.get("summary", "")) if hasattr(n, 'evaluation') 
                and n.evaluation is not None and isinstance(n.evaluation, dict) 
                else "" for n in mct_nodes],

        opinion = [str(n.evaluation.get("opinion", "")) if hasattr(n, 'evaluation') 
                and n.evaluation is not None and isinstance(n.evaluation, dict) 
                else "" for n in mct_nodes],
    )

    tree_graph_str = json.dumps(res_info)
    html = generate_html(tree_graph_str)

    with open(output_path, "w") as f:
        f.write(html)
    

def get_edges(nodes):
    for node in nodes:
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                yield (int(node.node_index), int(child.node_index))


def generate_layout(n_nodes, edges, layout_type="rt"):
    """Generate visual layout of graph"""
    logger.info(f"generating layer = Graph, paramaters: n_nodes={n_nodes},edges={edges}")
    layout = Graph(
        n_nodes,
        edges=edges,
        directed=True,
    ).layout(layout_type)
    y_max = max(layout[k][1] for k in range(n_nodes))
    layout_coords = []
    for n in range(n_nodes):
        layout_coords.append((layout[n][0], 2 * y_max - layout[n][1]))
    return np.array(layout_coords)



def normalize_layout(layout: np.ndarray):
    if len(layout) == 0:
        return np.array([])
    
    if len(layout) == 1:
        return np.array([[0.5, 0.5]])
    
    normalized = layout.copy().astype(float)
    
    for dim in [0, 1]:
        min_val = normalized[:, dim].min()
        max_val = normalized[:, dim].max()
        
        if np.isclose(min_val, max_val):
            normalized[:, dim] = 0.5
        else:
            normalized[:, dim] = (normalized[:, dim] - min_val) / (max_val - min_val)
    
    normalized[:, 1] = 1 - normalized[:, 1]
    
    return normalized


def generate_html(tree_graph_str: str):
    template_dir = Path(__file__).parent / "template"
    
    with open(template_dir / "template_mcts.js") as f:
        js = f.read()
        js = js.replace("<placeholder>", tree_graph_str)


    with open(template_dir / "template_mcts.html") as f:
        html = f.read()
        html = html.replace("<!-- placeholder -->", js)

        return html



if __name__ == "__main__":

    cfg = {"enable":True, "output_path":"/data/wkJIN/PHY_Master/HTML/try.html"}
    tree = MCTSTree(
            root_subtask_id=0,
            root_description="Virtual Root",
        )
    virtual_node = list(tree.nodes.values())[0]
    node = MCTSNode(subtask_id='1',node_index='1',node_type='none',subtask_description='no mes')
    virtual_node.children.append(node)
    node.parent = virtual_node
    node.result = """{"core_results": {"t_s": 1.0099504938362078, "v_m_per_s": 9.904544411531507, "check_v_minus_g_t": 0.0}, "analysis": "We model free fall from rest under uniform gravity: y(t) = h - 0.5 g t^2. Impact at y=0 gives t = sqrt(2h/g). Final speed v = sqrt(2 g h); cross-checked by v = g t. Using h=5 m, g=9.81 m/s^2 yields t≈1.00995 s and v≈9.90454 m/s downward. The numerical cross-check v - g t is at machine precision zero.", "code": "import math, os, csv\\n\\n# Parameters\\nh = 5.0           # meters\\ng = 9.81          # m/s^2\\n\\n# Calculations\\nt = math.sqrt(2*h/g)\\nv = math.sqrt(2*g*h)\\nvg = g * t\\n\\ndiff = v - vg\\n\\n# Ensure output directory exists\\noutdir = \\"outputs/test/depth1_node2\\"\\nos.makedirs(outdir, exist_ok=True)\\n\\n# Save results to CSV\\ncsv_path = os.path.join(outdir, \\"result.csv\\")\\nwith open(csv_path, \\"w\\", newline=\\"\\") as f:\\n    writer = csv.writer(f)\\n    writer.writerow([\\"h_m\\", \\"g_m_per_s2\\", \\"t_s\\", \\"v_m_per_s\\", \\"g_t_m_per_s\\", \\"v_minus_gt\\"])\\n    writer.writerow([h, g, t, v, vg, diff])\\n\\nprint({\\n    \\"h\\": h,\\n    \\"g\\": g,\\n    \\"t\\": t,\\n    \\"v\\": v,\\n    \\"g_t\\": vg,\\n    \\"v_minus_gt\\": diff,\\n    \\"csv_path\\": csv_path\\n})", "files": ["outputs/test/depth1_node2/result.csv"], "confidence": 0.49}"""
    
    # result = {}
    # result["core_results"] = "iamamamasmdsaaaaaaaaaaaaaaass"
    # result["analysis"] = "ssssssssssssssssssssssssssssssss"
    # result["files"] = "\dsdds\sadsdsad"
    
    node2 = MCTSNode(subtask_id='2',node_index='2',node_type='none',subtask_description='no mes')
    node2.parent = node
    node.children.append(node2)

    evaluation = {}
    evaluation["code"] = "However, this digital landscape is fraught with pitfalls. The curated nature of online posts often leads to social comparison and envy. Constant exposure to highlight reels of others’ lives can fuel feelings of inadequacy and anxiety, secretly eroding self-esteem. Moreover, the phenomenon of FOMO (Fear Of Missing Out) is intensified. Seeing pictures of a gathering one wasn’t invited to can cause real emotional pain, creating rifts in friend groups. Another serious issue is the rise of cyberbullying, where conflicts can escalate quickly and publicly, causing profound psychological harm that is harder to escape than schoolyard disputes.In conclusion, social media is neither a hero nor a villain in the story of teenage friendship. It is a tool that magnifies and complicates human connection. It can nurture and sustain bonds across distances, yet simultaneously breed insecurity and conflict. The ultimate impact depends largely on how it is used. Teaching digital literacy and encouraging a balanced, mindful approach—where online interaction complements, rather than replaces, genuine face-to-face connection—is key to ensuring that this powerful tool serves to enhance, rather than undermine, the precious friendships of youth."
    evaluation["decision"] = evaluation["code"]
    evaluation["score"] = "0.5555"
    evaluation["summary"] = evaluation["code"]
    evaluation["opinion"] = evaluation["code"]
    node2.result = node.result
    tree.add_node(node)
    tree.add_node(node2)

    node2.evaluation = evaluation
    generate_vis(cfg,tree)


