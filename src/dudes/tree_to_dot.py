from queue import Queue
from typing import Optional

from treelib import Tree, Node  # type: ignore
import graphviz as gv  # type: ignore

from dudes.dudes import SelectionPair
from lemon.lemon import Reference


def node_to_label(node: Node):
    text = max(node.data.token.text_, key=len)
    deppos = node.data.token.dep_ + " " + node.data.token.pos_
    add_text = ""
    if len(node.data.dudes_candidates) > 0:
        add_text = "\\n" + str(node.data.dudes_candidates[0]).replace("\n", "\\n")
    elif len(node.data.lex_candidates) > 0 and len(node.data.lex_candidates[0].sense) > 0 and len(node.data.lex_candidates[0].sense[0].reference) > 0:
        if isinstance(node.data.lex_candidates[0].sense[0].reference[0], Reference):
            add_text = "\\n" + str(node.data.lex_candidates[0].sense[0].reference[0].uri)
        else:
            add_text = "\\n" + str(node.data.lex_candidates[0].sense[0].reference[0])
    elif len(node.data.token.ent_kb_id_+node.data.token.tagger_kb_ids) > 0:
        add_text = "\\n" + str((node.data.token.ent_kb_id_+node.data.token.tagger_kb_ids)[0])

    return gv.nohtml(f"{text}\\n{deppos}{add_text}".replace('"', "&#x22;").replace("'", "&#x27;"))


def tree_to_dot(tree: Tree,
                graph: Optional[gv.Digraph] = None,
                node_name_postfix: str = "",
                merged: Optional[Node] = None,
                merged_into: Optional[Node] = None,
                selection_pair: Optional[SelectionPair] = None):
    def node_name(nid):
        return "v" + str(nid) + node_name_postfix

    g = gv.Digraph()
    if graph is not None:
        g = graph
    q = Queue()
    root_node: Node = tree.get_node(tree.root)
    g.node(name=node_name(tree.root),
           label=node_to_label(root_node),
           shape="box",
           color="red" if root_node in [merged, merged_into] else "black")
    q.put(tree.root)

    while not q.empty():
        nid = q.get()
        for child in tree.children(nid):
            g.node(name=node_name(child.identifier),
                   label=node_to_label(child),
                   shape="box",
                   color="red" if child in [merged, merged_into] else "black")
            g.edge(node_name(nid), node_name(child.identifier))
            q.put(child.identifier)

    if merged is not None and merged_into is not None:
        g.edge(node_name(merged.identifier),
               node_name(merged_into.identifier),
               label="Merge" + (":\\n" + str(selection_pair) if selection_pair is not None else ""),
               color="red")

    # DotFile.runXDot(str(g))
    return g
