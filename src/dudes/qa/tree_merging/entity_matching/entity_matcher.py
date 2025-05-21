import os
import sys
from typing import Optional, List, Iterable, Tuple

import rpyc
from rdflib import Namespace
from rdflib.namespace import NamespaceManager
from treelib import Tree

from dudes import consts
from dudes.qa.tree_merging.entity_matching.data_sources import DataSource, LexiconSource, DBpediaSpotlightSource, \
    TrieTaggerSource, DictSource, RPCTrieTaggerSource
from dudes.qa.tree_merging.entity_matching.trie_tagger import TrieTagger
from lemon.lemon_parser import LEMONParser
from lemon.lexicon import Lexicon


class EntityMatcher:
    def __init__(self, data_sources: List[DataSource]):
        self.data_sources = data_sources

    @classmethod
    def default(cls,
                preset: str = "dbpedia",
                lexicon: Optional[Lexicon] = None,
                rpc_conn: Optional[rpyc.Connection] = None,
                namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
                namespace_manager: Optional[NamespaceManager] = None,
                ):
        ds: List[DataSource] = []

        if lexicon is None:
            if preset == "dbpedia" or preset == "qald9":
                lexicon = LEMONParser.from_ttl_dir(namespaces=namespaces, namespace_manager=namespace_manager).lexicon
            elif preset == "fuzzy":
                lexicon = LEMONParser.from_ttl_dir(
                    ttl_dir=os.path.join(
                        os.path.dirname(sys.modules["lemon"].__file__),
                        "resources",
                        "lexicon_fuzzy"
                    ),
                    csv_dir=None,
                    namespace_manager=namespace_manager
                ).lexicon

        ds.append(LexiconSource(lexicon))

        if preset == "dbpedia" or preset == "qald9":
            ds.append(DBpediaSpotlightSource())
            ds.append(RPCTrieTaggerSource(conn=rpc_conn))
        elif preset == "fuzzy":
            ds.append(DictSource(uri_dict={
                "John": ["http://example.org/John"],
                "Mary": ["http://example.org/Mary"],
                "Tom": ["http://example.org/Tom"],
                "Jerry": ["http://example.org/Jerry"],
                "Alice": ["http://example.org/Alice"],
                "Bob": ["http://example.org/Bob"],
            }))

        return cls(ds)

    @staticmethod
    def all_nodes_recognized(tree: Tree):
        return all([len(node.data.token.candidate_uris) > 0 or len(node.data.lex_candidates) > 0 for node in tree.all_nodes()])

    def match(self, tree: Tree):
        for data_source in self.data_sources:
            changed = data_source.match_tree(tree)
            # CHANGED - apply all sources and generate all possible DUDES!
            # source exec conditions should be defined in a restrictive way such that
            # "not changed" means "all nodes have entries/URIs"
            # if self.all_nodes_recognized(tree):
            #     break
