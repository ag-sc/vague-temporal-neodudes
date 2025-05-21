import copy
import itertools
import logging
import os
import statistics
import sys
import traceback
from collections import defaultdict
from typing import Optional, Tuple, Iterable, Dict, Any, List, Generator, Set

import rpyc
import spacy
import xxhash
from rdflib import Graph
from rdflib.namespace import NamespaceManager, Namespace

from dudes import consts, utils
from dudes.dudes import DUDES
from dudes.dudes_tree import Tree
from dudes.qa.dudes_creation.dudes_creator import DUDESCreator
from dudes.qa.dudes_composition.dudes_composer import DUDESComposer
from dudes.qa.preprocessing.question_preprocessor import QuestionPreprocessor
from dudes.qa.sparql.sparql_generator import SPARQLGenerator, BasicSPARQLGenerator
from dudes.qa.dudes_composition.tree_compose_strategy import TreeComposeStrategy, BottomUpTreeComposeStrategy, \
    YieldTreeComposeStrategy
from dudes.qa.tree_merging.tree_merger import TreeMerger
from lemon.lemon_parser import LEMONParser
from lemon.lexicon import Lexicon
from lemon.namespaces import default_namespaces


class QAPipeline:
    def __init__(self,
                 question_preprocessor: QuestionPreprocessor,
                 tree_merger: TreeMerger,
                 dudes_creator: DUDESCreator,
                 compose_strategy: TreeComposeStrategy,
                 sparql_generator: SPARQLGenerator):
        self.question_preprocessor = question_preprocessor
        self.tree_merger = tree_merger
        self.dudes_creator = dudes_creator
        self.compose_strategy = compose_strategy
        self.sparql_generator = sparql_generator

    @classmethod
    def default(cls,
                preset: str = "dbpedia",
                tree_merger_candidate_limit: int = 3,
                dudes_composer_candidate_limit: int = 10,
                lexicon: Optional[Lexicon] = None,
                entity_predicates_path: Optional[str] = None,
                rpc_conn: Optional[rpyc.Connection] = None,
                sparql_endpoint: Optional[str] = None,
                dbpedia_spotlight_endpoint: Optional[str] = None,
                nlp: Optional[spacy.language.Language] = None,
                namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
                cache: Optional[Dict[str, Dict[str, Any]]] = None,
                namespace_manager: Optional[NamespaceManager] = None,
                vague_temp_upper_percentage=None,
                vague_temp_lower_percentage=None,
                vague_temp_ref_date=None,
                verifier_test_full_triples: Optional[bool] = None,
                verifier_test_no_filter: Optional[bool] = None,
                prefer_wh_words: Optional[bool] = None,
                wh_word_types: Optional[Dict[str, List[str]]] = None,
                generaltype: Optional[str] = None,
                domaintype: Optional[str] = None,
                rangetype: Optional[str] = None,
               ):
        namespace_manager = utils.create_namespace_manager(namespaces=namespaces,
                                                           namespace_manager=namespace_manager)

        if generaltype is not None:
            consts.generaltype = generaltype

        if preset == "dbpedia" or preset == "qald9":
            if entity_predicates_path is None:
                entity_predicates_path = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "entity_predicates.sqlite")
            if sparql_endpoint is None:
                sparql_endpoint = consts.sparql_endpoint
            if dbpedia_spotlight_endpoint is None:
                dbpedia_spotlight_endpoint = consts.dbpedia_spotlight_endpoint
            if rpc_conn is None:
                rpc_conn = rpyc.connect(consts.rpc_host,
                                        consts.rpc_port,
                                        config={
                                            "allow_public_attrs": True,
                                            "allow_pickle": True,
                                            "sync_request_timeout": 300
                                        })
        elif preset == "fuzzy":
            if sparql_endpoint is None:
                sparql_endpoint = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "vaguetemp", "household_events.ttl")
            if entity_predicates_path is None:
                entity_predicates_path = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "vaguetemp", "entity_predicates_fuzzy.sqlite")
        else:
            raise ValueError(f"Unknown preset: {preset}")

        return QAPipeline(
            question_preprocessor=QuestionPreprocessor.default(
                preset=preset,
                dbpedia_spotlight_endpoint=dbpedia_spotlight_endpoint,
                nlp=nlp
            ),
            tree_merger=TreeMerger.default(
                preset=preset,
                candidate_limit=tree_merger_candidate_limit,
                lexicon=lexicon,
                rpc_conn=rpc_conn,
                namespace_manager=namespace_manager
            ),
            dudes_creator=DUDESCreator.default(preset=preset, namespace_manager=namespace_manager),
            compose_strategy=YieldTreeComposeStrategy(
                dudes_composer=DUDESComposer.default(preset=preset, candidate_limit=dudes_composer_candidate_limit),
                namespace_manager=namespace_manager,
                entity_predicates_path=entity_predicates_path,
            ),
            sparql_generator=SPARQLGenerator.default(
                preset=preset,
                nsmanager=namespace_manager,
                endpoint=sparql_endpoint,
                cache=cache,
                vague_temp_upper_percentage=vague_temp_upper_percentage,
                vague_temp_lower_percentage=vague_temp_lower_percentage,
                vague_temp_ref_date=vague_temp_ref_date,
                verifier_test_full_triples=verifier_test_full_triples,
                verifier_test_no_filter=verifier_test_no_filter,
                prefer_wh_words=prefer_wh_words,
                wh_word_types=wh_word_types,
                generaltype=generaltype,
                domaintype=domaintype,
                rangetype=rangetype,
            )
            # BasicSPARQLGenerator(nsmanager=namespace_manager, endpoint=sparql_endpoint, cache=cache)
        )

    def create_dudes_generator_from_tree(self, tree_id, tree) -> Generator[Tuple[int, Tree, int, DUDES], None, None]:
        tree = copy.deepcopy(tree) # should be unnecessary
        tree = self.dudes_creator.assign_atomic_dudes(tree)
        dudes_candidates = self.compose_strategy.compose_tree(tree)
        for idx, dc in enumerate(dudes_candidates):
            newtree = copy.deepcopy(tree) # should be unnecessary
            yield tree_id, newtree, idx, dc

    def dudes_generator(self, question: str) -> itertools.chain[Tuple[int, Tree, int, DUDES]]:
        start_trees = self.question_preprocessor.preprocess(question)
        trees = list(self.tree_merger.merge(start_trees))
        dudes_generators = [self.create_dudes_generator_from_tree(tree_id, scored_tree[1]) for tree_id, scored_tree in enumerate(trees)]
        #generator_scores = [int(statistics.mean(scored_tree[0])*10)+1 for scored_tree in trees]
        #generator_scores = [int(scored_tree[0]*10)+1 for scored_tree in trees]

        if len(dudes_generators) > self.tree_merger.candidate_limit:
            return itertools.chain(
                utils.roundrobin(
                    *dudes_generators[:self.tree_merger.candidate_limit],
                ),
                utils.roundrobin(
                    *dudes_generators[self.tree_merger.candidate_limit:],
                )
            )
        else:
            return itertools.chain(utils.roundrobin(*dudes_generators))

        # if len(dudes_generators) > self.tree_merger.candidate_limit:
        #     return itertools.chain(
        #         utils.weighted_roundrobin(
        #             dudes_generators[:self.tree_merger.candidate_limit],
        #             generator_scores[:self.tree_merger.candidate_limit]
        #         ),
        #         utils.weighted_roundrobin(
        #             dudes_generators[self.tree_merger.candidate_limit:],
        #             generator_scores[self.tree_merger.candidate_limit:]
        #         )
        #     )
        # else:
        #     return itertools.chain(utils.weighted_roundrobin(dudes_generators, generator_scores))

    def process_qald(self, question: str, skip_unrecognized: bool = True) -> Generator[Tuple[str, DUDES, str], None, None]:
        query_set: Set[int] = set()
        xh = xxhash.xxh32()
        for tree_id, tree, idx, dc in self.dudes_generator(question):
            if idx % 100 == 0:
                logging.info(f"Constructing query {idx + 1}, tree {tree_id + 1} of question '{question}'")
            try:
                sparql_queries = self.sparql_generator.to_sparql(dudes=copy.deepcopy(dc),
                                                                 data=dict(
                                                                     question=question,
                                                                     skip_unrecognized=skip_unrecognized,
                                                                     include_redundant=False
                                                                 ))
                sparql_queries_full = self.sparql_generator.to_sparql(dudes=copy.deepcopy(dc),
                                                                      data=dict(
                                                                          question=question,
                                                                          skip_unrecognized=skip_unrecognized,
                                                                          include_redundant=True
                                                                      ))

                for query, dds, query_full in zip(sparql_queries,
                                                  (dc for _ in range(len(sparql_queries))),
                                                  sparql_queries_full):
                    if query not in query_set:
                        query_set.add(utils.hash_str(query, xh))
                        yield query, dds, query_full
                    else:
                        logging.info(f"Skipping duplicate query")
                # res.extend(sparql_queries)
                # res_full.extend(sparql_queries_full)
            except Exception as e:
                logging.warning(f"Failed to generate SPARQL query for DUDES: {dc} ({e})")

        # if len(res) > 0 and len(res_dudes) > 0 and len(res_full) > 0:
        #     res_dict: Dict = {x: (x, y, z) for x, y, z in zip(res, res_dudes, res_full)}
        #
        #     res, res_dudes, res_full = zip(*res_dict.values())
        #
        # return res, res_dudes, res_full

    def process_query_and_dudes(self, question: str, skip_unrecognized: bool = True, include_redundant: bool = False) -> Generator[Tuple[str, str], None, None]:
        res: Set[str] = set()
        for tree_id, tree, idx, dc in self.dudes_generator(question):
            if idx % 100 == 0:
                logging.info(f"Constructing query {idx + 1}, tree {tree_id + 1} of question '{question}'")
            try:
                sparql_queries = self.sparql_generator.to_sparql(dudes=dc,
                                                                 data=dict(
                                                                     question=question,
                                                                     skip_unrecognized=skip_unrecognized,
                                                                     include_redundant=include_redundant
                                                                 ))
                for query in sparql_queries:
                    if query not in res:
                        res.add(query)
                        yield query, str(dc)
                    else:
                        logging.info(f"Skipping duplicate query")
            except Exception as e:
                logging.warning(f"Failed to generate SPARQL query for DUDES: {dc} ({e})")
                #print(traceback.format_exc())

    def process(self, question: str, skip_unrecognized: bool = True, include_redundant: bool = False) -> Generator[str, None, None]:
        for q, d in self.process_query_and_dudes(question, skip_unrecognized, include_redundant):
            yield q
        #return list(set(res))

    def process_and_fetch(self,
                          question: str,
                          skip_unrecognized: bool = True,
                          include_redundant: bool = False
                          ) -> Generator[Tuple[str, Any], None, None]:
        queries = self.process(question, skip_unrecognized, include_redundant)

        for idx, sparql_query in enumerate(queries):
            if idx % 10 == 0:
                logging.info(f"Fetching query {idx+1} of question '{question}'")
            try:
                yield sparql_query, self.sparql_generator.get_results_query(sparql_query)
            except Exception as e:
                logging.warning(f"Failed to fetch results for SPARQL query: {sparql_query} ({e})")

