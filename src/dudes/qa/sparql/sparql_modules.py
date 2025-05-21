import copy
import inspect
import logging
import math
import os
import pickle
import re
import sys
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from queue import Queue
from typing import Optional, List, Tuple, Iterable, Dict, Any

import numpy as np
import rdflib
from scipy.special import erfinv

from dudes import utils, consts
from dudes.consts import ask_keywords
from dudes.dudes import DUDES
from dudes.duplex_condition import CDUDES, DuplexCondition, Quantifier
from rdflib.namespace import NamespaceManager, Namespace

from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint
from dudes.qa.sparql.sparqlburger.SPARQLQueryBuilder import SPARQLQuery, SPARQLAskQuery, SPARQLGraphPattern, \
    SPARQLSelectQuery
from dudes.qa.sparql.sparqlburger.SPARQLSyntaxTerms import Prefix, Triple, Filter, GroupBy, Having, OrderBy, \
    FilterCombinator, UnionData, FilterData
from dudes.qa.sparql.triple_modules import TripleGeneratorModule, TopPreparerModule, CompPreparerModule, \
    WithTripleGeneratorModule, BasicTripleGeneratorModule, VagueTemporalPreparerModule, \
    PropertyDomainRangeTripleGeneratorModule


class SPARQLModule(ABC):
    @abstractmethod
    def process(
            self,
            dudes: CDUDES,
            queries: List[SPARQLQuery],
            data: Dict[str, Any],
    ) -> Tuple[CDUDES, List[SPARQLQuery], Dict[str, Any]]:
        pass

class DuplexSupport(SPARQLModule):
    @staticmethod
    def _unify_duplex(dudes: Optional[CDUDES]):
        if dudes is None:
            return None
        elif isinstance(dudes, DUDES):
            return dudes
        elif isinstance(dudes, DuplexCondition):
            dudes.refresh_pred_var_dict()
            dudes.distinctify_and_unify_main_vars()
            assert dudes.quantifier == Quantifier.AND
            restr = DuplexSupport._unify_duplex(dudes.restrictor)
            scope = DuplexSupport._unify_duplex(dudes.scope)
            if isinstance(restr, DUDES) and isinstance(scope, DUDES):
                return restr.union(scope)
            elif isinstance(restr, DUDES):
                return restr
            elif isinstance(scope, DUDES):
                return scope
            else:
                return None

    def process(
            self,
            dudes: CDUDES,
            queries: List[SPARQLQuery],
            data: Dict[str, Any],
    ) -> Tuple[CDUDES, List[SPARQLQuery], Dict[str, Any]]:
        if queries is None:
            queries = []

        if isinstance(dudes, DuplexCondition):
            assert dudes.quantifier == Quantifier.AND
            dudes = self._unify_duplex(dudes)
        return dudes, queries, data

class AskSupport(SPARQLModule):
    def process(
            self,
            dudes: CDUDES,
            queries: List[SPARQLQuery],
            data: Dict[str, Any],
    ) -> Tuple[CDUDES, List[SPARQLQuery], Dict[str, Any]]:
        clean_query = re.sub(' +', ' ', data["question"].lower().strip())
        is_ask_query = any([clean_query.startswith(ak) for ak in ask_keywords])
        if is_ask_query:
            data["include_redundant"] = True
            queries = [SPARQLAskQuery() for _ in range(len(queries))]
        return dudes, queries, data

class NamespacePrefixes(SPARQLModule):
    def __init__(
            self,
            namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
            nsmanager: Optional[NamespaceManager] = None
    ):
        if nsmanager is None:
            self.nsmanager = utils.create_namespace_manager(namespaces=namespaces, namespace_manager=nsmanager)
        else:
            self.nsmanager = nsmanager

    def process(
            self,
            dudes: CDUDES,
            queries: List[SPARQLQuery],
            data: Dict[str, Any],
    ) -> Tuple[CDUDES, List[SPARQLQuery], Dict[str, Any]]:
        for sparql_query in queries:
            for name, uri in self.nsmanager.namespaces():
                sparql_query.add_prefix(
                    prefix=Prefix(prefix=name, namespace=str(uri))
                )
        return dudes, queries, data

class SPARQLVarMapper(SPARQLModule):
    @staticmethod
    def _get_new_var(data):
        var = "?v" + str(data["next_var_id"])
        data["next_var_id"] += 1
        return var

    def process(
            self,
            dudes: CDUDES,
            queries: List[SPARQLQuery],
            data: Dict[str, Any],
    ) -> Tuple[CDUDES, List[SPARQLQuery], Dict[str, Any]]:
        var_map = dict()
        for v in dudes.all_variables:
            var_map[str(v)] = self._get_new_var(data)

        data["var_map"] = var_map

        return dudes, queries, data


class TripleGenerator(SPARQLModule):
    def __init__(self, modules: List[TripleGeneratorModule]):
        self.modules = modules

    @classmethod
    def default(cls,
                preset: str = "dbpedia",
                vague_temp_upper_percentage=None,
                vague_temp_lower_percentage=None,
                vague_temp_ref_date=None,
                generaltype: Optional[str] = None,
                domaintype: Optional[str] = None,
                rangetype: Optional[str] = None,
                ):
        if generaltype is not None:
            consts.generaltype = generaltype

        if preset == "fuzzy":
            if vague_temp_upper_percentage is None:
                vague_temp_upper_percentage = 1.0
            if vague_temp_lower_percentage is None:
                vague_temp_lower_percentage = 0.6
            if vague_temp_ref_date is None:
                vague_temp_ref_date = datetime.now()
            if generaltype is None:
                generaltype = "rdf:type/(rdfs:subClassOf|owl:equivalentClass)*"
            if domaintype is None:
                domaintype = "rdf:type/(rdfs:subClassOf|owl:equivalentClass)*"
            if rangetype is None:
                rangetype = "rdf:type/(rdfs:subClassOf|owl:equivalentClass)*"


            return cls(modules=[
                TopPreparerModule(),
                CompPreparerModule(),
                WithTripleGeneratorModule(),
                VagueTemporalPreparerModule(
                    vague_temp_upper_percentage=vague_temp_upper_percentage,
                    vague_temp_lower_percentage=vague_temp_lower_percentage,
                    vague_temp_ref_date=vague_temp_ref_date
                ),
                PropertyDomainRangeTripleGeneratorModule(generaltype=generaltype, domaintype=domaintype, rangetype=rangetype),
                BasicTripleGeneratorModule(),
            ])
        else:
            return cls(modules=[
                TopPreparerModule(),
                CompPreparerModule(),
                WithTripleGeneratorModule(),
                PropertyDomainRangeTripleGeneratorModule(generaltype=generaltype, domaintype=domaintype, rangetype=rangetype),
                BasicTripleGeneratorModule(),
            ])

    @staticmethod
    def _skip_unrecognized(pred: str, data: Dict[str, Any]) -> bool:
        return ":" not in pred and ("skip_unrecognized" not in data or data["skip_unrecognized"])

    def process(
            self,
            dudes: CDUDES,
            queries: List[SPARQLQuery],
            data: Dict[str, Any],
    ) -> Tuple[CDUDES, List[SPARQLQuery], Dict[str, Any]]:
        pvd = dudes.pred_var_dict
        triples: List[Triple] = []
        for pred, vars in pvd.items():
            if self._skip_unrecognized(pred, data): #":" not in pred and ("skip_unrecognized" not in data or data["skip_unrecognized"]):  # TODO: better heuristic for "not a recognized entity"?
                continue

            vars = [list(x) for x in set(tuple(x) for x in vars)]
            pred = utils.fix_unicode(pred)

            for var_order in vars:
                for module in self.modules:
                    triples, updated, data = module.process(triples, pred, vars, var_order, dudes, data)
                    if updated:
                        break

        vpd = defaultdict(set)
        for t in triples:
            if "?" in t.subject:
                vpd[t.subject].add(t.predicate)
            if "?" in t.object:
                vpd[t.object].add(t.predicate)

        res_triples = []

        for t in triples:
            # Either values/entities (i.e., no ?var) or at least one variable is further restricted by another triple
            # or redundant triples are included
            if ("?" not in t.subject
                    or "?" not in t.object
                    or len(vpd[t.subject]) > 1
                    or len(vpd[t.object]) > 1
                    or ("include_redundant" in data and data["include_redundant"])):
                res_triples.append(t)

        new_queries = []
        for query in queries:
            if query.where is None:
                query.where = SPARQLGraphPattern()
            query.where.add_triples(
                triples=copy.deepcopy(res_triples)
            )
            new_queries.append(query)
        return dudes, new_queries, data

class SPARQLCountSupport(SPARQLModule):

    def process(
            self,
            dudes: CDUDES,
            queries: List[SPARQLQuery],
            data: Dict[str, Any],
    ) -> Tuple[CDUDES, List[SPARQLQuery], Dict[str, Any]]:
        predicates = set(sum([k.split("_") if ":" not in k else [k] for k in dudes.pred_var_dict.keys()], []))
        if any([all([w in predicates for w in ck]) for ck in consts.count_keywords]):
            for q in queries:
                if isinstance(q, SPARQLSelectQuery):
                    q.count = True

        return dudes, queries, data

class SPARQLSelectVarChooser(SPARQLModule):
    def __init__(self,
                 prefer_wh_words: bool = False,
                 wh_word_types: Optional[Dict[str, List[str]]] = None,
                 ):
        self.prefer_wh_words = prefer_wh_words
        self.wh_word_types = wh_word_types

    def process(
            self,
            dudes: CDUDES,
            queries: List[SPARQLQuery],
            data: Dict[str, Any],
    ) -> Tuple[CDUDES, List[SPARQLQuery], Dict[str, Any]]:
        existing_exprs = set()
        # existing_exprs = set([str(t.subject) for query in queries if isinstance(query.where, SPARQLGraphPattern) for t in query.where.graph]
        #                      + [str(t.predicate) for query in queries if isinstance(query.where, SPARQLGraphPattern) for t in query.where.graph]
        #                      + [str(t.object) for query in queries if isinstance(query.where, SPARQLGraphPattern) for t in query.where.graph])

        queue = Queue()
        for query in queries:
            queue.put(query.where.graph)
        while not queue.empty():
            graph = queue.get()
            for t in graph:
                if isinstance(t, Triple):
                    existing_exprs.add(str(t.subject))
                    existing_exprs.add(str(t.predicate))
                    existing_exprs.add(str(t.object))
                elif isinstance(t, SPARQLGraphPattern):
                    queue.put(t.graph)
                else:
                    logging.warning(f"Unknown graph element: {t} {type(t)}")


        wh_word = None
        variables = [data["var_map"][str(v)] for v in dudes.unassigned_variables]

        if self.prefer_wh_words:
            predicates = set(dudes.pred_var_dict.keys())

            found_wh_words = consts.question_words.intersection(predicates)
            if len(found_wh_words) > 1:
                logging.warning("Found more than one question word, fallback to regular behavior: " + str(found_wh_words))
            else:
                if len(found_wh_words) == 1:
                    wh_word = list(found_wh_words)[0]
                    variables = [data["var_map"][str(v)] for vl in dudes.pred_var_dict[wh_word] for v in vl]

        select_vars = [v for v in variables if v in existing_exprs]

        # sparql_query.add_variables(variables=select_vars)

        res_queries: List[SPARQLQuery] = []

        # if len(select_vars) == 0:
        #     select_vars = [select_vars]

        for q in queries:
            if isinstance(q, SPARQLSelectQuery):
                if self.prefer_wh_words and self.wh_word_types is not None and wh_word in self.wh_word_types:
                    for wtype in self.wh_word_types[wh_word]:
                        for sv in select_vars:
                            if isinstance(sv, List):
                                assert not isinstance(wtype, List)
                            q_new = copy.deepcopy(q)
                            q_new.add_variables([sv])
                            q_new.where.add_triples([Triple(subject=sv, predicate=consts.generaltype, object=wtype)])
                            res_queries.append(q_new)

                for sv in select_vars:
                    q_new = copy.deepcopy(q)
                    q_new.add_variables(sv if isinstance(sv, List) else [sv])
                    res_queries.append(q_new)
            else:
                res_queries.append(q)

        return dudes, res_queries, data

class SPARQLFilterSupport(SPARQLModule):
    def process(
            self,
            dudes: CDUDES,
            queries: List[SPARQLQuery],
            data: Dict[str, Any],
    ) -> Tuple[CDUDES, List[SPARQLQuery], Dict[str, Any]]:
        if "filterdata" not in data:
            logging.warning("No filter data provided, did you call the corresponding preparer module in the triple generator before?")
            return dudes, queries, data

        res_queries: List[SPARQLQuery] = []

        for q_old in queries:
            for fdl in utils.powerset(data["filterdata"]):
                q = copy.deepcopy(q_old)
                updated = False
                for fd in fdl:
                    if isinstance(fd, FilterCombinator) or (isinstance(fd, FilterData) and not fd.count):
                        if isinstance(q.where, SPARQLGraphPattern):
                            q.where.add_filter(
                                Filter(expression=fd.filter_str)
                            )
                            updated = True
                    elif isinstance(fd, UnionData):
                        fd.apply(q)
                        updated = True
                    elif isinstance(q, SPARQLSelectQuery) and isinstance(fd, FilterData):
                            gb, fexpr = fd.filter_str
                            if len(q.variables) == 1:  # TODO: always use sv here?
                                gb = q.variables[0]
                            q.add_group_by(GroupBy(variables=[gb]))
                            q.add_having(Having(expression=fexpr))
                            updated = True
                    else:
                        logging.warning(f"Filter data not recognized: {fd} {type(fd)}")

                if updated:
                    res_queries.append(q)
            res_queries.append(q_old)

        return dudes, res_queries, data

class SPARQLOrderBySupport(SPARQLModule):
    def process(
            self,
            dudes: CDUDES,
            queries: List[SPARQLQuery],
            data: Dict[str, Any],
    ) -> Tuple[CDUDES, List[SPARQLQuery], Dict[str, Any]]:
        if "orderdata" not in data:
            logging.warning("No filter data provided, did you call the corresponding preparer module in the triple generator before?")
            return dudes, queries, data

        res_queries: List[SPARQLQuery] = []

        for q_old in queries:
            for ov in data["orderdata"]:
                if ov is not None:
                    q = copy.deepcopy(q_old)
                    q.add_order_by(OrderBy(expression=ov.order_str))
                    q.limit = ov.limit
                    res_queries.append(q)
            res_queries.append(q_old)

        return dudes, res_queries, data


class SPARQLVerifier(SPARQLModule):
    def __init__(self,
                 endpoint: Optional[str] = None,
                 cache: Optional[Dict[str, Dict[str, Any]]] = None,
                 verifier_test_full_triples: Optional[bool] = None,
                 verifier_test_no_filter: Optional[bool] = None,
                 ):
        self.endpoint = endpoint
        self.cache = cache
        self.sparql_endpoint = SPARQLEndpoint(endpoint=endpoint, cache=self.cache)
        self.verifier_test_full_triples = verifier_test_full_triples
        self.verifier_test_no_filter = verifier_test_no_filter


    def process(
            self,
            dudes: CDUDES,
            queries: List[SPARQLQuery],
            data: Dict[str, Any],
    ) -> Tuple[CDUDES, List[SPARQLQuery], Dict[str, Any]]:
        checked_queries = []

        for query in queries:
            try:  # check for syntax errors
                qresult = rdflib.Graph().query(query.get_text())
                if self.verifier_test_full_triples:
                    for t in query.where.graph:
                        if not isinstance(t, Triple) or t.subject.startswith("?") or t.object.startswith("?") or t.predicate.startswith("?"):
                            continue
                        if not self.sparql_endpoint.ask_triple(t):
                            raise ValueError(f"Triple not found: {t}")
                    if self.verifier_test_no_filter:
                        if query.where.filters:
                            query_copy = copy.deepcopy(query)
                            query_copy.where.filters = []
                            tres = self.sparql_endpoint.get_results_query(query_copy.get_text())
                            if len(tres) == 0:
                                raise ValueError(f"No results without filters: {query_copy.get_text()}")

                checked_queries.append(query)
            except Exception as e:
                #print(traceback.format_exc())
                logging.error(f"Query is not valid, skipping: {query.get_text()} {e}")
                continue



        return dudes, checked_queries, data
