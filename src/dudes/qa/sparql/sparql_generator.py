import copy
import logging
import os
import re
import sys
from collections import defaultdict
from abc import ABC
from typing import Any, Dict, Optional, Tuple, List, Iterable
import lemon

from dudes.qa.sparql.sparqlburger.SPARQLQueryBuilder import SPARQLSelectQuery, SPARQLGraphPattern
from dudes.qa.sparql.sparqlburger.SPARQLSyntaxTerms import Prefix, Filter, GroupBy, Having, Triple, OrderByData, \
    FilterData, OrderBy

from rdflib.namespace import NamespaceManager, Namespace
from z3 import z3

from dudes import consts, utils
from dudes.consts import ask_keywords, count_keywords

from dudes.dudes import DUDES
from dudes.duplex_condition import DuplexCondition, CDUDES, Quantifier
from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint
from dudes.qa.sparql.sparql_modules import SPARQLModule, DuplexSupport, AskSupport, NamespacePrefixes, SPARQLVarMapper, \
    TripleGenerator, SPARQLCountSupport, SPARQLSelectVarChooser, SPARQLFilterSupport, SPARQLOrderBySupport, \
    SPARQLVerifier
from dudes.qa.sparql.sparqlburger.SPARQLQueryBuilder import SPARQLQuery
from dudes.qa.sparql.triple_modules import TopPreparerModule, CompPreparerModule, WithTripleGeneratorModule, \
    BasicTripleGeneratorModule


class SPARQLGenerator:
    endpoint: str
    cache: Optional[Dict[str, Dict[str, Any]]]
    modules: List[SPARQLModule]

    # Alternative: http://client.linkeddatafragments.org/#datasources=http%3A%2F%2Ffragments.dbpedia.org%2F2016-04%2Fen
    def __init__(self,
                 namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
                 nsmanager: Optional[NamespaceManager] = None,
                 endpoint: str = "http://dbpedia.org/sparql",
                 cache: Optional[Dict[str, Dict[str, Any]]] = None,
                 modules: Optional[List[SPARQLModule]] = None, ):
        self.endpoint = endpoint
        self.cache = cache
        if nsmanager is None:
            self.nsmanager = utils.create_namespace_manager(namespaces=namespaces, namespace_manager=nsmanager)
        else:
            self.nsmanager = nsmanager
        self.sparql_endpoint = SPARQLEndpoint(endpoint=endpoint, cache=cache)
        self.modules = modules if modules is not None else []

    @classmethod
    def default(
            cls,
            preset: str = "dbpedia",
            namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
            nsmanager: Optional[NamespaceManager] = None,
            endpoint: Optional[str] = None,
            cache: Optional[Dict[str, Dict[str, Any]]] = None,
            vague_temp_upper_percentage: Optional[float] = None,
            vague_temp_lower_percentage: Optional[float] = None,
            vague_temp_ref_date=None,
            verifier_test_full_triples: Optional[bool] = None,
            verifier_test_no_filter: Optional[bool] = None,
            prefer_wh_words: Optional[bool] = None,
            wh_word_types: Optional[Dict[str, List[str]]] = None,
            generaltype: Optional[str] = None,
            domaintype: Optional[str] = None,
            rangetype: Optional[str] = None,
    ):
        if generaltype is not None:
            consts.generaltype = generaltype

        if preset == "dbpedia" or preset == "qald9":
            if endpoint is None:
                endpoint = "http://dbpedia.org/sparql"
            if verifier_test_full_triples is None:
                verifier_test_full_triples = False
            if verifier_test_no_filter is None:
                verifier_test_no_filter = False
            if prefer_wh_words is None:
                prefer_wh_words = False
            if wh_word_types is None:
                wh_word_types = {
                    "who": ["dbo:Person", "dbo:Organisation"],
                    "what": ["dbo:Place", "dbo:Work", "dbo:Species", "dbo:Organisation", "dbo:Person"],
                    "where": ["dbo:Place"],
                    "when": ["dbo:Event", "dbo:TimePeriod"],
                }
        elif preset == "fuzzy":
            if endpoint is None:
                endpoint = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "vaguetemp", "household_events.ttl")
            if verifier_test_full_triples is None:
                verifier_test_full_triples = True
            if verifier_test_no_filter is None:
                verifier_test_no_filter = True
            if prefer_wh_words is None:
                prefer_wh_words = True
            if wh_word_types is None:
                wh_word_types = {
                    "who": ["dul:Agent"],
                    "what": ["ex:Event"],
                    # "what": ["ex:Bathing", "ex:Personal_Hygiene", "ex:Housekeeping", "ex:Eating", "ex:Leave_Home",
                    #          "ex:Meal_Preparation", "ex:Out_of_Home", "ex:Sleep", "ex:Bed_Toilet_Transition",
                    #          "ex:Watch_TV", "ex:Work"],#["ex:Event"],
                    "when": ["time:ProperInterval"],
                }
        else:
            raise ValueError("Unknown preset: " + preset)

        modules = [
            DuplexSupport(),
            AskSupport(),
            NamespacePrefixes(namespaces=namespaces, nsmanager=nsmanager),
            SPARQLVarMapper(),
            TripleGenerator.default(
                preset=preset,
                vague_temp_upper_percentage=vague_temp_upper_percentage,
                vague_temp_lower_percentage=vague_temp_lower_percentage,
                vague_temp_ref_date=vague_temp_ref_date,
                generaltype=generaltype,
                domaintype=domaintype,
                rangetype=rangetype,
            ),
            SPARQLCountSupport(),
            SPARQLFilterSupport(),
            SPARQLOrderBySupport(),
            SPARQLSelectVarChooser(prefer_wh_words=prefer_wh_words, wh_word_types=wh_word_types),
            SPARQLVerifier(
                endpoint=endpoint,
                cache=cache,
                verifier_test_full_triples=verifier_test_full_triples,
                verifier_test_no_filter=verifier_test_no_filter,
            ),
        ]
        return cls(endpoint=endpoint, cache=cache, modules=modules)


    def to_sparql(
            self,
            #question: str,
            dudes: CDUDES,
            #skip_unrecognized: bool = True,
            #include_redundant: bool = False
            data: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        if data is None:
            data = defaultdict(lambda: None)
            data["next_var_id"] = 0

        if "next_var_id" not in data:
            data["next_var_id"] = 0

        queries: List[SPARQLQuery] = [SPARQLSelectQuery(distinct=True)]
        for module in self.modules:
            dudes, queries, data = module.process(
                #question=question,
                dudes=dudes,
                queries=queries,
                #skip_unrecognized=skip_unrecognized,
                #include_redundant=include_redundant,
                data=data,
            )
        return list({q.get_text() for q in queries})

    def get_results_query(self, query: str):
        return self.sparql_endpoint.get_results_query(query)


class BasicSPARQLGenerator:
    endpoint: str
    cache: Optional[Dict[str, Dict[str, Any]]]

    def __init__(self,
                 namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
                 nsmanager: Optional[NamespaceManager] = None,
                 endpoint: str = "http://dbpedia.org/sparql",
                 cache: Optional[Dict[str, Dict[str, Any]]] = None):
        self.endpoint = endpoint
        self.cache = cache
        if nsmanager is None:
            self.nsmanager = utils.create_namespace_manager(namespaces=namespaces, namespace_manager=nsmanager)
        else:
            self.nsmanager = nsmanager
        self.sparql_endpoint = SPARQLEndpoint(endpoint=endpoint, cache=cache)
        self.next_var_id = 0
        self.unicodere = re.compile(r'\\u\{(.*?)}')

        #self.new_generator = SPARQLGenerator.default(namespaces=namespaces, nsmanager=nsmanager, endpoint=endpoint, cache=cache)

    def _get_new_var(self):
        var = "?v" + str(self.next_var_id)
        self.next_var_id += 1
        return var

    def get_results_query(self, query: str):
        return self.sparql_endpoint.get_results_query(query)

    def to_sparql(
            self,
            # question: str,
            dudes: CDUDES,
            # skip_unrecognized: bool = True,
            # include_redundant: bool = False
            data: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        if data is None:
            data = defaultdict(lambda: None)

        data["next_var_id"] = 0

        if isinstance(dudes, DuplexCondition):
            assert dudes.quantifier == Quantifier.AND
            dudes = DuplexSupport._unify_duplex(dudes)

        self.next_var_id = 0

        clean_query = re.sub(' +', ' ', data["question"].lower().strip())
        is_ask_query = any([clean_query.startswith(ak) for ak in ask_keywords])
        if is_ask_query:
            data["include_redundant"] = True

        sparql_query = SPARQLSelectQuery(distinct=True)  # , limit=4267+1)
        # TODO: derive limit automatically from training data!

        for name, uri in self.nsmanager.namespaces():
            sparql_query.add_prefix(
                prefix=Prefix(prefix=name, namespace=str(uri))
            )

        var_map = dict()
        for v in dudes.all_variables:
            var_map[str(v)] = self._get_new_var()

        # Create a graph pattern to use for the WHERE part and add some triples
        where_pattern = SPARQLGraphPattern()
        triples, orderdata, filterdata = self._to_triples(dudes=dudes,
                                                          var_map=var_map,
                                                          skip_unrecognized=data["skip_unrecognized"] if "skip_unrecognized" in data else True,
                                                          include_redundant=data["include_redundant"] if "include_redundant" in data else False)
        where_pattern.add_triples(
            triples=triples
        )



        existing_exprs = set([str(t.subject) for t in triples]
                             + [str(t.predicate) for t in triples]
                             + [str(t.object) for t in triples])

        variables = [var_map[str(v)] for v in dudes.unassigned_variables]

        predicates = predicates = set(sum([k.split("_") if ":" not in k else [k] for k in dudes.pred_var_dict.keys()], [])) #set(dudes.pred_var_dict.keys())

        found_wh_words = consts.question_words.intersection(predicates)
        if len(found_wh_words) > 1:
            logging.warning("Found more than one question word, adding all variables: " + str(found_wh_words))

        # TODO: reactivate, more sophisticated selection var determination
        # if len(found_wh_words) == 1:
        #    variables = [var_map[str(v)] for vl in dudes.pred_var_dict[list(found_wh_words)[0]] for v in vl]

        select_vars = [v for v in variables if v in existing_exprs]

        # found_count_words = count_keywords.intersection(predicates)
        # if len(found_count_words) > 0:
        # if len(found_wh_words) == 1:
        #     variables = [var_map[str(v)] for vl in dudes.pred_var_dict[list(found_count_words)[0]] for v in vl]
        # else:
        if any([all([w in predicates for w in ck]) for ck in count_keywords]):
            select_vars = ["(COUNT(DISTINCT {}) as {}Count)".format(v, v) for v in select_vars]

        # sparql_query.add_variables(variables=select_vars)

        res_queries: List[str] = []

        orig_query = copy.deepcopy(sparql_query)
        orig_where_pattern = copy.deepcopy(where_pattern)

        if len(select_vars) == 0:
            select_vars = [select_vars]

        if len(orderdata) == 0:
            orderdata = [None]
        for ov in orderdata:
            for sv in select_vars:
                sparql_query = copy.deepcopy(orig_query)
                where_pattern = copy.deepcopy(orig_where_pattern)
                sparql_query.add_variables(sv if isinstance(sv, List) else [sv])

                havings = []

                for fd in filterdata:
                    if not fd.count:
                        where_pattern.add_filter(
                            Filter(expression=fd.filter_str)
                        )
                    else:
                        gb, fexpr = fd.filter_str
                        if len(select_vars) == 1:#TODO: always use sv here?
                            gb = select_vars[0]
                        sparql_query.add_group_by(GroupBy(variables=[gb]))
                        sparql_query.add_having(Having(expression=fexpr))

                if ov is not None:
                    sparql_query.add_order_by(OrderBy(expression=ov.order_str))
                    sparql_query.limit = ov.limit

                # print(sparql_query.get_text())

                # Set this graph pattern to the WHERE part
                sparql_query.set_where_pattern(graph_pattern=where_pattern)

                res_sparql = sparql_query.get_text()
                #res_sparql += " " + " ".join([h.get_text() for h in havings])
                #res_sparql += (" ORDER BY " + ov.order_str if ov is not None else "")
                res_sparql = re.sub(r"SELECT .*?WHERE", "ASK WHERE", res_sparql,
                                    flags=re.DOTALL) if is_ask_query else res_sparql

                res_queries.append(res_sparql)

        # new_res_queries = self.new_generator.to_sparql(dudes, data)
        # cold = set([utils.remove_prefix(q) for q in res_queries])
        # cnew = set([utils.remove_prefix(q) for q in new_res_queries])
        #
        # oldres = set()
        # for q in res_queries:
        #     try:
        #         oldres.update(self.get_results_query(q))
        #     except Exception as e:
        #         logging.error("Error in old query: " + q)
        #         logging.error(e)
        #
        # newres = set()
        # for q in new_res_queries:
        #     try:
        #         newres.update(self.get_results_query(q))
        #     except Exception as e:
        #         logging.error("Error in new query: " + q)
        #         logging.error(e)
        #
        # if len(oldres - newres) > 0:
        #     logging.warning("Different queries: " + str(cold) + " vs. " + str(cnew))
        #     logging.warning("Different results: " + str(oldres) + " vs. " + str(newres))
        return res_queries

    def _fix_unicode(self, val: str) -> str:
        unicodes = set(self.unicodere.findall(val))

        for u in unicodes:
            val = val.replace("\\u{" + u + "}", chr(int(u, 16)))

        return val

    def _to_triples(self,
                    dudes: DUDES,
                    var_map: Dict[str, str],
                    skip_unrecognized: bool = True,
                    include_redundant: bool = False) -> Tuple[List[Triple], List[OrderByData], List[FilterData]]:
        pvd = dudes.pred_var_dict
        # vpd = dudes.var_pred_dict
        model = dudes.get_model()
        triples = []
        orderdata = []
        filterdata = []
        unassigned_variables = dudes.unassigned_variables

        def var_or_value(n: z3.ExprRef) -> str:
            if n in unassigned_variables:
                return var_map[str(n)]
            else:
                val = self._fix_unicode(str(model[n]))

                if ":" in val:
                    nsval = utils.rem_quotes(val)
                    try:
                        if "ns1" in nsval:
                            logging.warning("ns1 in value: " + nsval)
                        return "<{}>".format(self.nsmanager.expand_curie(nsval) if not nsval.startswith("http")
                                             else nsval)
                    except ValueError:
                        return nsval
                else:
                    return val
                # TODO: better heuristic for "not a recognized entity"?

        for pred, vars in pvd.items():
            if ":" not in pred and skip_unrecognized:  # TODO: better heuristic for "not a recognized entity"?
                continue

            vars = [list(x) for x in set(tuple(x) for x in vars)]
            pred = self._fix_unicode(pred)

            for var_order in vars:
                if pred.lower() == "local:top":
                    #assert orderdata is None
                    assert len(var_order) == 3
                    direction: Optional[str] = None
                    match str(var_order[2]):
                        case '"Degree.HIGH"':
                            direction = "DESC"
                        case '"Degree.STRONG"':
                            direction = "DESC"
                        case '"Degree.LOW"':
                            direction = "ASC"
                        case '"Degree.WEAK"':
                            direction = "ASC"
                        case _:
                            raise RuntimeError("Unknown enum element: " + str(var_order[2]))

                    assert direction is not None
                    orderdata.append(OrderByData(
                        var=var_or_value(var_order[1]),
                        limit=int(utils.rem_quotes(str(var_order[0]), all_border_quotes=True)),
                        direction=direction))
                elif pred.lower() == "local:comp":  # TODO: Switch var0 and var1 if var0 is number literal etc.
                    assert len(var_order) == 3
                    operator: Optional[str] = None
                    match str(var_order[2]):
                        case '"Degree.HIGH"':
                            operator = ">"
                        case '"Degree.STRONG"':
                            operator = ">"
                        case '"Degree.LOW"':
                            operator = "<"
                        case '"Degree.WEAK"':
                            operator = "<"
                        case _:
                            raise RuntimeError("Unknown enum element: " + str(var_order[2]))

                    var1 = var_or_value(var_order[0])
                    var2 = var_or_value(var_order[1])

                    if (utils.rem_quotes(var1, all_border_quotes=True).isnumeric()
                            and not utils.rem_quotes(var2, all_border_quotes=True).isnumeric()):
                        tvar = var2
                        var2 = var1
                        var1 = tvar
                        # operator = "<" if operator == ">" else ">"

                    assert operator is not None
                    filterdata.append(FilterData(
                        var=var1,
                        operator=operator,
                        num=utils.rem_quotes(var2, all_border_quotes=True),
                        count=False,
                    ))
                elif pred.lower() == "local:countcomp":
                    assert len(var_order) == 4
                    operator: Optional[str] = None
                    match str(var_order[3]):
                        case '"Degree.HIGH"':
                            operator = ">"
                        case '"Degree.STRONG"':
                            operator = ">"
                        case '"Degree.LOW"':
                            operator = "<"
                        case '"Degree.WEAK"':
                            operator = "<"
                        case _:
                            raise RuntimeError("Unknown enum element: " + str(var_order[3]))

                    var1 = var_or_value(var_order[0])
                    var2 = var_or_value(var_order[1])
                    var3 = var_or_value(var_order[2])

                    if (utils.rem_quotes(var1, all_border_quotes=True).isnumeric()
                            and not utils.rem_quotes(var2, all_border_quotes=True).isnumeric()):
                        tvar = var2
                        var2 = var1
                        var1 = tvar
                        # operator = "<" if operator == ">" else ">"

                    assert operator is not None
                    filterdata.append(FilterData(
                        var=var1,
                        operator=operator,
                        num=utils.rem_quotes(var2, all_border_quotes=True),
                        count=True,
                        bound=var3
                    ))
                elif pred.lower() == "local:with":  # self._get_new_var()
                    triples.append(  # alternative: rdf:type dbo:class
                        Triple(subject=var_or_value(var_order[0]),
                               predicate=self._get_new_var(),
                               object=var_or_value(var_order[1]))
                    )
                elif pred.lower() == "local:rwith":  # self._get_new_var()
                    triples.append(  # alternative: rdf:type dbo:class
                        Triple(subject=var_or_value(var_order[1]),
                               predicate=self._get_new_var(),
                               object=var_or_value(var_order[0]))
                    )
                elif len(var_order) == 1:
                    if var_order[0] in dudes.unassigned_variables or include_redundant:
                        triples.append(  # alternative: rdf:type dbo:class
                            Triple(subject=var_or_value(var_order[0]), predicate="rdf:type", object=pred)
                        )
                elif len(var_order) == 2:
                    unassigned = [v for v in var_order if v in dudes.unassigned_variables]
                    # used = [v for v in var_order if v in dudes.assigned_variables or len(vpd[v]) > 1]
                    if len(unassigned) > 0 or include_redundant:  # and len(used) > 0:
                        triples.append(
                            Triple(subject=var_or_value(var_order[0]),
                                   predicate="<{}>".format(
                                       self.nsmanager.expand_curie(pred) if not pred.startswith("http")
                                       else pred
                                   ),
                                   object=var_or_value(var_order[1]))
                        )
                else:
                    logging.warning("Invalid number of arguments, skipping {}{}".format(pred, str(var_order)))

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
                    or include_redundant):
                res_triples.append(t)

        return res_triples, orderdata, filterdata
