from datetime import datetime, timedelta
import inspect
import logging
import math
import os
import pickle
import re
import sys
from abc import ABC, abstractmethod
from typing import Optional, Iterable, Tuple, Dict, Any, List

import numpy as np
import z3
from rdflib import Namespace
from rdflib.namespace import NamespaceManager
from scipy.special import erfinv

from dudes import utils, consts
from dudes.dudes import DUDES
from dudes.qa.sparql.sparqlburger.SPARQLQueryBuilder import SPARQLGraphPattern
from dudes.qa.sparql.sparqlburger.SPARQLSyntaxTerms import Triple, OrderByData, FilterData, FilterCombinator, UnionData, \
    Filter

import json
import joblib
from scipy.special import erfinv
from sentence_transformers import SentenceTransformer

class TripleGeneratorModule(ABC):
    def __init__(
            self,
            namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
            nsmanager: Optional[NamespaceManager] = None,
    ):
        if nsmanager is None:
            self.nsmanager = utils.create_namespace_manager(namespaces=namespaces, namespace_manager=nsmanager)
        else:
            self.nsmanager = nsmanager

    def var_or_value(self, n: z3.ExprRef, dudes: DUDES, data: Dict[str, Any]) -> str:
        if n in dudes.unassigned_variables:
            return data["var_map"][str(n)]
        else:
            val = utils.fix_unicode(str(dudes.get_model()[n]))

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

    @abstractmethod
    def process(
            self,
            triples: List[Triple],
            pred: str,
            vars: List[List[z3.ExprRef]],
            var_order: List[z3.ExprRef],
            dudes: DUDES,
            data: Dict[str, Any],
    ) -> Tuple[List[Triple], bool, Dict[str, Any]]:
        pass


class TopPreparerModule(TripleGeneratorModule):
    def process(
            self,
            triples: List[Triple],
            pred: str,
            vars: List[List[z3.ExprRef]],
            var_order: List[z3.ExprRef],
            dudes: DUDES,
            data: Dict[str, Any],
    ) -> Tuple[List[Triple], bool, Dict[str, Any]]:
        updated = False
        if "orderdata" not in data:
            data["orderdata"] = []

        if pred.lower() == "local:top":
            # assert orderdata is None
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
            data["orderdata"].append(OrderByData(
                var=self.var_or_value(var_order[1], dudes, data),
                limit=int(utils.rem_quotes(str(var_order[0]), all_border_quotes=True)),
                direction=direction))
            updated = True

        return triples, updated, data

class CompPreparerModule(TripleGeneratorModule):
    def process(
            self,
            triples: List[Triple],
            pred: str,
            vars: List[List[z3.ExprRef]],
            var_order: List[z3.ExprRef],
            dudes: DUDES,
            data: Dict[str, Any],
    ) -> Tuple[List[Triple], bool, Dict[str, Any]]:
        updated = False
        if "filterdata" not in data:
            data["filterdata"] = []

        if pred.lower() == "local:comp":  # TODO: Switch var0 and var1 if var0 is number literal etc.
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

            var1 = self.var_or_value(var_order[0], dudes, data)
            var2 = self.var_or_value(var_order[1], dudes, data)

            if (utils.rem_quotes(var1, all_border_quotes=True).isnumeric()
                    and not utils.rem_quotes(var2, all_border_quotes=True).isnumeric()):
                tvar = var2
                var2 = var1
                var1 = tvar
                # operator = "<" if operator == ">" else ">"

            assert operator is not None

            data["filterdata"].append(FilterData(
                var=var1,
                operator=operator,
                num=utils.rem_quotes(var2, all_border_quotes=True),
                count=False,
            ))
            updated = True
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

            var1 = self.var_or_value(var_order[0], dudes, data)
            var2 = self.var_or_value(var_order[1], dudes, data)
            var3 = self.var_or_value(var_order[2], dudes, data)

            if (utils.rem_quotes(var1, all_border_quotes=True).isnumeric()
                    and not utils.rem_quotes(var2, all_border_quotes=True).isnumeric()):
                tvar = var2
                var2 = var1
                var1 = tvar
                # operator = "<" if operator == ">" else ">"

            assert operator is not None
            if "filterdata" not in data:
                data["filterdata"] = []
            data["filterdata"].append(FilterData(
                var=var1,
                operator=operator,
                num=utils.rem_quotes(var2, all_border_quotes=True),
                count=True,
                bound=var3
            ))
            updated = True

        return triples, updated, data

class RefDateWrapper:
    def __init__(self, ref_date: datetime):
        self.ref_date = ref_date


class VagueTemporalPreparerModule(TripleGeneratorModule):

    all_adverbials = [
        "recently",
        "just",
        "some time ago",
        "long time ago"
    ]

    def __init__(self,
                 namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
                 nsmanager: Optional[NamespaceManager] = None,
                 entity_prefix: Optional[str] = None,
                 vague_temp_percentage=0.6,
                 vague_temp_ref_date=None):
        super().__init__(namespaces=namespaces, nsmanager=nsmanager)
        self.vague_temp_percentage = vague_temp_percentage
        if vague_temp_ref_date is None:
            vague_temp_ref_date = RefDateWrapper(datetime.now())
        elif isinstance(vague_temp_ref_date, datetime):
            vague_temp_ref_date = RefDateWrapper(vague_temp_ref_date)
        elif not isinstance(vague_temp_ref_date, RefDateWrapper):
            raise ValueError("Invalid ref_date type")
        self.vague_temp_ref_date = vague_temp_ref_date

        self._embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device='cpu')
        with open(os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "fuzzylli_new", "adverbial_params.json"), "r", encoding="utf-8") as fh:
            self._adverbial_params = json.load(fh)
        with open(os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "fuzzylli_new", "kgqa_event_types.json"), "r", encoding="utf-8") as fh:
            self._kgqa_event_types = tuple(json.load(fh))
        self._embedding_regressor = joblib.load(os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "fuzzylli_new", "configuration_word_embeddings.pkl"))
        encoder = self._embedding_model
        records = self._kgqa_event_types
        vectors = []

        for record in records:
            embeddings = encoder.encode(self._candidate_texts(record), normalize_embeddings=True)
            vector = np.mean(np.asarray(embeddings), axis=0)
            norm = np.linalg.norm(vector)
            if norm:
                vector = vector / norm
            vectors.append(vector)

        self._kgqa_event_type_index = (records, np.vstack(vectors))

    @staticmethod
    def _safe_round(value):
        value = float(value)
        return int(round(value)) if math.isfinite(value) else value

    @staticmethod
    def _gauss_inverse(y, mean, std):
        val = std * np.sqrt(-2 * np.log(y))
        return mean - val, mean + val

    @staticmethod
    def _inverse_event_specific_function(y, std):
        clipped_y = max(-1, min(1, 2 * y - 1))
        return math.sqrt(2) * std * erfinv(clipped_y)

    @staticmethod
    def _readable_event_type(event_type):
        name = event_type.split(":", 1)[-1]
        return name.replace("_", " ").strip()

    def _candidate_texts(self, record):
        texts = [record.get("label") or self._readable_event_type(record["event_type"])]
        texts.extend(record.get("examples", []))
        texts.append(self._readable_event_type(record["event_type"]))
        return [text for text in texts if text]

    def _query_text(self, event):
        return self._readable_event_type(event) if ":" in event else event.replace("_", " ")

    def resolve_event_type_embedding(self, event):
        records, matrix = self._kgqa_event_type_index
        encoder = self._embedding_model
        query = encoder.encode([self._query_text(event)], normalize_embeddings=True)[0]
        scores = matrix @ np.asarray(query)

        best = int(np.argmax(scores))
        best_record = records[best]

        return {
            "clean_event_type": best_record["event_type"],
            "label": best_record.get("label") or self._readable_event_type(best_record["event_type"]),
            "score": float(scores[best]),
        }

    def _predict_event_std_embedding(self, event, use_clean_event_type=True):
        resolved = self.resolve_event_type_embedding(event) if use_clean_event_type else None
        embedding_text = resolved["label"] if resolved else event

        encoder = self._embedding_model
        ridge = self._embedding_regressor
        vector = encoder.encode(embedding_text)
        log_pred = ridge.predict(np.asarray(vector).reshape(1, -1))[0]
        event_std = float(max(0.0, np.expm1(log_pred)))

        return event_std

    def predict_time_frame_embedding(self,
                                     event,
                                     adverbial,
                                     min_prob=0.6,
                                     use_clean_event_type=True):
        params = self._adverbial_params
        adverbial_mean = params["adverbial_means"][adverbial]
        adverbial_std = params["adverbial_stds"][adverbial]
        event_std = self._predict_event_std_embedding(
            event,
            use_clean_event_type=use_clean_event_type,
        )

        lower_adv, higher_adv = self._gauss_inverse(min_prob, adverbial_mean, adverbial_std)
        upper_raw = self._inverse_event_specific_function(higher_adv, event_std)
        lower_raw = self._inverse_event_specific_function(lower_adv, event_std)

        upper = max(0, self._safe_round(upper_raw))
        lower = max(0, self._safe_round(lower_raw))
        return upper, lower

    def predict_kgqa_interval_embedding(self,
                                        event,
                                        adverbial,
                                        min_prob=0.6):
        resolved = self.resolve_event_type_embedding(event)
        event_std = self._predict_event_std_embedding(
            resolved["label"],
            use_clean_event_type=False,
        )
        upper, lower = self.predict_time_frame_embedding(
            resolved["label"],
            adverbial,
            min_prob=min_prob,
            use_clean_event_type=False,
        )

        return {
            "input_event": event,
            "clean_event_type": resolved["clean_event_type"],
            "event_type_label": resolved["label"],
            "event_type_score": resolved["score"],
            "adverbial": adverbial,
            "min_prob": min_prob,
            "event_std": event_std,
            "interval": {
                "upper_minutes_ago": upper,
                "lower_minutes_ago": lower,
            },
        }

    def event_type(self, event):
        event = event.lower().strip().replace("_", " ")
        return self.resolve_event_type_embedding(
            event,
        )["clean_event_type"]

    def get_minutes_ago(self, adverbial, event):
        adverbial = adverbial.lower().strip().replace("_", " ")
        event = event.lower().strip().replace("_", " ")

        if adverbial not in self.all_adverbials:
            raise ValueError(f"Invalid adverbial: {adverbial}")

        prediction = self.predict_kgqa_interval_embedding(
            event,
            adverbial,
            min_prob=self.vague_temp_percentage,
        )
        interval = prediction["interval"]

        res = [
            (
                interval["upper_minutes_ago"],
                interval["lower_minutes_ago"],
                prediction["clean_event_type"],
            )
        ]
        # print("Interval results: ", res, flush=True)
        return res

    def process(
            self,
            triples: List[Triple],
            pred: str,
            vars,
            var_order,
            dudes: DUDES,
            data: Dict[str, Any],
    ) -> Tuple[List[Triple], bool, Dict[str, Any]]:
        updated = False
        if "filterdata" not in data:
            data["filterdata"] = []
        #data["include_redundant"] = True

        if pred.lower() == "local:vaguetemp":
            assert len(var_order) == 3

            adverb = utils.rem_quotes(str(var_order[0]))
            event = utils.rem_quotes(str(var_order[1]))
            var = self.var_or_value(var_order[2], dudes, data)

            mins_ago = self.get_minutes_ago(adverb, event)
            assert len(mins_ago) > 0

            triples.extend([
                Triple(subject=var, predicate="ex:happensAt", object=var + "Interval"),
            ])
            patterns = []
            for upper, lower, event_type in mins_ago:
                assert min(upper, lower) < float('inf')
                upper_date = self.vague_temp_ref_date.ref_date - timedelta(minutes=min(upper, lower))
                pattern = SPARQLGraphPattern()
                fconj = FilterCombinator(combinator=" && ")
                fconj.filters.append(FilterData(
                    var=var + "End",
                    operator="<=",
                    num='"' + upper_date.isoformat() + '"',
                    count=False,
                    datetime=True
                ))

                if max(upper, lower) < float('inf'):
                    lower_date = self.vague_temp_ref_date.ref_date - timedelta(minutes=max(upper, lower))
                    fconj.filters.append(FilterData(
                        var=var + "End",
                        operator=">=",
                        num='"' + lower_date.isoformat() + '"',
                        count=False,
                        datetime=True
                    ))

                pattern.add_triples([
                    Triple(subject=var + "Interval", predicate="time:hasEnd", object=var + "End"),
                ])
                pattern.add_filter(Filter(expression=fconj.filter_str))
                if event_type is not None:
                    pattern.add_triples([
                        Triple(subject=var, predicate="rdf:type", object=event_type),
                    ])

                updated = True
                patterns.append(pattern)
            data["filterdata"].append(UnionData(patterns))

        return triples, updated, data

class WithTripleGeneratorModule(TripleGeneratorModule):
    @staticmethod
    def _get_new_var(data):
        var = "?v" + str(data["next_var_id"])
        data["next_var_id"] += 1
        return var

    def process(
            self,
            triples: List[Triple],
            pred: str,
            vars: List[List[z3.ExprRef]],
            var_order: List[z3.ExprRef],
            dudes: DUDES,
            data: Dict[str, Any],
    ) -> Tuple[List[Triple], bool, Dict[str, Any]]:
        updated = False
        if pred.lower() == "local:with":  # self._get_new_var()
            triples.append(  # alternative: rdf:type dbo:class
                Triple(subject=self.var_or_value(var_order[0], dudes, data),
                       predicate=self._get_new_var(data),
                       object=self.var_or_value(var_order[1], dudes, data))
            )
            updated = True
        elif pred.lower() == "local:rwith":  # self._get_new_var()
            triples.append(  # alternative: rdf:type dbo:class
                Triple(subject=self.var_or_value(var_order[1], dudes, data),
                       predicate=self._get_new_var(data),
                       object=self.var_or_value(var_order[0], dudes, data))
            )
            updated = True

        return triples, updated, data

class PropertyDomainRangeTripleGeneratorModule(TripleGeneratorModule):
    def __init__(self,
                 generaltype: Optional[str] = None,
                 domaintype: Optional[str] = None,
                 rangetype: Optional[str] = None,
                 namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
                 nsmanager: Optional[NamespaceManager] = None
                 ):
        if generaltype is not None:
            consts.generaltype = generaltype
        super().__init__(namespaces=namespaces, nsmanager=nsmanager)
        self.generaltype = generaltype
        self.domaintype = domaintype
        self.rangetype = rangetype

        if self.generaltype is None:
            self.generaltype = "rdf:type"
        if self.domaintype is None:
            self.domaintype = "rdf:type"
        if self.rangetype is None:
            self.rangetype = "rdf:type"

    @staticmethod
    def _get_new_var(data):
        var = "?v" + str(data["next_var_id"])
        data["next_var_id"] += 1
        return var

    def process(
            self,
            triples: List[Triple],
            pred: str,
            vars: List[List[z3.ExprRef]],
            var_order: List[z3.ExprRef],
            dudes: DUDES,
            data: Dict[str, Any],
    ) -> Tuple[List[Triple], bool, Dict[str, Any]]:
        updated = False
        if pred.lower() == "local:generaltype":  # self._get_new_var()
            triples.append(  # alternative: rdf:type dbo:class
                Triple(subject=self.var_or_value(var_order[0], dudes, data),
                       predicate=self.generaltype,
                       object=self.var_or_value(var_order[1], dudes, data))
            )
            updated = True
        elif pred.lower() == "local:propertydomain":  # self._get_new_var()
            triples.append(  # alternative: rdf:type dbo:class
                Triple(subject=self.var_or_value(var_order[0], dudes, data),
                       predicate=self.domaintype,
                       object=self.var_or_value(var_order[1], dudes, data))
            )
            updated = True
        elif pred.lower() == "local:propertyrange":  # self._get_new_var()
            triples.append(  # alternative: rdf:type dbo:class
                Triple(subject=self.var_or_value(var_order[0], dudes, data),
                       predicate=self.rangetype,
                       object=self.var_or_value(var_order[1], dudes, data))
            )
            updated = True

        return triples, updated, data

class BasicTripleGeneratorModule(TripleGeneratorModule):
    def process(
            self,
            triples: List[Triple],
            pred: str,
            vars: List[List[z3.ExprRef]],
            var_order: List[z3.ExprRef],
            dudes: DUDES,
            data: Dict[str, Any],
    ) -> Tuple[List[Triple], bool, Dict[str, Any]]:
        updated = False
        if len(var_order) == 1:
            if var_order[0] in dudes.unassigned_variables or ("include_redundant" in data and data["include_redundant"]):
                triples.append(  # alternative: rdf:type dbo:class
                    Triple(subject=self.var_or_value(var_order[0], dudes, data), predicate="rdf:type", object=pred)
                )
                updated = True
        elif len(var_order) == 2:
            unassigned = [v for v in var_order if v in dudes.unassigned_variables]
            # used = [v for v in var_order if v in dudes.assigned_variables or len(vpd[v]) > 1]
            if len(unassigned) > 0 or ("include_redundant" in data and data["include_redundant"]):  # and len(used) > 0:
                triples.append(
                    Triple(subject=self.var_or_value(var_order[0], dudes, data),
                           predicate="<{}>".format(
                               self.nsmanager.expand_curie(pred) if not pred.startswith("http")
                               else pred
                           ),
                           object=self.var_or_value(var_order[1], dudes, data))
                )
                updated = True
        else:
            logging.warning("Invalid number of arguments, skipping {}{}".format(pred, str(var_order)))

        return triples, updated, data

