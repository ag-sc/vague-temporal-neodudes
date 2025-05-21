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

    events_durations = [
        "minutes",
        "hours"
    ]
    all_adverbials = [
        "recently",
        "just",
        "some time ago",
        "long time ago"
    ]

    all_events = [
        "bath",
        "clean house",
        "care hygiene",
        "out home",
        "leave home",
        # "return home",
        "meal prepar",
        "sleep",
        "toilet",
        "watch tv",
        "work",
        "eat",
    ]


     # a ex:Bathing ;
     # a ex:Bed_Toilet_Transition ;
     # a ex:Eating ;
     # a ex:Housekeeping ;
     # a ex:Leave_Home ;
     # a ex:Meal_Preparation ;
     # a ex:Out_of_Home ;
     # a ex:Personal_Hygiene ;
     # a ex:Sleep ;
     # a ex:Watch_TV ;
     # a ex:Work ;

    # event_types = {
    #     "brushing_teeth": "?undef",
    #     "brushing teeth": "?undef",
    #     "birthday": "?undef",
    #     "vacation": "?undef",
    #     "sabbatical": "?undef",
    #     "year_abroad": "?undef",
    #     "year abroad": "?undef",
    #     "marriage": "?undef",
    #     "brush teeth": "?undef",
    #     "bath": "ex:Bathing",
    #     "clean house": "ex:Housekeeping",
    #     "care hygiene": "ex:Personal_Hygiene",
    #     "leave home": "ex:Leave_Home",
    #     "return home": "ex:Out_of_Home",
    #     "meal prepar": "ex:Meal_Preparation",
    #     "sleep": "ex:Sleep",
    #     "toilet": "ex:Bed_Toilet_Transition",
    #     "watch tv": "ex:Watch_TV",
    #     "work": "ex:Work",
    #     "eat": "ex:Eating",
    # }

    def __init__(self,
                 namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
                 nsmanager: Optional[NamespaceManager] = None,
                 vague_temp_upper_percentage=1.0,
                 vague_temp_lower_percentage=0.6,
                 vague_temp_ref_date=None,
                 param_path_dur = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "vaguetemp", "optimized_parameters_overAllVotes_event_duration"),
                 param_path_adv = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "vaguetemp", "optimized_parameters_overAllVotes_adverbials")):
        super().__init__(namespaces=namespaces, nsmanager=nsmanager)
        self.vague_temp_upper_percentage = vague_temp_upper_percentage
        self.vague_temp_lower_percentage = vague_temp_lower_percentage
        if vague_temp_ref_date is None:
            vague_temp_ref_date = RefDateWrapper(datetime.now())
        elif isinstance(vague_temp_ref_date, datetime):
            vague_temp_ref_date = RefDateWrapper(vague_temp_ref_date)
        elif not isinstance(vague_temp_ref_date, RefDateWrapper):
            raise ValueError("Invalid ref_date type")
        self.vague_temp_ref_date = vague_temp_ref_date
        self.param_path_dur = param_path_dur
        self.param_path_adv = param_path_adv

    @staticmethod
    def gauss_inverse(y, mean, std):
        return mean + std * np.sqrt(-2 * np.log(y))

    @staticmethod
    def inverse_event_specific_function(y, std):
        clipped_y = max(-1, min(1, 2 * y - 1))
        return math.sqrt(2) * std * erfinv(clipped_y)

    @staticmethod
    def safe_round(value):
        return round(value) if math.isfinite(value) else value

    def event_idx_det(self, event):
        event = event.lower()
        event_duration_idx = None
        if "brush" in event and "teeth" in event:
            event_duration_idx = self.events_durations.index("minutes")
        elif "bath" in event:
            event_duration_idx = self.events_durations.index("minutes")
        elif "clean" in event and "house" in event:
            event_duration_idx = self.events_durations.index("hours")
        elif "care" in event and "hygiene" in event:
            event_duration_idx = self.events_durations.index("minutes")
        elif "out" in event and "home" in event:
            event_duration_idx = self.events_durations.index("hours")
        elif ("leave" in event or "left" in event) and "home" in event:
            event_duration_idx = self.events_durations.index("minutes")
        elif "return" in event and "home" in event:
            event_duration_idx = self.events_durations.index("hours")
        elif "meal" in event and "prepar" in event:  # covering prepare and preparing
            event_duration_idx = self.events_durations.index("minutes")
        elif "sleep" in event or "slept" in event:
            event_duration_idx = self.events_durations.index("hours")
        elif "toilet" in event:
            event_duration_idx = self.events_durations.index("minutes")
        elif "watch" in event and "tv" in event:
            event_duration_idx = self.events_durations.index("hours")
        elif "work" in event:
            event_duration_idx = self.events_durations.index("hours")
        elif "eat" in event or "ate" in event:
            event_duration_idx = self.events_durations.index("minutes")
        else:
            event_duration_idx = None
            # raise ValueError(f"Invalid Event: {event}")
        return event_duration_idx

    @staticmethod
    def event_type(event):
        # event_types = {
        #     "brushing_teeth": "?undef",
        #     "brushing teeth": "?undef",
        #     "birthday": "?undef",
        #     "vacation": "?undef",
        #     "sabbatical": "?undef",
        #     "year_abroad": "?undef",
        #     "year abroad": "?undef",
        #     "marriage": "?undef",
        #     "brush teeth": "?undef",
        #     "bath": "ex:Bathing",
        #     "clean house": "ex:Housekeeping",
        #     "care hygiene": "ex:Personal_Hygiene",
        #     "leave home": "ex:Leave_Home",
        #     "return home": "ex:Out_of_Home",
        #     "meal prepar": "ex:Meal_Preparation",
        #     "sleep": "ex:Sleep",
        #     "toilet": "ex:Bed_Toilet_Transition",
        #     "watch tv": "ex:Watch_TV",
        #     "work": "ex:Work",
        #     "eat": "ex:Eating",
        # }
        event = event.lower()
        event_duration_idx = None
        if "brush" in event and "teeth" in event:
            return None#"?undef"
        elif "bath" in event:
            return "ex:Bathing"
        elif "clean" in event and "house" in event:
            return "ex:Housekeeping"
        elif "care" in event and "hygiene" in event:
            return "ex:Personal_Hygiene"
        elif ("leave" in event or "left" in event) and "home" in event:
            return "ex:Leave_Home"
        elif ("out" in event or "return" in event) and "home" in event:
            return "ex:Out_of_Home"
        elif "meal" in event and "prepar" in event:  # covering prepare and preparing
            return "ex:Meal_Preparation"
        elif "sleep" in event or "slept" in event:
            return "ex:Sleep"
        elif "toilet" in event:
            return "ex:Bed_Toilet_Transition"
        elif "watch" in event and "tv" in event:
            return "ex:Watch_TV"
        elif "work" in event:
            return "ex:Work"
        elif "eat" in event or "ate" in event:
            return "ex:Eating"
        else:
            return None#"?undef"

    def get_minutes_ago(self, adverbial, event):
        adverbial = adverbial.lower().strip().replace("_", " ")
        event = event.lower().strip().replace(" ", "_")

        chosen_events = []
        if self.event_idx_det(event) is None:
            chosen_events = self.all_events
        else:
            chosen_events = [event]

        res = []
        for curr_event in chosen_events:
            event_duration_idx = self.event_idx_det(curr_event)

            if adverbial not in self.all_adverbials:
                raise ValueError(f"Invalid adverbial: {adverbial}")

            with open(self.param_path_dur + '.pkl', 'rb') as f:
                optimized_event_duration_params = pickle.load(f)
                num_params = len(inspect.signature(self.inverse_event_specific_function).parameters) - 1
                event_params = optimized_event_duration_params[event_duration_idx:event_duration_idx + num_params]
                print(event_params)
            with open(self.param_path_adv + '.pkl', 'rb') as f:
                optimized_adverbial_params = pickle.load(f)
                idx = self.all_adverbials.index(adverbial)
                adverbial_params = [
                    optimized_adverbial_params[idx],
                    optimized_adverbial_params[idx + len(self.all_adverbials)]
                ]
                print(adverbial_params)

            upper_raw = self.inverse_event_specific_function(self.gauss_inverse(self.vague_temp_upper_percentage, *adverbial_params), *event_params)
            lower_raw = self.inverse_event_specific_function(self.gauss_inverse(self.vague_temp_lower_percentage, *adverbial_params), *event_params)

            upper = max(0, self.safe_round(upper_raw))
            lower = self.safe_round(lower_raw)

            res.append((upper, lower, self.event_type(curr_event)))# if len(chosen_events) > 1 else None
        print("Interval results: ", res, flush=True)
        return res

    # def get_minutes_ago(self, adverbial, event):
    #     adverbial = adverbial.lower().strip().replace("_", " ")
    #     event = event.lower().strip().replace(" ", "_")
    #     logging.debug(f"adverbial: {adverbial}, event: {event}")
    #     if "brush" in event and "teeth" in event:
    #         event = "brushing_teeth"
    #     elif "bath" in event:
    #         event = "brushing_teeth"
    #     elif "clean" in event and "house" in event:
    #         event = "brushing_teeth"
    #     elif "care" in event and "hygiene" in event:
    #         event = "brushing_teeth"
    #     elif "leave" in event and "home" in event:
    #         event = "brushing_teeth"
    #     elif "return" in event and "home" in event:
    #         event = "brushing_teeth"
    #     elif "meal" in event and "prepar" in event:#covering prepare and preparing
    #         event = "brushing_teeth"
    #     elif "sleep" in event:
    #         event = "brushing_teeth"
    #     elif "toilet" in event:
    #         event = "brushing_teeth"
    #     elif "watch" in event and "tv" in event:
    #         event = "brushing_teeth"
    #     elif "work" in event:
    #         event = "brushing_teeth"
    #     elif "eat" in event:
    #         event = "brushing_teeth"
    #     else:
    #         event = None
    #
    #     #event = "brushing_teeth"  # temporary fix until we have more data
    #     if adverbial not in self.all_adverbials:
    #         raise ValueError(f"Invalid adverbial: {adverbial}")
    #
    #     chosen_events = [event]
    #
    #     if event not in self.all_events:
    #         chosen_events = ["brushing_teeth", "brushing_teeth"]
    #
    #     with open(self.param_path + '.pkl', 'rb') as f:
    #         optimized_params = pickle.load(f)
    #
    #     num_params = len(inspect.signature(self.inverse_event_specific_function).parameters) - 1
    #     idx = self.all_adverbials.index(adverbial)
    #
    #     adverbial_params = [
    #         optimized_params[len(self.all_events) * num_params + idx],
    #         optimized_params[
    #             len(self.all_events) * num_params + idx + len(self.all_adverbials)]
    #     ]
    #
    #     result = []
    #
    #     for event in chosen_events:
    #
    #         event_idx = self.all_events.index(event)
    #         event_params = optimized_params[event_idx:event_idx + num_params]
    #
    #         upper_raw = self.inverse_event_specific_function(self.gauss_inverse(self.vague_temp_upper_percentage, *adverbial_params),
    #                                                          *event_params)
    #         lower_raw = self.inverse_event_specific_function(self.gauss_inverse(self.vague_temp_lower_percentage, *adverbial_params),
    #                                                          *event_params)
    #
    #         upper = max(0, self.safe_round(upper_raw))
    #         lower = self.safe_round(lower_raw)
    #
    #         result.append((upper, lower, self.event_types[event] if len(chosen_events) > 1 else None))
    #     return result

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
        #data["include_redundant"] = True

        if pred.lower() == "local:vaguetemp":
            assert len(var_order) == 3

            adverb = utils.rem_quotes(str(var_order[0]))
            event = utils.rem_quotes(str(var_order[1]))
            var = self.var_or_value(var_order[2], dudes, data)

            mins_ago = self.get_minutes_ago(adverb, event)
            assert len(mins_ago) > 0

            # if len(mins_ago) == 1:
            #     triples.extend([
            #         Triple(subject=var, predicate="ex:happensAt", object=var + "Interval"),
            #         Triple(subject=var + "Interval", predicate="time:hasEnd", object=var + "End"),
            #     ])
            #     upper, lower, event_type = mins_ago[0]
            #     assert min(upper, lower) < float('inf')
            #     upper_date = self.vague_temp_ref_date.ref_date - timedelta(minutes=min(upper, lower))
            #     data["filterdata"].append(FilterData(
            #         var=var+"End",
            #         operator="<=",
            #         num='"' + upper_date.isoformat() + '"',
            #         count=False,
            #         datetime=True
            #     ))
            #
            #     if max(upper, lower) < float('inf'):
            #         lower_date = self.vague_temp_ref_date.ref_date - timedelta(minutes=max(upper, lower))
            #         data["filterdata"].append(FilterData(
            #             var=var+"End",
            #             operator=">=",
            #             num='"' + lower_date.isoformat() + '"',
            #             count=False,
            #             datetime=True
            #         ))
            #
            #     updated = True
            # else:
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
                # fdisj = FilterCombinator(combinator=" || ")
                # for upper, lower, event_type in mins_ago:
                #     assert min(upper, lower) < float('inf')
                #     upper_date = self.vague_temp_ref_date.ref_date - timedelta(minutes=min(upper, lower))
                #     fconj = FilterCombinator(combinator=" && ")
                #     fconj.filters.append(FilterData(
                #         var=var + "End",
                #         operator="<=",
                #         num='"' + upper_date.isoformat() + '"',
                #         count=False,
                #         datetime=True
                #     ))
                #
                #     if max(upper, lower) < float('inf'):
                #         lower_date = self.vague_temp_ref_date.ref_date - timedelta(minutes=max(upper, lower))
                #         fconj.filters.append(FilterData(
                #             var=var + "End",
                #             operator=">=",
                #             num='"' + lower_date.isoformat() + '"',
                #             count=False,
                #             datetime=True
                #         ))
                #
                #     updated = True
                #     fdisj.filters.append(fconj)
                # data["filterdata"].append(fdisj)


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

