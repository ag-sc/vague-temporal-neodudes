import argparse
import csv
import datetime
import json
import logging
import multiprocessing
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict
from multiprocessing import Pool

import jsonlines
import pandas as pd
import torch
from sklearn.model_selection import train_test_split  # type: ignore

from dudes import utils
from dudes.qa.qa_pipeline import QAPipeline
from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint
from dudes.qa.sparql.triple_modules import RefDateWrapper
from dudes.qa.sparql_selection.llm_query_selector import MultiLLMQuerySelector
from dudes.qa.sparql_selection.query_evaluator import QueryEvaluator
from dudes.utils import EvalStats

logging.basicConfig(level=logging.DEBUG, force=True)

import z3  # type: ignore

from dudes.qa.sparql.sparqlburger.SPARQLQueryBuilder import *  # type: ignore

from lemon.lemon_parser import LEMONParser

import compress_pickle as cpl  # type: ignore


def _llm_fuzzy_eval_init():
    global query_scorer
    query_scorer = MultiLLMQuerySelector.from_paths([os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "comp_query_score_models", "query_score_llm_fuzzy_2.2592503361963202e-05_0.9451347279061211_64_5_2025-05-13_08-54-39-698556_best_val_loss.ckpt")])
    global se
    path = os.path.join(
        os.path.dirname(sys.modules["lemon"].__file__),
        "resources",
        "vaguetemp",
        "household_events.ttl"
    )
    se = SPARQLEndpoint(path)


def _llm_fuzzy_eval_thread(qdata):
    # gold_numres = 0
    # if len(qdata[0]["Ground Truth"]) > 2:
    #     gold_numres = qdata[0]["Ground Truth"].count(",") + 1

    gtres = []
    if qdata[0]["Ground Truth"] == "Yes" or qdata[0]["Ground Truth"] == "No":
        gtres = [qdata[0]["Ground Truth"]]
    else:
        print("GT: ", qdata[0]["Ground Truth"], "-", flush=True)
        gt = json.loads(qdata[0]["Ground Truth"].replace("'", '"'))
        assert isinstance(gt, list)
        gtres = ["http://example.org/" + r if not r.startswith("http") else r for r in gt]

    gtresset = set(gtres)

    # gold_numres = 0
    # if qdata[0]["Ground Truth"] == "Yes" or qdata[0]["Ground Truth"] == "No":
    #     gold_numres = 1
    # elif len(qdata[0]["Ground Truth"]) != "[]":
    #     gold_numres = qdata[0]["Ground Truth"].count(",") + 1

    qe = QueryEvaluator.default(gold=gtres, question=qdata[0]["Question"], preset="fuzzy", rpc_conn=None, query_scorer=query_scorer)

    #for idx, i in enumerate(ids):
    #    if idx >= 100:
    #        print("Stopping for 100 queries")
    #        break
    for row in qdata:
        print(row, qe.best_stats)
        predres = []
        if row["Prediction"] == "Yes" or row["Prediction"] == "No":
            predres = [row["Prediction"]]
        else:
            pred = json.loads(row["Prediction"].replace("'", '"'))
            assert isinstance(pred, list)
            predres = ["http://example.org/" + r if not r.startswith("http") else r for r in pred]

        predresset = set(predres)

        # numres = 0
        # if row["Prediction"] == "Yes" or row["Prediction"] == "No":
        #     numres = 1
        # elif len(row["Prediction"]) != "[]":
        #     numres = row["Prediction"].count(",") + 1

        stats = EvalStats(tp=len(gtresset.intersection(predresset)), fp=len(predresset.difference(gtresset)), fn=len(gtresset.difference(predresset)), emc=1 if float(row["F1"]) > 0.99 else 0)
        qe.eval(curr_stats=stats, query=row["SPARQL Query"], dudes=row["DUDES"], full_query=row["SPARQL Query"])
    return {
        "ID": qdata[0]["ID"],
        "Question": qdata[0]["Question"],
        "Ref Date": qdata[0]["Ref Date"],
        "best_stats": {k: v.to_dict() for k, v in qe.best_stats.items()},
        "best_query": qe.best_query,
        "best_dudes": qe.best_dudes
    }

def test_query_eval(results_path, out_path, threads):
    if out_path is None:
        out_path = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "vaguetemp", f"2025-05-07-full-final-evaluation_data_results_llm_selected.jsonl")

    #thr = rpc_thread(preset="fuzzy")
    #thr = dudes_rpc_service.start_rpc_service(preset="fuzzy")
    print("Query eval started.")
    df = pd.read_csv(results_path)
    torch.multiprocessing.set_start_method('spawn')
    # max_first_correct_result = None

    # rpc_conn = rpyc.connect(consts.rpc_host,
    #                         consts.rpc_port,
    #                         config={
    #                             "allow_public_attrs": True,
    #                             "allow_pickle": True,
    #                             "sync_request_timeout": 300
    #                         })

    with Pool(processes=threads, initializer=_llm_fuzzy_eval_init) as pool:
        data = [
            [
                df.loc[i].to_dict()
                for idx, i in enumerate(ids) if idx < 64
            ]
            for q, ids in df.groupby("ID").groups.items()
        ]
        print(len(data), flush=True)
        with open(out_path, "a+") as f:
            with jsonlines.Writer(f) as writer:
                for res in pool.imap_unordered(_llm_fuzzy_eval_thread, data):
                    writer.write(res)
                    f.flush()

def test_min_result_num_perfect():
    df = pd.read_csv(os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "vaguetemp", "2025-05-07-full-final-evaluation_data_results.csv.zst"))
    max_first_correct_result = None
    for q, ids in df.groupby("ID").groups.items():
        found = False
        for idx, i in enumerate(ids):
            if float(df.loc[i]["F1"]) > 0.99:
                de = df.loc[i]
                if "FILTER" in de['SPARQL Query'] and (("What" in de["Question"] and "UNION" in de["SPARQL Query"]) or ("What" not in de["Question"] and "UNION" not in de["SPARQL Query"])):
                    found = True
                    if max_first_correct_result is None or max_first_correct_result < idx:
                        max_first_correct_result = idx
                        break
        if not found:
            for i in ids:
                print(df.loc[i].to_dict())
            raise AssertionError()
        print(max_first_correct_result)


def test_endpoint():
    path = os.path.join(
        os.path.dirname(sys.modules["lemon"].__file__),
        "resources",
        "vaguetemp",
        "household_events.ttl"
    )
    se = SPARQLEndpoint(path)

    q5 = """
        PREFIX brick: <https://brickschema.org/schema/Brick#>
        PREFIX csvw: <http://www.w3.org/ns/csvw#>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        PREFIX dcat: <http://www.w3.org/ns/dcat#>
        PREFIX dcmitype: <http://purl.org/dc/dcmitype/>
        PREFIX dcam: <http://purl.org/dc/dcam/>
        PREFIX doap: <http://usefulinc.com/ns/doap#>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX geo: <http://www.opengis.net/ont/geosparql#>
        PREFIX odrl: <http://www.w3.org/ns/odrl/2/>
        PREFIX org: <http://www.w3.org/ns/org#>
        PREFIX prof: <http://www.w3.org/ns/dx/prof/>
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX qb: <http://purl.org/linked-data/cube#>
        PREFIX schema: <https://schema.org/>
        PREFIX sh: <http://www.w3.org/ns/shacl#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX sosa: <http://www.w3.org/ns/sosa/>
        PREFIX ssn: <http://www.w3.org/ns/ssn/>
        PREFIX time: <http://www.w3.org/2006/time#>
        PREFIX vann: <http://purl.org/vocab/vann/>
        PREFIX void: <http://rdfs.org/ns/void#>
        PREFIX wgs: <https://www.w3.org/2003/01/geo/wgs84_pos#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX xml: <http://www.w3.org/XML/1998/namespace>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbp: <http://dbpedia.org/property/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX dbc: <http://dbpedia.org/resource/Category:>
        PREFIX dct: <http://purl.org/dc/terms/>
        PREFIX oils: <http://localhost:8000/oils.owl/>
        PREFIX local: <http://localhost:8000/#>
        PREFIX yago: <http://dbpedia.org/class/yago/>
        PREFIX yago-res: <https://yago-knowledge.org/resource/>
        PREFIX ex: <http://example.org/>
        PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
        PREFIX lexinfo: <http://www.lexinfo.net/ontology/2.0/lexinfo#>
        PREFIX lemon: <http://localhost:8000/lemon.owl#>
        PREFIX lexicon: <http://localhost:8000/lexicon#>
        ASK 
WHERE {
   ?v0 rdf:type <http://example.org/Eating> . 
   ?v0 <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#hasAgent> <http://example.org/Tom> . 
   <http://example.org/Tom> rdf:type <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Agent> . 
   ?v0 ex:happensAt ?v0Interval . 
   {
      ?v0Interval time:hasEnd ?v0End . 
      ?v0 rdf:type ex:Eating . 
      FILTER (xsd:dateTime(?v0End) <= xsd:dateTime(\"2014-10-20T02:52:43.019818\"))
   }
}
        """

    r3 = se.endpoint_graph.query(q5)
    print(list(r3))

def test_lexicon():
    lexicon = LEMONParser.from_ttl_dir(
        ttl_dir=os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "lexicon_fuzzy"
        ),
        csv_dir = None,
    ).lexicon
    pass

def test_fuzzy_single():
    path = os.path.join(
        os.path.dirname(sys.modules["lemon"].__file__),
        "resources",
        "vaguetemp",
        "household_events.ttl"
    )

    eval_path = os.path.join(
        os.path.dirname(sys.modules["lemon"].__file__),
        "resources",
        "vaguetemp",
        "evaluation_data_who.json"# change to other part of the dataset as needed
    )

    evalds = json.load(open(eval_path))

    se = SPARQLEndpoint(path)

    ref_date = RefDateWrapper(datetime.datetime.fromisoformat("2009-10-25T13:17:38.004601"))
    qa = QAPipeline.default(
        preset="fuzzy",
        #vague_temp_ref_date=datetime.datetime.fromisoformat("2009-10-30T16:52:38.004601")
        vague_temp_ref_date=ref_date
    )
    # testid = 12
    # question = "Did Tom eat recently?"

    for i, d in enumerate(evalds):
        #Add your filter here
        # if i != testid:
        #     continue
        # if "2012-02-07T12:20:25.053" not in d["ref_date"] or "Who watched TV a long time ago?" not in d["question"]:
        #     continue
        ref_date.ref_date = datetime.datetime.fromisoformat(d["ref_date"])
        question = d["question"]
        qres = list(set(qa.process_query_and_dudes(question)))
        qres = sorted([(q, d) for q, d in qres if q.count("hasAgent") > 0],
                      key=lambda x: (-x[0].count("hasAgent"), x[0].count("\n")))
        if len(qres) == 0:
            q = ""
            logging.error("No query found for question: %s", question)

        return_result = []
        for q, q_dudes in qres:
            res = se.get_results_query(q)
            gt = d["gt"]
            matchres = None

            if isinstance(gt, list):
                pred = [str(r) for r in res]
                gt = ["http://example.org/" + r if not r.startswith("http") else r for r in gt]

                # if "UNION" not in q and "FILTER" in q:
                print("Length of gt:", len(gt))
                print("Length of pred:", len(pred))
                print("Intersection of gt and pred:", len(set(gt).intersection(set(pred))))
                print("Difference of gt and pred:", set(gt).symmetric_difference(set(pred)))
                if set(gt) == set(pred):
                    matchres = True
                else:
                    matchres = False
            else:
                pred = "Yes" if True in res else "No"
                if gt == pred:
                    matchres = True
                else:
                    matchres = False
            # for q in res:
            # if "UNION" not in q and "FILTER" in q:
            print(i, question, d["ref_date"], se.get_results_query(q), gt, pred, "MATCH" if matchres else "NO MATCH", "\n", utils.remove_prefix(q), q_dudes)
            return_result.append([i, question, str(d["ref_date"]), utils.remove_prefix(q), gt, pred, "MATCH" if matchres else "NO MATCH", 1 if matchres else 0, q_dudes])
        pass
        break

def fuzzy_initializer():
    global se
    global ref_date
    global qa
    path = os.path.join(
        os.path.dirname(sys.modules["lemon"].__file__),
        "resources",
        "vaguetemp",
        "household_events.ttl"
    )
    se = SPARQLEndpoint(endpoint=path)
    ref_date = RefDateWrapper(datetime.datetime.fromisoformat("2009-10-25T13:17:38.004601"))
    qa = QAPipeline.default(
        preset="fuzzy",
        # vague_temp_ref_date=datetime.datetime.fromisoformat("2009-10-30T16:52:38.004601")
        vague_temp_ref_date=ref_date
    )

def fuzzy_eval_row(data):
    i, d = data
    ref_date.ref_date = datetime.datetime.fromisoformat(d["ref_date"])
    question = d["question"]
    # maxresq = ""
    # for resq in qa.process(question):
    #     resq_score = (-resq.count("hasAgent"), resq.count("\n"))
    #     maxresq_score = (-maxresq.count("hasAgent"), maxresq.count("\n"))
    #     if resq_score > maxresq_score and resq.count("hasAgent") > 0: #len(resq) > len(maxresq):
    #         maxresq = resq
    #
    # q = maxresq
    qres = list(set(qa.process_query_and_dudes(question)))
    qres = sorted([(q, d) for q, d in qres if q.count("hasAgent") > 0], key=lambda x: (-x[0].count("hasAgent"), x[0].count("\n")))
    if len(qres) == 0:
        q = ""
        logging.error("No query found for question: %s", question)
    # else:
    #     q = qres[-1]
    return_result = []
    for q, q_dudes in qres:
        res = se.get_results_query(q)
        gt = d["gt"]
        matchres = None

        if isinstance(gt, list):
            pred = [str(r) for r in res]
            gt = ["http://example.org/" + r if not r.startswith("http") else r for r in gt]
            if set(gt) == set(pred):
                matchres = True
            else:
                matchres = False
        else:
            pred = "Yes" if True in res else "No"
            if gt == pred:
                matchres = True
            else:
                matchres = False
        # for q in res:
        print(i, question, d["ref_date"], se.get_results_query(q), gt, pred, "MATCH" if matchres else "NO MATCH", "\n", utils.remove_prefix(q), q_dudes)
        return_result.append([i, question, str(d["ref_date"]), utils.remove_prefix(q), gt, pred, "MATCH" if matchres else "NO MATCH", 1 if matchres else 0, q_dudes])
    return return_result

def test_split_fuzzy():
    eval_paths = [
        os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_who.json"
        ),
        os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_what.json"
        ),
        os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_what_happened.json"
        ),
        os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_did.json"
        ),
    ]

    eval_paths_dev = [
        os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_who_dev.json"
        ),
        os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_what_dev.json"
        ),
        os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_what_happened_dev.json"
        ),
        os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_did_dev.json"
        ),
    ]

    eval_path_out = os.path.join(
        os.path.dirname(sys.modules["lemon"].__file__),
        "resources",
        "vaguetemp",
        "evaluation_data_results.csv"
    )

    evalds = [(d | {"id": i}) for i, d in enumerate(sum([json.load(open(ep)) for ep in eval_paths], []))]
    train, test = train_test_split(evalds, test_size=0.7, train_size=0.2, random_state=42)
    val = [d for d in evalds if d not in train and d not in test]

    json.dump(train, open(
        os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_train.json"
        ), "w")
    )
    json.dump(val, open(
        os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_val.json"
        ), "w")
    )
    json.dump(test, open(
        os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_test.json"
        ), "w")
    )

    by_question = defaultdict(list)
    for d in train:
        by_question[d["question"]].append(d | {"id": i})

    #for q, data in by_question.items():
    pass


def test_fuzzy(eval_path_out, threads):
    if eval_path_out is None:
        eval_path_out = os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_results.csv"
        )

    path = os.path.join(
        os.path.dirname(sys.modules["lemon"].__file__),
        "resources",
        "vaguetemp",
        "household_events.ttl"
    )

    eval_paths = [
        os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_who.json"
        ),
        os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_what.json"
        ),
        os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_what_happened.json"
        ),
        os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_did.json"
        ),
    ]

    eval_paths_dev = [
        os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_who_dev.json"
        ),
        os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_what_dev.json"
        ),
        os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_what_happened_dev.json"
        ),
        os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "vaguetemp",
            "evaluation_data_did_dev.json"
        ),
    ]

    evalds = [(i, d) for i, d in enumerate(sum([json.load(open(ep)) for ep in eval_paths], []))]

    with Pool(processes=threads, initializer=fuzzy_initializer) as pool:
        with open(eval_path_out, "a+") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Question", "Ref Date", "SPARQL Query", "Ground Truth", "Prediction", "Match", "F1", "DUDES"])
            for rows in pool.imap_unordered(fuzzy_eval_row, evalds):
                any_match = False
                for row in rows:
                    writer.writerow(row)
                    if float(row[-2]) > 0.9999:
                        any_match = True
                if not any_match:
                    if len(rows) > 0:
                        print("No match found for question:", list(rows)[0][0], list(rows)[0][1], flush=True)
                    else:
                        print("No match found!", flush=True)
                else:
                    print("Match found for question:", list(rows)[0][0], list(rows)[0][1], flush=True)
                f.flush()

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--threads", type=int, default=multiprocessing.cpu_count())
    argparser.add_argument("--outpath", type=str, default=None)

    argparser.add_argument('--eval', action=argparse.BooleanOptionalAction)

    argparser.add_argument('--llmselect', action=argparse.BooleanOptionalAction)
    argparser.add_argument("--resultspath", type=str, default=os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "vaguetemp", "2025-05-07-full-final-evaluation_data_results.csv.zst"))

    arguments = argparser.parse_args()

    if arguments.llmselect:
        test_query_eval(results_path=arguments.resultspath, out_path=arguments.outpath, threads=arguments.threads)
    elif arguments.eval:
        test_fuzzy(eval_path_out=arguments.outpath, threads=arguments.threads)
    else:
        raise ValueError("Please specify --llmselect or --eval")
    #test_fuzzy()
    #test_qald()
    #test_qa_pipeline()
