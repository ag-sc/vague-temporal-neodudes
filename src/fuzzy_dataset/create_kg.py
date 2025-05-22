#!/usr/bin/env python3

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import XSD, RDF, RDFS, OWL
import json
import argparse

# Constants
FILE_PATH = '_activities.json'
SAVE_TTL = "household_events.ttl"
RESIDENTS = ["R1", "R2"]
NEW_RESIDENTS_NAME = ["Tom", "Mary"]
EVENTS_MAPPINGS = {
    "Bed_Toilet_Transition": "go to the toilet",
    "Personal_Hygiene": "take care of personal hygiene",
    "Meal_Preparation": "do meal preparation",
    "Out_of_Home": "return home",
    "Leave_Home": "leave the home",
    "Eating": "eat",
    "Watch_TV": "watch TV",
    "Housekeeping": "clean the house",
    "Bathing": "take a bath"
}

def get_activities_resident():
    all_activities = []
    i = 0
    for resident, new_resident_name in zip(RESIDENTS, NEW_RESIDENTS_NAME):
        with open(resident + FILE_PATH, 'r') as file:
            data = json.load(file)
        for activity in data:
            activity["agent"] = new_resident_name
            activity["id"] = activity["activity"]+"_"+str(i)
            i += 1
        all_activities.extend(data)
    return all_activities


def create_graph(events):
    EX = Namespace("http://example.org/")
    DUL = Namespace("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#")
    TIME = Namespace("http://www.w3.org/2006/time#")
    agents = {
        "Tom": EX.Tom,
        "Mary": EX.Mary,
    }

    g = Graph()
    g.bind("ex", EX)
    g.bind("dul", DUL)
    g.bind("time", TIME)

    g.add((EX.Event, RDF.type, OWL.Class))
    g.add((DUL.Agent, RDF.type, OWL.Class))
    g.add((TIME.ProperInterval, RDF.type, OWL.Class))

    for name, uri in agents.items():
        g.add((uri, RDF.type, DUL.Agent))
        g.add((uri, RDFS.label, Literal(name, datatype=XSD.string)))

    for e in events:
        event_uri = EX[e["id"]]
        event_class = EX[e["activity"]]
        agent_uri = agents[e["agent"]]

        g.add((event_class, RDF.type, OWL.Class))
        g.add((event_class, RDFS.subClassOf, EX.Event))

        g.add((event_uri, RDF.type, event_class))

        interval_uri = EX[f"Interval_{e['id']}"]
        g.add((interval_uri, RDF.type, TIME.ProperInterval))
        g.add((interval_uri, TIME.hasBeginning, Literal(e["begin"], datatype=XSD.dateTime)))
        g.add((interval_uri, TIME.hasEnd, Literal(e["end"], datatype=XSD.dateTime)))

        g.add((event_uri, EX.happensAt, interval_uri))

        g.add((event_uri, DUL.hasAgent, agent_uri))

    g.serialize(SAVE_TTL, format="turtle")
    print("RDF data saved to " + SAVE_TTL)


def main():
    # Argument Parsing (but no actual arguments required)
    parser = argparse.ArgumentParser(description="Preprocess the defined dataset and generate RDF graph.")
    parser.parse_args()

    all_activities = get_activities_resident()
    create_graph(all_activities)


if __name__ == "__main__":
    main()