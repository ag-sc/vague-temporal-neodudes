#!/usr/bin/env python3

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import math
import matplotlib.dates as mdates
import pandas as pd
import json
import random
from collections import defaultdict
import rdflib
import argparse

from VagueTempModell import vague_temp


dateformat = '%Y-%m-%dT%H:%M:%S.%f'
max_no_range_minutes = 150
dataset_start_date = "2009-08-23T00:00:00.000000"
dataset_end_date = "2010-05-01T00:00:00.000000"
#Dataset range in minutes + 1 month, Infinity value for long time ago
infinity_minutes = ((datetime.fromisoformat(dataset_end_date) - datetime.fromisoformat(dataset_start_date)).total_seconds() / 60)+43800

# # Arguments
# NUM_WHAT_QUESTIONS_PER_ADVERBIAL = 100 #Total: * (Num_adverbials + num_residents)
# NUM_WHO_QUESTIONS_PER_ADVERBIAL = 200 #Total * Num_Adverbials
# NUM_WHAT_HAPPENED_QUESTIONS_PER_ADVERBIAL = 100
# NUM_DID_QUESTIONS_PER_EVENT = 5 # Total: * Num_Events

# Constants
FILE_PATH = '_activities.json'
RESIDENTS = ["R1", "R2"]
NEW_RESIDENTS_NAME = ["Tom", "Mary"]
ADVERBIALS = ["long time ago", "some time ago", "recently", "just"]
EVENTS_MAPPINGS = {
    "Bed_Toilet_Transition": "go to the toilet",
    "Personal_Hygiene": "take care of personal hygiene",
    "Meal_Preparation": "do meal preparation",
    "Out_of_Home": "return home",
    "Leave_Home": "leave the home",
    "Eating": "eat",
    "Watch_TV": "watch TV",
    "Housekeeping": "clean the house",
    "Bathing": "take a bath",
    "Sleep": "sleep"
}
EVENTS_MAPPINGS_PAST = {
    "Bed_Toilet_Transition": "went to the toilet",
    "Personal_Hygiene": "took care of personal hygiene",
    "Meal_Preparation": "prepared a meal",
    "Out_of_Home": "returned home",
    "Leave_Home": "left the home",
    "Eating": "ate",
    "Watch_TV": "watched TV",
    "Housekeeping": "cleaned the house",
    "Bathing": "took a bath",
    "Sleep": "slept"
}


def get_activities_resident(resident="R1"):
    with open(resident + FILE_PATH, 'r') as file:
        data = json.load(file)
    grouped_activities = defaultdict(list)
    for activity in data:
        grouped_activities[activity['activity']].append(activity)
    # Sort each activity list by the 'begin' time
    for activity_list in grouped_activities.values():
        activity_list.sort(key=lambda x: datetime.fromisoformat(x['begin']))
    return grouped_activities, data



def get_possible_activity(grouped_activities, answer_type, higher_percentage_minutes, lower_percentage_minutes):
    if higher_percentage_minutes > lower_percentage_minutes:
        interval_minutes_start = higher_percentage_minutes
        interval_minutes_end = lower_percentage_minutes
    else:
        interval_minutes_start = lower_percentage_minutes
        interval_minutes_end = higher_percentage_minutes
    if not answer_type:
        # Return activities and ref_dates with a "no" answer
        if interval_minutes_start in [None, float('inf')]:
            # For long time ago only the first activity is possible otherwise any activity of the same type will be a "long time ago"
            reference_activity = grouped_activities[0]
            end = datetime.fromisoformat(reference_activity["end"])
            reference_dates = [(end + timedelta(minutes=random.randint(0, interval_minutes_end-1))).strftime(dateformat) for _ in range(NUM_DID_QUESTIONS_PER_EVENT)]
            return reference_activity, reference_dates
        else:
            # All possible activities must be before the interval (before interval_minutes_start)
            # But there should be no other activity in between
            possible_activities = []
            outside_minutes = max_no_range_minutes + interval_minutes_start

            if len(grouped_activities) == 1:
                reference_activity = grouped_activities[0]
                end_ref_act = datetime.fromisoformat(reference_activity["end"])
                reference_dates = [
                    (end_ref_act + timedelta(minutes=random.randint(interval_minutes_start+1, outside_minutes))).strftime(
                        dateformat) for _ in range(NUM_DID_QUESTIONS_PER_EVENT)]
                return reference_activity, reference_dates

            for current_activity, next_activity in zip(grouped_activities, grouped_activities[1:] + [None]):
                if next_activity:
                    end_ref_act = datetime.fromisoformat(current_activity["end"])
                    begin_next_act = datetime.fromisoformat(next_activity["begin"])
                    if begin_next_act > (end_ref_act + timedelta(minutes=outside_minutes)):
                        possible_activities.append(current_activity)
            if len(possible_activities) == 0:
                # print("Warning: If just or recently or some time ago no possible activities for "+grouped_activities[0]["activity"])
                # print("TODO: for some time ago the activity could also be between interval_end and ref_date")
                return None, []
            else:
                reference_activity = random.choice(possible_activities)
                end_ref_act = datetime.fromisoformat(reference_activity["end"])
                reference_dates = [(end_ref_act + timedelta(minutes=random.randint(interval_minutes_start+1, outside_minutes))).strftime(dateformat) for _ in range(NUM_DID_QUESTIONS_PER_EVENT)]
                return reference_activity, reference_dates
    else:
        # Return activities and ref_dates with a "yes" answer
        reference_activity = random.choice(grouped_activities)
        end_ref_act = datetime.fromisoformat(reference_activity["end"])
        if interval_minutes_start in [None, float('inf')]:
            interval_minutes_start = 9999999
        reference_dates = [
            (end_ref_act + timedelta(minutes=random.randint(interval_minutes_end, interval_minutes_start))).strftime(
                dateformat) for _ in range(NUM_DID_QUESTIONS_PER_EVENT)]
        return reference_activity, reference_dates

def get_did_questions(random_activities, adverbial, resident_name):
    evaluation_data = []
    for event in random_activities:
        event_name = EVENTS_MAPPINGS.get(event, event.lower())

        if adverbial == "just":
            question = f"Did {resident_name} just {event_name}?"
        elif adverbial == "long time ago":
            question = f"Did {resident_name} {event_name} a {adverbial}?"
        else:
            question = f"Did {resident_name} {event_name} {adverbial}?"

        # Create "Yes" Questions
        ref_activity, ref_dates = get_possible_activity(random_activities[event], True,
                                                        *vague_temp.get_minutes_ago(event, adverbial))
        for ref_date in ref_dates:
            data_dict = {
                'ref_date': ref_date,
                'activity_end_date': ref_activity['end'],
                'question': question,
                'gt': "Yes"
            }
            evaluation_data.append(data_dict)

        # Create "No" Questions
        ref_activity, ref_dates = get_possible_activity(random_activities[event], False,
                                                        *vague_temp.get_minutes_ago(event, adverbial))

        for ref_date in ref_dates:
            data_dict = {
                'ref_date': ref_date,
                'activity_end_date': ref_activity['end'],
                'question': question,
                'gt': "No"
            }
            evaluation_data.append(data_dict)
    return evaluation_data


def get_what_questions(random_activities, adverbial, resident_name):
    def load_ttl(file_path="household_events.ttl"):
        g = rdflib.Graph()
        g.parse(file_path, format='ttl')
        return g
    def check_exact_match(interval_1_start, interval_1_end, interval_2_start, interval_2_end):
        return (interval_1_start == interval_2_start) and (interval_1_end == interval_2_end)
    def parse_datetime(date_str):
        try:
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f")
    def extract_all_events(graph):
        events = []
        query = """
        SELECT ?work ?event ?begin ?end WHERE {
            ?work ex:happensAt ?event .  # Get the work and its associated event
            ?event time:hasBeginning ?begin .
            ?event time:hasEnd ?end .
        }
        """
        result = graph.query(query)
        for row in result:
            event_uri = row['work']  # Extract the vent uri
            begin = row['begin'].toPython()  # Get the beginning time
            end = row['end'].toPython()  # Get the end time
            events.append({
                'event_uri': event_uri,
                'begin': parse_datetime(str(begin)),
                'end': parse_datetime(str(end))
            })
        return events

    evaluation_data = []
    if adverbial == "just":
        question = f"What has {resident_name} {adverbial} done?"
    elif adverbial == "long time ago":
        question = f"What did {resident_name} do a {adverbial}?"
    elif adverbial == "some time ago":
        question = f"What did {resident_name} do {adverbial}?"
    else:
        question = f"What has {resident_name} done {adverbial}?"

    graph_events = extract_all_events(load_ttl())
    for _ in range(NUM_WHAT_QUESTIONS_PER_ADVERBIAL):
        random_seconds = random.randint(0,
                                        (datetime.strptime(dataset_end_date, dateformat)
                                         - datetime.strptime(dataset_start_date, dateformat)).total_seconds())
        ref_date = datetime.strptime(dataset_start_date, dateformat) + timedelta(seconds=random_seconds)
        gt = []
        for act in random_activities:
            act_end_date = datetime.strptime(act["end"], dateformat)
            if act_end_date <= ref_date:
                lower_percentage, higher_percentage = vague_temp.get_minutes_ago(act["activity"], adverbial)
                if higher_percentage == math.inf:
                    higher_percentage = infinity_minutes
                higher_bound = ref_date-timedelta(minutes=lower_percentage)
                lower_bound = ref_date-timedelta(minutes=higher_percentage)
                if lower_bound <= act_end_date <= higher_bound:
                    event_start = parse_datetime(act['begin'])
                    event_end = parse_datetime(act['end'])
                    for event in graph_events:
                        ttl_start = event['begin']
                        ttl_end = event['end']
                        if check_exact_match(ttl_start, ttl_end, event_start, event_end):
                            if event["event_uri"] not in gt:
                                gt.append(event["event_uri"])
        data_dict = {
            'ref_date': ref_date.strftime(dateformat),
            'question': question,
            'gt': gt
        }
        evaluation_data.append(data_dict)
    return evaluation_data

def get_who_question(event_name, adverbial):
    if adverbial == "just":
        question = f"Who has just {event_name}?"
    elif adverbial == "long time ago":
        question = f"Who {event_name} a {adverbial}?"
    elif adverbial == "some time ago":
        question = f"Who {event_name} {adverbial}?"
    else:
        question = f"Who has {event_name} {adverbial}?"
    return question
def create_gt_who_one_resident(resident_events, reference_resident, other_resident_events, other_resident, adverbial, event):
    gt = []
    nearer_minutes, further_minutes = vague_temp.get_minutes_ago(event, adverbial)
    if further_minutes == math.inf:
        further_minutes = infinity_minutes
    if resident_events:
        ref_date = datetime.strptime(random.choice(resident_events)["end"], dateformat) + timedelta(
            minutes=random.randint(nearer_minutes, further_minutes))
        gt.append(NEW_RESIDENTS_NAME[reference_resident])
        higher_bound = ref_date - timedelta(minutes=nearer_minutes)
        lower_bound = ref_date - timedelta(minutes=further_minutes)
        for r in other_resident_events:
            act_end_date = datetime.strptime(r["end"], dateformat)
            if act_end_date <= ref_date:
                if lower_bound <= act_end_date <= higher_bound:
                    gt.append(NEW_RESIDENTS_NAME[other_resident])
                    break
            else:
                break
        data_dict = {
            'ref_date': ref_date.strftime(dateformat),
            'question': get_who_question(EVENTS_MAPPINGS_PAST.get(event, event.lower()), adverbial),
            'gt': gt
        }
        return data_dict
    else:
        # print("WARNING: No Activities for " + str(NEW_RESIDENTS_NAME[reference_resident]) + " and " + str(event) + ": Skipping")
        return None
def create_gt_who_zero_resident(r1_events, r2_events, event, adverbial):
    nearer_minutes, further_minutes = vague_temp.get_minutes_ago(event, adverbial)
    if further_minutes == math.inf:
        further_minutes = infinity_minutes
    nearer_minutes += 1
    further_minutes -= 1
    if r1_events and r2_events:
        ref_date = datetime.strptime(min(r1_events[0]["end"], r2_events[0]["end"]), dateformat) - timedelta(
                minutes=random.randint(nearer_minutes, further_minutes))
    elif r1_events:
        ref_date = datetime.strptime(r1_events[0]["end"], dateformat) - timedelta(
            minutes=random.randint(nearer_minutes, further_minutes))
    else:
        ref_date = datetime.strptime(r2_events[0]["end"], dateformat) - timedelta(
            minutes=random.randint(nearer_minutes, further_minutes))
    data_dict = {
        'ref_date': ref_date.strftime(dateformat),
        'question': get_who_question(EVENTS_MAPPINGS_PAST.get(event, event.lower()), adverbial),
        'gt': []
    }
    return data_dict

def get_who_questions(r1_grouped_activities, r2_grouped_activities, adverbial):
    evaluation_data = []
    num_questions = 0
    while num_questions < NUM_WHO_QUESTIONS_PER_ADVERBIAL:
        for event in EVENTS_MAPPINGS:
            r1_events = r1_grouped_activities[event]
            r2_events = r2_grouped_activities[event]

            result = create_gt_who_one_resident(r1_events, 0, r2_events, 1, adverbial, event)
            if result is not None:
                evaluation_data.append(result)
                num_questions += 1
                if num_questions == NUM_WHO_QUESTIONS_PER_ADVERBIAL:
                    return evaluation_data

            result = create_gt_who_one_resident(r2_events, 1, r1_events, 0, adverbial, event)
            if result is not None:
                evaluation_data.append(result)
                num_questions += 1
                if num_questions == NUM_WHO_QUESTIONS_PER_ADVERBIAL:
                    return evaluation_data

            evaluation_data.append(create_gt_who_zero_resident(r1_events, r2_events, event, adverbial))
            num_questions += 1
            if num_questions == NUM_WHO_QUESTIONS_PER_ADVERBIAL:
                return evaluation_data


def get_what_happened_questions(all_activities, adverbial):
    def load_ttl(file_path="household_events.ttl"):
        g = rdflib.Graph()
        g.parse(file_path, format='ttl')
        return g
    def check_exact_match(interval_1_start, interval_1_end, interval_2_start, interval_2_end):
        return (interval_1_start == interval_2_start) and (interval_1_end == interval_2_end)
    def parse_datetime(date_str):
        try:
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f")
    def extract_all_events(graph):
        events = []
        query = """
        SELECT ?work ?event ?begin ?end WHERE {
            ?work ex:happensAt ?event .  # Get the work and its associated event
            ?event time:hasBeginning ?begin .
            ?event time:hasEnd ?end .
        }
        """
        result = graph.query(query)
        for row in result:
            event_uri = row['work']  # Extract the vent uri
            begin = row['begin'].toPython()  # Get the beginning time
            end = row['end'].toPython()  # Get the end time
            events.append({
                'event_uri': event_uri,
                'begin': parse_datetime(str(begin)),
                'end': parse_datetime(str(end))
            })
        return events

    evaluation_data = []
    if adverbial == "just":
        question = f"What {adverbial} happened?"
    elif adverbial == "long time ago":
        question = f"What happened a {adverbial}?"
    else:
        question = f"What happened {adverbial}?"

    graph_events = extract_all_events(load_ttl())
    for _ in range(NUM_WHAT_HAPPENED_QUESTIONS_PER_ADVERBIAL):
        random_seconds = random.randint(0,
                                        (datetime.strptime(dataset_end_date, dateformat)
                                         - datetime.strptime(dataset_start_date, dateformat)).total_seconds())
        ref_date = datetime.strptime(dataset_start_date, dateformat) + timedelta(seconds=random_seconds)
        gt = []
        for act in all_activities:
            act_end_date = datetime.strptime(act["end"], dateformat)
            if act_end_date <= ref_date:
                lower_percentage, higher_percentage = vague_temp.get_minutes_ago(act["activity"], adverbial)
                if higher_percentage == math.inf:
                   higher_percentage = infinity_minutes
                higher_bound = ref_date-timedelta(minutes=lower_percentage)
                lower_bound = ref_date-timedelta(minutes=higher_percentage)
                if lower_bound <= act_end_date <= higher_bound:
                    event_start = parse_datetime(act['begin'])
                    event_end = parse_datetime(act['end'])
                    for event in graph_events:
                        ttl_start = event['begin']
                        ttl_end = event['end']
                        if check_exact_match(ttl_start, ttl_end, event_start, event_end):
                            if event["event_uri"] not in gt:
                                gt.append(event["event_uri"])
        data_dict = {
            'ref_date': ref_date.strftime(dateformat),
            'question': question,
            'gt': gt
        }
        evaluation_data.append(data_dict)
    return evaluation_data


def save_to_json(filename, data):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def main():
    global NUM_WHAT_QUESTIONS_PER_ADVERBIAL, NUM_WHO_QUESTIONS_PER_ADVERBIAL, NUM_WHAT_HAPPENED_QUESTIONS_PER_ADVERBIAL, NUM_DID_QUESTIONS_PER_EVENT

    args = parse_args()
    NUM_WHAT_QUESTIONS_PER_ADVERBIAL = args.num_what
    NUM_WHO_QUESTIONS_PER_ADVERBIAL = args.num_who
    NUM_WHAT_HAPPENED_QUESTIONS_PER_ADVERBIAL = args.num_what_happened
    NUM_DID_QUESTIONS_PER_EVENT = args.num_did

    eval_did, eval_what, eval_who, eval_what_happened = [], [], [], []
    r1_activities, r2_activities, all_activities = [], [], []

    for resident, name in zip(RESIDENTS, NEW_RESIDENTS_NAME):
        grouped_acts, acts = get_activities_resident(resident=resident)
        all_activities += acts

        if resident == "R1":
            r1_activities = grouped_acts
        else:
            r2_activities = grouped_acts

        for adv in ADVERBIALS:
            eval_did.extend(get_did_questions(grouped_acts, adv, name))
            eval_what.extend(get_what_questions(acts, adv, name))

    for adv in ADVERBIALS:
        eval_who.extend(get_who_questions(r1_activities, r2_activities, adv))
        eval_what_happened.extend(get_what_happened_questions(all_activities, adv))

    save_to_json(f'./evaluation_data/evaluation_data_did.json', eval_did)
    save_to_json(f'./evaluation_data/evaluation_data_what.json', eval_what)
    save_to_json(f'./evaluation_data/evaluation_data_who.json', eval_who)
    save_to_json(f'./evaluation_data/evaluation_data_what_happened.json', eval_what_happened)

    print(f"Data saved to './evaluation_data/evaluation_data_?.json'")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate evaluation data based on household activities.")
    parser.add_argument('--num_what', type=int, default=100, help='Number of WHAT questions per adverbial')
    parser.add_argument('--num_who', type=int, default=200, help='Number of WHO questions per adverbial')
    parser.add_argument('--num_what_happened', type=int, default=100, help='Number of WHAT HAPPENED questions per adverbial')
    parser.add_argument('--num_did', type=int, default=5, help='Number of DID questions per event')
    return parser.parse_args()

if __name__ == "__main__":
    main()
