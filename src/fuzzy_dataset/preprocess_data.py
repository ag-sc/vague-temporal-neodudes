#!/usr/bin/env python3

from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import json
import sys
import argparse

# Constants
ACTIVITIES_TO_MISS = ["Wandering_in_room"]
dateformat = '%Y-%m-%dT%H:%M:%S.%f'

class CleanCode:
    def __init__(self, activities):
        self.activities = activities

    def merge_activities(self):
        merged_activities = []
        current = self.activities[0]
        for next_activity in self.activities[1:]:
            if next_activity['activity'] == current['activity']:
                current['end'] = next_activity['end']
            elif current['end'] is None:
                current['end'] = next_activity['begin']
            else:
                merged_activities.append(current)
                current = next_activity
        merged_activities.append(current)
        self.activities = merged_activities

    def remove_short_activities(self, activities_to_remove=['Work', 'Sleep'], min_duration_seconds=600):
        filtered_activities = []
        for activity in self.activities:
            if activity['begin'] and activity['end']:
                begin_time = datetime.fromisoformat(activity['begin'])
                end_time = datetime.fromisoformat(activity['end'])
                duration = (end_time - begin_time).total_seconds()

                if activity['activity'] in activities_to_remove and duration < min_duration_seconds:
                    continue
            filtered_activities.append(activity)
        self.activities = filtered_activities

    def merge_away_from_home(self):
        merged_activities = []
        skip_next = False
        for i in range(len(self.activities) - 1):
            if skip_next:
                skip_next = False
                continue
            current = self.activities[i]
            next_activity = self.activities[i + 1]
            if current['activity'] == 'Leave_Home':
                merged_activities.append({
                    'activity': 'Out_of_Home',
                    'begin': current['begin'],
                    'end': next_activity['end']
                })
                if next_activity['activity'] == 'Enter_Home':
                    skip_next = True
            elif current['activity'] == 'Enter_Home' and not next_activity['activity'] == 'Leave_Home':
                end_time = datetime.fromisoformat(next_activity['begin']) - timedelta(seconds=100)
                merged_activities.append({
                    'activity': 'Out_of_Home',
                    'begin': current['begin'],
                    'end': end_time.strftime(dateformat)
                })
            else:
                merged_activities.append(current)
        if not skip_next:
            merged_activities.append(self.activities[-1])
        self.activities = merged_activities


def get_activities(file_path):
    activities = {'R1': [], 'R2': []}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 6:
                date, time, _, _, resident_activity, begin_end = parts[:6]
                resident, activity = resident_activity.split('_', 1)
                if activity == "Sleeping_Not_in_Bed":
                    activity = "Sleep"
                timestamp = f"{date}T{time}"
                if activity not in ACTIVITIES_TO_MISS:
                    if begin_end == 'begin':
                        activities[resident].append({'activity': activity, 'begin': timestamp, 'end': None})
                    elif begin_end == 'end':
                        for act in activities[resident]:
                            if act['activity'] == activity and act['end'] is None:
                                act['end'] = timestamp
                                break
    return activities


def save_activities_as_json(activities):
    for resident, acts in activities.items():
        file_path = f"{resident}_activities.json"
        with open(file_path, 'w') as file:
            json.dump(acts, file, indent=4)
        print(f"Activities saved to {file_path}.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess the defined dataset.")
    parser.add_argument('--filepath',
                        default="twor.2010/data",
                        help="Path to the activity data file (default: twor.2010/data)"
    )
    args = parser.parse_args()

    activities = get_activities(args.filepath)

    for resident, acts in activities.items():
        cleaner = CleanCode(acts)
        cleaner.merge_activities()
        cleaner.remove_short_activities()
        cleaner.merge_away_from_home()
        cleaner.merge_activities()
        activities[resident] = cleaner.activities

    save_activities_as_json(activities)


if __name__ == "__main__":
    main()
