import pickle
import math
import inspect
from scipy.special import erfinv
import numpy as np
import os


upper_percentage = 1.0
lower_percentage = 0.6

save_file_name_event_duration_params = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimized_parameters_overAllVotes_event_duration")
save_file_name_adverbial_params = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimized_parameters_overAllVotes_adverbials")
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


def gauss_inverse(y, mean, std):
    return mean + std * np.sqrt(-2 * np.log(y))

def inverse_event_specific_function(y, std):
    clipped_y = max(-1, min(1, 2 * y - 1))
    return math.sqrt(2) * std * erfinv(clipped_y)

def safe_round(value):
    return round(value) if math.isfinite(value) else value

def get_minutes_ago(event, adverbial):
    adverbial = adverbial.lower().strip().replace("_", " ")
    event = event.lower().strip().replace(" ", "_")

    if "brush" in event and "teeth" in event:
        event_duration_idx = events_durations.index("minutes")
    elif "bath" in event:
        event_duration_idx = events_durations.index("minutes")
    elif "housekeeping" in event:
        event_duration_idx = events_durations.index("hours")
    elif "hygiene" in event:
        event_duration_idx = events_durations.index("minutes")
    elif "leave" in event and "home" in event:
        event_duration_idx = events_durations.index("minutes")
    elif "return" in event and "home" in event:
        event_duration_idx = events_durations.index("minutes")
    elif "meal" in event and "prepar" in event:#covering prepare and preparing
        event_duration_idx = events_durations.index("minutes")
    elif "sleep" in event:
        event_duration_idx = events_durations.index("hours")
    elif "toilet" in event:
        event_duration_idx = events_durations.index("minutes")
    elif "watch" in event and "tv" in event:
        event_duration_idx = events_durations.index("hours")
    elif "work" in event:
        event_duration_idx = events_durations.index("hours")
    elif "out" in event:
        event_duration_idx = events_durations.index("hours")
    elif "eat" in event:
        event_duration_idx = events_durations.index("minutes")
    else:
        raise ValueError(f"Invalid Event: {event}")

    if adverbial not in all_adverbials:
        raise ValueError(f"Invalid adverbial: {adverbial}")

    with open(save_file_name_event_duration_params + '.pkl', 'rb') as f:
        optimized_event_duration_params = pickle.load(f)
        num_params = len(inspect.signature(inverse_event_specific_function).parameters) - 1
        event_params = optimized_event_duration_params[event_duration_idx:event_duration_idx + num_params]
    with open(save_file_name_adverbial_params + '.pkl', 'rb') as f:
        optimized_adverbial_params = pickle.load(f)
        idx = all_adverbials.index(adverbial)
        adverbial_params = [
            optimized_adverbial_params[idx],
            optimized_adverbial_params[idx + len(all_adverbials)]
        ]

    upper_raw = inverse_event_specific_function(gauss_inverse(upper_percentage, *adverbial_params), *event_params)
    lower_raw = inverse_event_specific_function(gauss_inverse(lower_percentage, *adverbial_params), *event_params)

    upper = max(0, safe_round(upper_raw))
    lower = safe_round(lower_raw)

    return upper, lower