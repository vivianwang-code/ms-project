#fuzzy logic

def user_habit_score(usage_probability, usage_stability, time_of_week):
    user_habit_score = usage_probability + usage_stability + time_of_week
    return user_habit_score

def deveice_activity_score(standby_duration, time_since_active):
    device_activity_score = standby_duration + time_since_active
    return device_activity_score

def context_score(co2, motion):
    context_score = co2 + motion
    return context_score

def control_confidence_score(is_sleep_hour, is_peak_hour, usage_stability, standby_duration):

    duration_threshold = 30
    stability_threshold = 0.5

    if is_sleep_hour == True:
        control_confidence += 0.2

    if is_peak_hour == True:
        control_confidence -= 0.1

    if standby_duration > duration_threshold:
        control_confidence += 0.1
    else:
        control_confidence -= 0.1

    if usage_stability > stability_threshold:
        control_confidence += usage_stability *0.2



def fuzzy_rule(user_habit, device_activity, context_score):
    if user_habit == "low":
        if device_activity == "low":
            if context_score == "low":
                return "OFF"
            elif context_score == "medium":
                return "OFF"
            elif context_score == "high":
                return "DELAY"
        if device_activity == "medium":
            if context_score == "low":
                return "DELAY"
            elif context_score == "medium":
                return "DELAY"
            elif context_score == "high":
                return "DELAY"
        if device_activity == "high":
            if context_score == "low":
                return "DELAY"
            elif context_score == "medium":
                return "DELAY"
            elif context_score == "high":
                return "ON"
    elif user_habit == "medium":
        if device_activity == "low":
            if context_score == "low":
                return "DELAY"
            elif context_score == "medium":
                return "DELAY"
            elif context_score == "high":
                return "NOTIFY"
        if device_activity == "medium":
            if context_score == "low":
                return "NOTIFY"
            elif context_score == "medium":
                return "NOTIFY"
            elif context_score == "high":
                return "NOTIFY"
        if device_activity == "high":
            if context_score == "low":
                return "NOTIFY"
            elif context_score == "medium":
                return "NOTIFY"
            elif context_score == "high":
                return "ON"
    elif user_habit == "high":
        if device_activity == "low":
            if context_score == "low":
                return "DELAY"
            elif context_score == "medium":
                return "DELAY"
            elif context_score == "high":
                return "NOTIFY"
        if device_activity == "medium":
            if context_score == "low":
                return "NOTIFY"
            elif context_score == "medium":
                return "NOTIFY"
            elif context_score == "high":
                return "NOTIFY"
        if device_activity == "high":
            if context_score == "low":
                return "ON"
            elif context_score == "medium":
                return "ON"
            elif context_score == "high":
                return "ON"
            
def integration(fuzzy_score, confidence_score):
    if fuzzy_score == "OFF":
        if confidence_score == "low":
            return "NOTIFY"
        elif confidence_score == "medium":
            return "DELAY"
        elif confidence_score == "high":
            return "OFF"
    elif fuzzy_score == "DELAY":
        if confidence_score == "low":
            return "NOTIFY"
        elif confidence_score == "medium":
            return "DELAY"
        elif confidence_score == "high":
            return "DELAY"
    elif fuzzy_score == "NOTIFY":
        if confidence_score == "low":
            return "ON"
        elif confidence_score == "medium":
            return "NOTIFY"
        elif confidence_score == "high":
            return "NOTIFY"
            
if __name__ == "__main__":
    # Example usage
    user_habit = "medium"
    device_activity = "high"
    context_score = "low"
    
    fuzzy_action = fuzzy_rule(user_habit, device_activity, context_score)
    print(f"Recommended action: {action}")

    user_habit_score = user_habit_score(usage_probability, usage_stability, time_of_week)

    final_action = integration(fuzzy_action, control_confidence_score)
