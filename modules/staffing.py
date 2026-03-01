import math
import pandas as pd

def erlang_c_probability(intensity, agents):
    if agents <= intensity: return 1.0
    utilization = intensity / agents
    sum_term = sum([(intensity**i) / math.factorial(i) for i in range(agents)])
    queue_term = (intensity**agents / (math.factorial(agents) * (1 - utilization)))
    return queue_term / (sum_term + queue_term)

def calculate_required_agents(calls, aht, interval_min=30, target_sl=0.8, target_time=20):
    if calls <= 0: return 0
    intensity = (calls * aht) / (interval_min * 60)
    agents = math.ceil(intensity) + 1
    while True:
        if agents > 500: break
        utilization = intensity / agents
        if utilization >= 1:
            agents += 1
            continue
        prob_wait = erlang_c_probability(intensity, agents)
        service_level = 1 - (prob_wait * math.exp(-(agents - intensity) * (target_time / aht)))
        if service_level >= target_sl: break
        agents += 1
    return agents

def get_staffing_requirements(df_forecast, aht, target_sl, shrinkage):
    # Forzamos los cálculos para evitar el TypeError
    df_forecast['agentes_netos'] = df_forecast['yhat'].apply(
        lambda x: calculate_required_agents(x, aht, target_sl=target_sl)
    )
    df_forecast['agentes_nominales'] = (df_forecast['agentes_netos'] / (1 - shrinkage)).apply(math.ceil)
    return df_forecast
