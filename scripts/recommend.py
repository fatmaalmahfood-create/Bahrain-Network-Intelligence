import numpy as np
import pandas as pd
import os
import sys
import json
import math
import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_FEATURES_CSV = os.path.join(_ROOT, 'data', 'processed', 'features_engineered.csv')
_TOWERS_CSV   = os.path.join(_ROOT, 'data', 'bahrain_towers.csv')

_W_PERF     = 0.35
_W_DEMAND   = 0.20
_W_SCARCITY = 0.35
_W_GROWTH   = 0.10


def _load_data():
    if not os.path.exists(_FEATURES_CSV):
        raise FileNotFoundError(f"\n Cannot find: {_FEATURES_CSV}\n")
    features = pd.read_csv(_FEATURES_CSV)
    towers   = pd.read_csv(_TOWERS_CSV)
    return features, towers


def _type_label(code: str) -> str:
    mapping = {
        'PPL':  'Populated Place',
        'PPLX': 'Section of Populated Place',
        'HTL':  'Hotel / Tourism',
        'PT':   'Point / landform',
        'ISL':  'Island',
        'ISLX': 'Section of Island',
        'ISLS': 'Islands',
        'PRT':  'Port',
        'TOWR': 'Tower'
    }
    return mapping.get(str(code).strip(), code)


def _haversine(lat1, lon1, lat2, lon2):
    R    = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) ** 2 +
            math.cos(math.radians(lat1)) *
            math.cos(math.radians(lat2)) *
            math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def get_top_recommendations(top_n: int = 5) -> list[dict]:
    features, towers = _load_data()

    latest_year = features['year'].max()
    latest      = features[features['year'] == latest_year].copy()

    agg = {
        'latitude':          'mean',
        'longitude':         'mean',
        'avg_d_mbps':        'mean',
        'avg_u_mbps':        'mean',
        'avg_lat_ms':        'mean',
        'tests':             'sum',
        'demand_growth_pct': 'mean',
        'typeOfArea':        'first',
        'city':              'first',
        'region':            'first',
    }
    for _col in ['nearest_tower_distance_km', 'nearest_tower_km',
                 'tower_count_1km', 'towers_1km',
                 'tower_count_2km',
                 'tower_count_5km', 'towers_3km']:
        if _col in latest.columns:
            agg[_col] = 'mean'

    area_df = latest.groupby('area').agg(agg).reset_index()

    def _coalesce(df, canonical, *alts):
        if canonical not in df.columns:
            for a in alts:
                if a in df.columns:
                    df[canonical] = df[a]
                    return
            df[canonical] = 0.0

    _coalesce(area_df, 'nearest_tower_distance_km', 'nearest_tower_km')
    _coalesce(area_df, 'tower_count_1km', 'towers_1km')
    _coalesce(area_df, 'tower_count_2km')
    _coalesce(area_df, 'tower_count_5km', 'towers_3km')

    def minmax(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)

    area_df['perf_weakness_score'] = 1.0 - minmax(area_df['avg_d_mbps'])
    area_df['demand_score']        = minmax(area_df['tests'])
    area_df['scarcity_score']      = 1.0 - minmax(area_df['tower_count_5km'])
    area_df['growth_score']        = minmax(area_df['demand_growth_pct'].clip(0, None))

    area_df['priority_score'] = (
        _W_PERF     * area_df['perf_weakness_score'] +
        _W_DEMAND   * area_df['demand_score']        +
        _W_SCARCITY * area_df['scarcity_score']      +
        _W_GROWTH   * area_df['growth_score']
    ) * 100

    top = area_df.nlargest(top_n, 'priority_score').reset_index(drop=True)

    results = []
    for rank, row in top.iterrows():
        type_label = _type_label(row['typeOfArea'])

        if row['perf_weakness_score'] > 0.7:
            weakness = 'significantly underperforming network speeds'
        elif row['perf_weakness_score'] > 0.4:
            weakness = 'below-average network speeds'
        else:
            weakness = 'moderate network speeds'

        if row['demand_score'] > 0.7:
            demand_txt = 'very high user demand'
        elif row['demand_score'] > 0.4:
            demand_txt = 'moderate-to-high user demand'
        else:
            demand_txt = 'growing user demand'

        summary = (
            f"{row['area']} ({type_label}) in {row['city']} shows {weakness} "
            f"with {demand_txt} ({int(row['tests']):,} tests recorded). "
            f"Average download is {row['avg_d_mbps']:.1f} Mbps with "
            f"{row['tower_count_5km']:.1f} towers within 5 km. "
            f"A new tower here would serve the most underserved users."
        )

        results.append({
            'rank':                      rank + 1,
            'area':                      row['area'],
            'city':                      row['city'],
            'region':                    row['region'],
            'typeOfArea':                row['typeOfArea'],
            'typeOfArea_label':          type_label,
            'latitude':                  round(row['latitude'], 6),
            'longitude':                 round(row['longitude'], 6),
            'avg_d_mbps':                round(row['avg_d_mbps'], 2),
            'avg_u_mbps':                round(row['avg_u_mbps'], 2),
            'avg_lat_ms':                round(row['avg_lat_ms'], 2),
            'tests':                     int(row['tests']),
            'nearest_tower_distance_km': round(row['nearest_tower_distance_km'], 3),
            'tower_count_1km':           round(row['tower_count_1km'], 1),
            'tower_count_2km':           round(row['tower_count_2km'], 1),
            'tower_count_5km':           round(row['tower_count_5km'], 1),
            'demand_growth_pct':         round(row['demand_growth_pct'] * 100, 2),
            'perf_weakness_score':       round(row['perf_weakness_score'], 4),
            'demand_score':              round(row['demand_score'], 4),
            'scarcity_score':            round(row['scarcity_score'], 4),
            'growth_score':              round(row['growth_score'], 4),
            'priority_score':            round(row['priority_score'], 2),
            'recommendation_summary':    summary,
        })

    return results


def _build_area_features(area_name: str, lat: float, lon: float,
                          year: int = 2026, quarter: int = 2) -> dict:
    features, towers = _load_data()

    area_rows = features[features['area'] == area_name]
    if area_rows.empty:
        return None

    encoders_path = os.path.join(_HERE, '..', 'assets', 'encoders.json')
    with open(encoders_path) as f:
        encoders = json.load(f)

    latest     = area_rows.sort_values('year').iloc[-1]
    region     = latest.get('region',     'Capital Governorate')
    typeOfArea = latest.get('typeOfArea', 'PPL')
    city       = latest.get('city',       '')

    region_enc     = encoders['region'].get(region, 0)
    typeOfArea_enc = encoders['typeOfArea'].get(typeOfArea, 0)
    city_enc       = encoders['city'].get(city, 0)

    tower_df = towers.dropna(subset=['Latitude', 'Longitude'])
    if len(tower_df) > 0:
        dists = tower_df.apply(
            lambda r: _haversine(lat, lon, r['Latitude'], r['Longitude']),
            axis=1
        )
        dist = round(float(dists.min()), 3)
        tc1  = float((dists <= 1).sum())
        tc2  = float((dists <= 2).sum())
        tc5  = float((dists <= 5).sum())
    else:
        dist = float(latest.get('nearest_tower_distance_km', 0.5))
        tc1  = float(latest.get('tower_count_1km', 2))
        tc2  = float(latest.get('tower_count_2km', 5))
        tc5  = float(latest.get('tower_count_5km', 10))

    dem = float(latest.get('digital_elevation_model', 5))

    # Quarterly aggregation — matches training pipeline
    area_q = area_rows.groupby(['year', 'quarter']).agg(
        avg_d_mbps=('avg_d_mbps', 'mean'),
        avg_u_mbps=('avg_u_mbps', 'mean'),
        avg_lat_ms=('avg_lat_ms', 'mean'),
        tests=('tests', 'sum'),
    )
    area_median_d   = float(area_q['avg_d_mbps'].median())
    area_median_u   = float(area_q['avg_u_mbps'].median())
    area_median_lat = float(area_q['avg_lat_ms'].median())
    area_test_count = float(area_q['tests'].sum())
    rolling_d       = float(area_q['avg_d_mbps'].mean())

    area_by_year = area_rows.groupby('year')['tests'].sum().sort_index()
    if len(area_by_year) >= 2:
        prev   = area_by_year.iloc[-2]
        curr   = area_by_year.iloc[-1]
        growth = ((curr - prev) / prev) if prev > 0 else 0.0
    else:
        growth = 0.0

    year_min            = int(features['year'].min())
    year_trend          = year - year_min
    tower_density_ratio = tc1 / (tc5 + 1)
    distance_x_density  = dist * tower_density_ratio

    return {
        'latitude':                  lat,
        'longitude':                 lon,
        'nearest_tower_distance_km': dist,
        'tower_count_1km':           tc1,
        'tower_count_2km':           tc2,
        'tower_count_5km':           tc5,
        'tower_density_ratio':       tower_density_ratio,
        'digital_elevation_model':   dem,
        'region_enc':                region_enc,
        'typeOfArea_enc':            typeOfArea_enc,
        'city_enc':                  city_enc,
        'demand_growth_pct':         growth,
        'year_trend':                year_trend,
        'quarter':                   quarter,
        'area_rolling_d_mbps':       rolling_d,
        'tests':                     int(area_q['tests'].mean()),
        'area_median_d':             area_median_d,
        'area_median_u':             area_median_u,
        'area_median_lat':           area_median_lat,
        'area_test_count':           area_test_count,
        'distance_x_density':        distance_x_density,
    }


def simulate_tower_impact(area_name: str, lat: float, lon: float) -> dict:
    sys.path.insert(0, _HERE)
    from predict import predict_speeds

    feat = _build_area_features(area_name, lat, lon)
    if feat is None:
        return {'error': f'No data found for area: {area_name}'}

    current_pred = predict_speeds(feat)

    future_feat = feat.copy()
    future_feat['nearest_tower_distance_km'] = 0.1
    future_feat['tower_count_1km']  += 1
    future_feat['tower_count_2km']  += 1
    future_feat['tower_count_5km']  += 1
    future_feat['tower_density_ratio'] = (
        future_feat['tower_count_1km'] / (future_feat['tower_count_5km'] + 1)
    )
    future_feat['distance_x_density'] = (
        future_feat['nearest_tower_distance_km'] *
        future_feat['tower_density_ratio']
    )

    future_pred = predict_speeds(future_feat)

    return {
        'area':                area_name,
        'lat':                 lat,
        'lon':                 lon,
        'current_download':    round(current_pred['avg_d_mbps'], 2),
        'current_upload':      round(current_pred['avg_u_mbps'], 2),
        'current_latency':     round(current_pred['avg_lat_ms'], 2),
        'future_download':     round(future_pred['avg_d_mbps'], 2),
        'future_upload':       round(future_pred['avg_u_mbps'], 2),
        'future_latency':      round(future_pred['avg_lat_ms'], 2),
        'download_gain':       round(future_pred['avg_d_mbps'] - current_pred['avg_d_mbps'], 2),
        'upload_gain':         round(future_pred['avg_u_mbps'] - current_pred['avg_u_mbps'], 2),
        'latency_improvement': round(current_pred['avg_lat_ms'] - future_pred['avg_lat_ms'], 2),
        'download_pct_gain':   round(
            ((future_pred['avg_d_mbps'] - current_pred['avg_d_mbps']) /
             max(current_pred['avg_d_mbps'], 1)) * 100, 1
        ),
    }


def run_tower_impact_agent(recommendations: list) -> str:
    simulations_text = ""
    valid_results    = []

    for rec in recommendations:
        area_name = rec['area']
        lat       = rec['latitude']
        lon       = rec['longitude']

        result = simulate_tower_impact(area_name, lat, lon)

        if 'error' in result:
            simulations_text += f"\n{area_name}: Data not available\n"
            continue

        valid_results.append(result)
        simulations_text += (
            f"\n--- {area_name} ---\n"
            f"  Location: ({lat}, {lon})\n"
            f"  CURRENT:  Download={result['current_download']} Mbps | "
            f"Upload={result['current_upload']} Mbps | "
            f"Latency={result['current_latency']} ms\n"
            f"  AFTER NEW TOWER AT THIS LOCATION:\n"
            f"            Download={result['future_download']} Mbps | "
            f"Upload={result['future_upload']} Mbps | "
            f"Latency={result['future_latency']} ms\n"
            f"  IMPROVEMENT: +{result['download_gain']} Mbps download "
            f"(+{result['download_pct_gain']}%) | "
            f"+{result['upload_gain']} Mbps upload | "
            f"-{result['latency_improvement']} ms latency\n"
        )

    prompt = (
        "You are an expert network infrastructure planning agent for Bahrain.\n\n"
        "The following simulations show predicted network speeds at specific "
        "GPS coordinates in Bahrain — first with the current tower infrastructure, "
        "then after placing a NEW cell tower at that exact location.\n"
        "The predictions come from a trained Gradient Boosting ML model.\n\n"
        f"=== TOWER IMPACT SIMULATIONS ===\n{simulations_text}\n"
        "=== YOUR TASK ===\n"
        "Based on the predicted improvements:\n"
        "1. Rank the TOP 3 locations where building a new tower would have "
        "the greatest positive impact on network performance\n"
        "2. For each location explain:\n"
        "   - What the current network situation is at those coordinates\n"
        "   - What improvement the new tower would bring\n"
        "   - Why this matters for the users in that area\n"
        "3. Give a final conclusion on which location should be built FIRST\n\n"
        "Use specific numbers from the simulations in your reasoning.\n"
        "Format clearly with Top 1, Top 2, Top 3 sections."
    )

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model":    "llama3.2",
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False
            },
            timeout=300
        )
        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            return f"Ollama error {response.status_code}: {response.text}"

    except requests.exceptions.ConnectionError:
        return "❌ Could not connect to Ollama. Run: ollama serve"
    except requests.exceptions.Timeout:
        return "❌ Ollama took too long. Try restarting Ollama."
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"




def query_area_speeds(area_name: str) -> str:
    features, _ = _load_data()
    area = features[features['area'] == area_name]
    if area.empty:
        return f"No data found for area: {area_name}"
    return (
        f"{area_name}: "
        f"Avg Download={round(float(area['avg_d_mbps'].mean()), 2)} Mbps, "
        f"Avg Upload={round(float(area['avg_u_mbps'].mean()), 2)} Mbps, "
        f"Avg Latency={round(float(area['avg_lat_ms'].mean()), 2)} ms, "
        f"Total Tests={int(area['tests'].sum())}, "
        f"Towers within 5km={round(float(area['tower_count_5km'].mean()), 1)}"
    )


def query_worst_areas(metric: str = 'download', top_n: int = 5) -> str:
    features, _ = _load_data()
    col_map = {
        'download': 'avg_d_mbps',
        'upload':   'avg_u_mbps',
        'latency':  'avg_lat_ms'
    }
    col       = col_map.get(metric.lower(), 'avg_d_mbps')
    ascending = metric.lower() != 'latency'
    area_stats = (
        features.groupby('area')[col]
        .mean()
        .sort_values(ascending=ascending)
        .head(top_n)
    )
    result = f"Worst {top_n} areas by {metric}:\n"
    for area, val in area_stats.items():
        result += f"  - {area}: {round(val, 2)} {'Mbps' if metric != 'latency' else 'ms'}\n"
    return result


def query_speed_trend(area_name: str) -> str:
    features, _ = _load_data()
    area = features[features['area'] == area_name]
    if area.empty:
        return f"No data found for area: {area_name}"
    yearly = area.groupby('year')['avg_d_mbps'].mean().sort_index()
    result = f"Download speed trend for {area_name}:\n"
    for year, speed in yearly.items():
        result += f"  {int(year)}: {round(float(speed), 2)} Mbps\n"
    return result


def query_tower_info(area_name: str) -> str:
    features, _ = _load_data()
    area = features[features['area'] == area_name]
    if area.empty:
        return f"No data found for area: {area_name}"
    latest = area.sort_values('year').iloc[-1]
    return (
        f"{area_name} tower info: "
        f"Nearest tower={round(float(latest.get('nearest_tower_distance_km', 0)), 3)} km, "
        f"Towers within 1km={round(float(latest.get('tower_count_1km', 0)), 1)}, "
        f"Towers within 2km={round(float(latest.get('tower_count_2km', 0)), 1)}, "
        f"Towers within 5km={round(float(latest.get('tower_count_5km', 0)), 1)}"
    )


def compare_two_areas(area_1: str, area_2: str) -> str:
    features, _ = _load_data()
    a1 = features[features['area'] == area_1]
    a2 = features[features['area'] == area_2]
    if a1.empty:
        return f"No data found for: {area_1}"
    if a2.empty:
        return f"No data found for: {area_2}"
    return (
        f"{area_1}: Download={round(float(a1['avg_d_mbps'].mean()), 2)} Mbps, "
        f"Upload={round(float(a1['avg_u_mbps'].mean()), 2)} Mbps, "
        f"Latency={round(float(a1['avg_lat_ms'].mean()), 2)} ms, "
        f"Towers 5km={round(float(a1['tower_count_5km'].mean()), 1)}\n"
        f"{area_2}: Download={round(float(a2['avg_d_mbps'].mean()), 2)} Mbps, "
        f"Upload={round(float(a2['avg_u_mbps'].mean()), 2)} Mbps, "
        f"Latency={round(float(a2['avg_lat_ms'].mean()), 2)} ms, "
        f"Towers 5km={round(float(a2['tower_count_5km'].mean()), 1)}"
    )


def run_chat_agent(user_question: str,
                   chat_history: list,
                   simulation_results: list) -> str:
    """
    Answers user questions about Bahrain network performance.
    Has access to the simulation results already computed,
    plus tools to query the real dataset.
    """

    sim_context = ""
    for r in simulation_results:
        if 'error' not in r:
            sim_context += (
                f"\n{r['area']} ({r['lat']}, {r['lon']}): "
                f"Current Download={r['current_download']} Mbps → "
                f"After Tower={r['future_download']} Mbps "
                f"(+{r['download_gain']} Mbps, +{r['download_pct_gain']}%) | "
                f"Current Latency={r['current_latency']} ms → "
                f"After Tower={r['future_latency']} ms\n"
            )

    fetched_data = ""
    question_lower = user_question.lower()

    features, _ = _load_data()
    all_areas = features['area'].dropna().unique()
    mentioned_areas = [a for a in all_areas if a.lower() in question_lower]

    if mentioned_areas:
        for area in mentioned_areas:
            fetched_data += query_area_speeds(area) + "\n"
            fetched_data += query_tower_info(area) + "\n"
            fetched_data += query_speed_trend(area) + "\n"

    if len(mentioned_areas) == 2:
        fetched_data += compare_two_areas(mentioned_areas[0], mentioned_areas[1]) + "\n"

    if any(w in question_lower for w in ['worst', 'bad', 'slow', 'lowest', 'weakest']):
        if 'upload' in question_lower:
            fetched_data += query_worst_areas('upload') + "\n"
        elif 'latency' in question_lower or 'lag' in question_lower:
            fetched_data += query_worst_areas('latency') + "\n"
        else:
            fetched_data += query_worst_areas('download') + "\n"

    if any(w in question_lower for w in ['impact', 'improve', 'new tower', 'add tower', 'build']):
        for area in mentioned_areas:
            result = simulate_tower_impact(
                area,
                float(features[features['area'] == area]['latitude'].mean()),
                float(features[features['area'] == area]['longitude'].mean())
            )
            if 'error' not in result:
                fetched_data += (
                    f"Tower impact for {area}: "
                    f"Download {result['current_download']} → "
                    f"{result['future_download']} Mbps "
                    f"(+{result['download_pct_gain']}%)\n"
                )

    # Build conversation history
    history_text = ""
    for msg in chat_history[-6:]:
        role = "User" if msg['role'] == 'user' else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = (
        "You are an intelligent network analysis assistant for Bahrain's telecom infrastructure.\n"
        "You have access to real speed test data, tower information, and ML-predicted "
        "tower impact simulations across Bahrain.\n\n"
        f"=== TOWER IMPACT SIMULATION RESULTS ===\n{sim_context}\n"
        f"=== ADDITIONAL DATA FETCHED ===\n{fetched_data if fetched_data else 'None needed'}\n"
        f"=== CONVERSATION HISTORY ===\n{history_text if history_text else 'None yet'}\n"
        f"=== USER QUESTION ===\n{user_question}\n\n"
        "Instructions:\n"
        "- Answer using the simulation results and fetched data above\n"
        "- Be specific with numbers\n"
        "- If asked why certain results appeared, explain based on tower counts, "
        "distances, and historical speeds\n"
        "- If asked about improvement, refer to the before/after simulation numbers\n"
        "- If asked to compare areas, use both datasets side by side\n"
        "- Keep answers clear and concise"
    )

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model":    "llama3.2",
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False
            },
            timeout=300
        )
        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            return f"Error {response.status_code}: {response.text}"

    except requests.exceptions.ConnectionError:
        return "❌ Could not connect to Ollama. Run: ollama serve"
    except requests.exceptions.Timeout:
        return "❌ Response took too long. Try a simpler question."
    except Exception as e:
        return f"❌ Error: {str(e)}"

        


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    top_recs = get_top_recommendations(top_n=5)
    for r in top_recs:
        print(f"\nTop {r['rank']}: {r['area']} — Priority Score: {r['priority_score']}")
        print(f"  {r['recommendation_summary']}")