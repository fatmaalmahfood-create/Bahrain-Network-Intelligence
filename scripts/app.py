import os, sys, json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import folium
from folium import Element
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Polygon
import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, '..'))

from predict import predict_speeds
from recommend import get_top_recommendations

st.set_page_config(
    page_title='Bahrain Network Intelligence',
    page_icon='📡',
    layout='wide',
)

NAVY      = '#0E3788'
NAVY_DARK = '#0A2864'
NAVY_MID  = '#1142A3'
RED       = '#B8102C'
RED_LIGHT = '#CC1231'
GOLD      = '#FFC000'
GOLD_SOFT = '#FFC000'
GREY_BG   = '#F2F2F2'
GREY_CARD = '#FFFFFF'
GREY_BORDER = '#DDE1EC'
GREY_TEXT = '#6B7280'
GREY_MUTED= '#9CA3AF'

#CSS
st.markdown(f"""
<style>

  html, body, [class*="css"] {{
      background-color: {GREY_BG};
  }}
  .stApp {{ background-color: {GREY_BG}; }}

  

  .top-nav {{
      background-color:{NAVY_DARK};
      padding: 35px 32px;
      margin: -1rem -2rem 2rem -2rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 3px solid {GOLD};
      box-shadow: 0 4px 20px rgba(8,26,61,0.35);
  }}
  .nav-brand {{
      display: flex;
      align-items: center;
      gap: 14px;
  }}
  .nav-icon {{
      width: 46px; height: 46px;
      background: {GOLD};
      border-radius: 12px;
      display: flex; align-items: center; justify-content: center;
      font-size: 22px;
  }}
  .nav-title {{
      font-size: 2rem;
      font-weight: bold;
      color: #FFFFFF;
      letter-spacing: 0.5px;
      line-height: 1.1;
  }}
  .nav-subtitle {{
      font-size: 0.72rem;
      color: {GREY_MUTED};
      letter-spacing: 1.5px;
      text-transform: uppercase;
      font-weight: 500;
  }}
  .nav-badge {{
      background: rgba(255,192,0,0.15);
      border: 1px solid rgba(255,192,0,0.4);
      color: {GOLD};
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: 1.2px;
      text-transform: uppercase;
      padding: 7px 14px;
      border-radius: 20px;
  }}

  .section-header {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 2rem 0 1.2rem 0;
  }}
  .section-header-bar {{
      width: 4px; height: 26px;
      background:{RED_LIGHT};
      border-radius: 2px;
  }}
  .section-header-text {{
      font-size: 1.2rem;
      font-weight: 700;
      color: {NAVY_DARK};
      letter-spacing: 0.3px;
      text-transform: uppercase;
  }}

  .kpi-card {{
      background: {GREY_CARD};
      border: 1px solid {GREY_BORDER};
      border-radius: 14px;
      padding: 18px 20px;
      position: relative;
      overflow: hidden;
      transition: box-shadow 0.2s;
  }}
  .kpi-card:hover {{ box-shadow: 0 6px 24px rgba(14,42,92,0.12); }}
  .kpi-card::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      border-bottom: 3px solid {GOLD};
  }}

  .kpi-label {{
      font-size: 0.72rem;
      font-weight: 600;
      color: {GREY_MUTED};
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 4px;
  }}
  .kpi-value {{
      font-size: 2rem;
      font-weight: 800;
      line-height: 1;
  }}
  .kpi-unit {{
      font-size: 0.85rem;
      font-weight: 500;
      color: {GREY_TEXT};
      margin-left: 4px;
  }}

  .map-card-title {{
      font-size: 1.1rem;
      font-weight: 700;
      color: {NAVY_DARK};
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 4px;
      display: flex;
      align-items: center;
      gap: 7px;
  }}
  .map-dot {{
      width: 8px; height: 8px;
      border-radius: 50%;
      background: {GOLD};
      display: inline-block;
  }}
  .map-hint {{
      font-size: 0.91rem;
      color: {GREY_MUTED};
      margin-bottom: 10px;
  }}
  [data-testid="stSelectbox"] > div > div {{
    background-color: white !important;
  }}



  .chart-title {{
      font-size: 1.1rem;
      font-weight: 700;
      color: {NAVY_DARK};
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 14px;
      padding-bottom: 10px;
      border-bottom: 1px solid {GREY_BORDER};
  }}

  .predict-header {{
      background:{NAVY};
      border-radius: 16px;
      padding: 22px 28px;
      margin-bottom: 1.5rem;
      border-left: 5px solid {GOLD};
      display: flex;
      align-items: center;
      gap: 16px;
  }}
  .predict-header-icon {{
      font-size: 2.2rem;
  }}
  .predict-header-title {{
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 1.35rem;
      font-weight: 800;
      color: #FFFFFF;
      text-transform: uppercase;
      letter-spacing: 0.5px;
  }}
  .predict-header-sub {{
      font-size: 0.78rem;
      color: rgba(255,255,255,0.55);
      margin-top: 2px;
  }}


  .form-group-label {{
      font-size: 0.88rem;
      font-weight: 700;
      color: {NAVY};
      text-transform: uppercase;
      letter-spacing: 1.2px;
      padding: 6px 12px;
      background: linear-gradient(90deg, rgba(14,42,92,0.07), transparent);
      border-left: 3px solid {GOLD};
      border-radius: 0 4px 4px 0;
      margin: 14px 0 10px 0;
  }}

  [data-testid="stForm"] {{
      border: none !important;
      padding: 0 !important;
  }}
  div[data-testid="stFormSubmitButton"] > button {{
      background: linear-gradient(135deg, {NAVY} 0%, {NAVY_DARK} 100%) !important;
      color: white !important;
      border: none !important;
      border-radius: 10px !important;
      font-size: 1.05rem !important;
      font-weight: 700 !important;
      letter-spacing: 1px !important;
      text-transform: uppercase !important;
      padding: 14px !important;
      border-bottom: 3px solid {GOLD} !important;
      transition: all 0.2s !important;
      margin-top: 8px !important;
  }}
  div[data-testid="stFormSubmitButton"] > button:hover {{
      background: linear-gradient(135deg, {NAVY_MID} 0%, {NAVY} 100%) !important;
      border-bottom-color: {GOLD_SOFT} !important;
      transform: translateY(-1px) !important;
      box-shadow: 0 6px 20px rgba(14,42,92,0.3) !important;
  }}

  .result-card {{
      background: {GREY_CARD};
      border: 1px solid {GREY_BORDER};
      border-radius: 16px;
      padding: 20px;
      text-align: center;
      position: relative;
      overflow: hidden;
  }}
  .result-card::after {{
      content: '';
      position: absolute;
      bottom: 0; left: 0; right: 0;
      height: 3px;
      background: {NAVY};
  }}
  .result-icon {{
      width: 48px; height: 48px;
      border-radius: 14px;
      display: flex; align-items: center; justify-content: center;
      font-size: 22px;
      margin: 0 auto 12px;
  }}
  .result-label {{
      font-size: 0.72rem;
      font-weight: 600;
      color: {GREY_MUTED};
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 6px;
  }}
  .result-value {{
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 2.4rem;
      font-weight: 800;
      color: {NAVY_DARK};
      line-height: 1;
  }}
  .result-unit {{
      font-size: 1rem;
      font-weight: 500;
      color: {GREY_TEXT};
  }}

  .quality-banner {{
      border-radius: 14px;
      padding: 18px 22px;
      display: flex;
      align-items: center;
      gap: 16px;
      margin-top: 16px;
      border: 1px solid;
  }}
  .quality-circle {{
      width: 52px; height: 52px;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 24px; font-weight: 800;
      color: white;
      flex-shrink: 0;
  }}
  .quality-label {{
      font-size: 0.7rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: {GREY_TEXT};
  }}
  .quality-text {{
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 1.25rem;
      font-weight: 700;
      color: {NAVY_DARK};
  }}

  .rec-header {{
      background: linear-gradient(135deg, {RED} 0%, {RED_LIGHT} 100%);
      border-radius: 16px;
      padding: 20px 26px;
      margin-bottom: 1.2rem;
      border-left: 5px solid {GOLD};
      display: flex;
      align-items: center;
      gap: 14px;
  }}
  .rec-header-icon {{ font-size: 2rem; }}
  .rec-header-title {{
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 1.3rem;
      font-weight: 800;
      color: #FFFFFF;
      text-transform: uppercase;
  }}
  .rec-header-sub {{
      font-size: 0.75rem;
      color: rgba(255,255,255,0.6);
      margin-top: 2px;
  }}

  .rec-area-title {{
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 1.6rem;
      font-weight: 800;
      color: {NAVY_DARK};
      line-height: 1.1;
  }}
  .rec-location-tag {{
      display: inline-block;
      background: rgba(14,42,92,0.07);
      border: 1px solid {GREY_BORDER};
      color: {NAVY};
      font-size: 0.72rem;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 20px;
      margin: 2px 3px 2px 0;
      letter-spacing: 0.3px;
  }}
  .rec-coords {{
      font-family: 'Courier New', monospace;
      font-size: 0.78rem;
      color: {GREY_TEXT};
      background: {GREY_BG};
      padding: 4px 10px;
      border-radius: 6px;
      display: inline-block;
      margin-top: 6px;
  }}
  .rec-summary {{
      font-size: 0.88rem;
      color: {GREY_TEXT};
      line-height: 1.55;
      font-style: italic;
      border-left: 3px solid {GOLD};
      padding-left: 12px;
      margin: 12px 0;
  }}

  .metric-mini-card {{
      background: {GREY_BG};
      border: 1px solid {GREY_BORDER};
      border-radius: 10px;
      padding: 12px;
      text-align: center;
  }}
  .metric-mini-label {{
      font-size: 0.66rem;
      font-weight: 600;
      color: {GREY_MUTED};
      text-transform: uppercase;
      letter-spacing: 0.8px;
      margin-bottom: 4px;
  }}
  .metric-mini-value {{
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 1.25rem;
      font-weight: 700;
      color: {NAVY_DARK};
  }}

  .score-pill {{
      display: inline-block;
      background: linear-gradient(135deg, {GOLD} 0%, {GOLD_SOFT} 100%);
      color: {NAVY_DARK};
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 1.1rem;
      font-weight: 800;
      padding: 6px 18px;
      border-radius: 30px;
      letter-spacing: 0.5px;
  }}

  .infra-group-title {{
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 0.78rem;
      font-weight: 700;
      color: {RED_LIGHT};
      text-transform: uppercase;
      letter-spacing: 1px;
      margin: 14px 0 8px 0;
  }}

  /* ── Tab overrides ── */
  .stTabs [data-baseweb="tab-list"] {{
      gap: 4px;
      background: {GREY_BG};
      padding: 4px;
      border-radius: 10px;
      border: 1px solid {GREY_BORDER};
  }}
  .stTabs [data-baseweb="tab"] {{
      border-radius: 8px !important;
      font-family: 'Barlow', sans-serif !important;
      font-weight: 600 !important;
      font-size: 0.82rem !important;
      padding: 8px 16px !important;
      color: {GREY_TEXT} !important;
      background: transparent !important;
      border: none !important;
  }}
  .stTabs [aria-selected="true"] {{
      background: {NAVY_DARK} !important;
      color: white !important;
  }}

  [data-testid="stSelectbox"] label,
  [data-testid="stNumberInput"] label {{
      font-size: 0.75rem !important;
      font-weight: 600 !important;
      color: {GREY_TEXT} !important;
      text-transform: uppercase !important;
      letter-spacing: 0.5px !important;
      background-color: white important!;
  }}

  [data-testid="stInfo"] {{
      background: white !important;
      border: 1px solid rgba(14,42,92,0.15) !important;
      border-radius: 10px !important;
      color: {NAVY} !important;
      font-size: 0.82rem !important;
      font-weight: 500 !important;
  }}

  [data-testid="stMetricValue"] {{
      font-size: 1.35rem !important;
      font-weight: 700 !important;
      color: {NAVY_DARK} !important;
  }}
  [data-testid="stMetricLabel"] {{
      font-size: 0.7rem !important;
      font-weight: 600 !important;
      color: {GREY_MUTED} !important;
      text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
  }}

  hr {{
      border: none !important;
      border-top: 1px solid {GREY_BORDER} !important;
      margin: 1.8rem 0 !important;
  }}
</style>
""", unsafe_allow_html=True)

SPEED_COLOR_MAP = {
    'Below 100':  '#D2D7EA',
    '100 to 299': '#8C99CA',
    '300 to 499': '#4A5CA0',
    '500 to 699': '#2C375E',
    '700+':       '#14192A',
}

OPERATOR_COLORS = {
    'Zain':    '#0092A1',
    'STC':     '#4E008D',
    'Batelco': '#D50037',
}

BAHRAIN_CENTER = [26.21, 50.57]

TYPE_LABELS = {
    'PPL':'Populated Place', 'PPLX':'Section of Populated Place',
    'HTL':'Hotel/Tourism',   'PT':'Point / landform',
    'ISL':'Island',          'ISLX':'Section of Island',
    'ISLS':'Islands',        'PRT':'Port',
    'TOWR':'Tower'
}


@st.cache_data
def load_data():
    speed  = pd.read_csv(os.path.join(ROOT, 'data', 'speedtests.csv'))
    towers = pd.read_csv(os.path.join(ROOT, 'data', 'bahrain_towers.csv'))
    speed.columns  = speed.columns.str.strip()
    towers.columns = towers.columns.str.strip()
    speed['date']  = pd.to_datetime(speed['date'], format='%m/%d/%Y', errors='coerce')
    speed['year']  = speed['date'].dt.year
    return speed, towers

@st.cache_data
def load_encoders():
    with open(os.path.join(ROOT, 'assets', 'encoders.json')) as f:
        return json.load(f)

@st.cache_data
def load_feature_cols():
    with open(os.path.join(ROOT, 'outputs', 'models', 'feature_cols2.json')) as f:
        return json.load(f)

speed_df, towers_df = load_data()
encoders            = load_encoders()
FEAT_COLS           = load_feature_cols()
active_towers       = towers_df[towers_df['Visible'] == True].copy()


def _speed_category(speed):
    if speed < 100:   return 'Below 100'
    elif speed < 300: return '100 to 299'
    elif speed < 500: return '300 to 499'
    elif speed < 700: return '500 to 699'
    else:             return '700+'

def _speed_legend_html(title='Download Speed (Mbps)'):
    rows = ''.join(
        f"<div style='margin:3px 0'>"
        f"<span style='background:{c};width:14px;height:14px;"
        f"display:inline-block;margin-right:8px;border-radius:2px'></span>{label}</div>"
        for label, c in SPEED_COLOR_MAP.items()
    )
    return f"""
    <div style="position:fixed;bottom:30px;left:55px;width:210px;
        border:1px solid #ccc;z-index:9999;font-size:13px;
        background:white;padding:10px;border-radius:6px;box-shadow:2px 2px 6px rgba(0,0,0,.15)">
        <b>{title}</b><br>{rows}
    </div>"""

def section_header(icon, text):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-header-bar"></div>
        <div class="section-header-text">{icon}&nbsp; {text}</div>
    </div>""", unsafe_allow_html=True)

def kpi_card(col, icon, label, value, unit, accent='kpi-accent-navy'):
    col.markdown(f"""
    <div class="kpi-card {accent}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>
    </div>""", unsafe_allow_html=True)

def metric_mini(col, label, value):
    col.markdown(f"""
    <div class="metric-mini-card">
        <div class="metric-mini-label">{label}</div>
        <div class="metric-mini-value">{value}</div>
    </div>""", unsafe_allow_html=True)



@st.cache_data
def build_speed_grid_map(year: int) -> str:
    df = speed_df[speed_df['year'] == year].copy()
    df = df.dropna(subset=['latitude', 'longitude', 'avg_d_mbps'])
    df = df[
        (df['latitude']  >= 25.6) & (df['latitude']  <= 26.5) &
        (df['longitude'] >= 50.3) & (df['longitude'] <= 50.8)
    ]
    if df.empty:
        return None

    lat_vals  = np.sort(df['latitude'].unique())
    lon_vals  = np.sort(df['longitude'].unique())
    lat_diffs = np.diff(lat_vals)
    lon_diffs = np.diff(lon_vals)
    lat_step  = pd.Series(lat_diffs[lat_diffs > 0]).round(8).mode().iloc[0]
    lon_step  = pd.Series(lon_diffs[lon_diffs > 0]).round(8).mode().iloc[0]
    half_lat  = lat_step / 2
    half_lon  = lon_step / 2

    cell_df = (
        df.groupby(['latitude', 'longitude'], as_index=False)
          .agg(avg_d_mbps=('avg_d_mbps','mean'),
               avg_u_mbps=('avg_u_mbps','mean'),
               avg_lat_ms=('avg_lat_ms','mean'),
               tests=('tests','sum'))
    )
    cell_df['speed_category'] = cell_df['avg_d_mbps'].apply(_speed_category)

    m = folium.Map(location=BAHRAIN_CENTER, zoom_start=10, tiles='cartodbpositron')
    for _, row in cell_df.iterrows():
        poly  = Polygon([
            (row['longitude'] - half_lon, row['latitude'] - half_lat),
            (row['longitude'] + half_lon, row['latitude'] - half_lat),
            (row['longitude'] + half_lon, row['latitude'] + half_lat),
            (row['longitude'] - half_lon, row['latitude'] + half_lat),
        ])
        color = SPEED_COLOR_MAP.get(row['speed_category'], '#cccccc')
        popup = folium.Popup(
            f"<b>Download:</b> {row['avg_d_mbps']:.1f} Mbps<br>"
            f"<b>Upload:</b> {row['avg_u_mbps']:.1f} Mbps<br>"
            f"<b>Latency:</b> {row['avg_lat_ms']:.1f} ms<br>"
            f"<b>Tests:</b> {int(row['tests'])}", max_width=250
        )
        folium.GeoJson(
            poly,
            style_function=lambda f, c=color: {
                'fillColor': c, 'fillOpacity': 0.85,
                'stroke': False, 'weight': 0,
            },
            popup=popup
        ).add_to(m)

    m.get_root().html.add_child(Element(
        _speed_legend_html(f'Download Speed (Mbps) — {year}')
    ))
    return m._repr_html_()

@st.cache_data
def build_tower_map() -> str:
    df = active_towers.dropna(subset=['Latitude', 'Longitude']).copy()
    m  = folium.Map(location=BAHRAIN_CENTER, zoom_start=10, tiles='cartodbpositron')
    for op, grp in df.groupby('Operator'):
        color = OPERATOR_COLORS.get(op, '#888888')
        for _, row in grp.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=4,
                color=color, fill=True, fill_color=color, fill_opacity=0.7,
                weight=1,
                popup=folium.Popup(
                    f"<b>{op}</b><br>"
                    f"Type: {row.get('RAT SubType','—')}<br>"
                    f"Tower: {row.get('Tower Type','—')}",
                    max_width=180
                )
            ).add_to(m)

    legend = ''.join(
        f"<div style='margin:3px 0'>"
        f"<span style='background:{c};width:12px;height:12px;"
        f"display:inline-block;margin-right:8px;border-radius:50%'></span>{op}</div>"
        for op, c in OPERATOR_COLORS.items()
    )
    m.get_root().html.add_child(Element(f"""
    <div style="position:fixed;bottom:30px;left:55px;width:160px;
        border:1px solid #ccc;z-index:9999;font-size:13px;
        background:white;padding:10px;border-radius:6px;box-shadow:2px 2px 6px rgba(0,0,0,.15)">
        <b>Operator</b><br>{legend}
    </div>"""))
    return m._repr_html_()

def build_input_map(selected_lat: float, selected_lon: float) -> folium.Map:
    m = folium.Map(location=BAHRAIN_CENTER, zoom_start=11, tiles='cartodbpositron')
    folium.Marker(
        location=[selected_lat, selected_lon],
        popup=f'📍 Selected: {selected_lat:.5f}, {selected_lon:.5f}',
        icon=folium.Icon(color='red', icon='map-marker', prefix='fa'),
        draggable=False,
    ).add_to(m)
    return m

def build_recommendation_map(lat: float, lon: float, area: str, score: float) -> str:
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles='cartodbpositron')
    folium.CircleMarker(
        location=[lat, lon], radius=18,
        color='#8B1A1A', fill=True, fill_color='#8B1A1A', fill_opacity=0.2, weight=3,
    ).add_to(m)
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(
            f"<b>{area}</b><br>Priority Score: {score:.1f}/100", max_width=200
        ),
        icon=folium.Icon(color='red', icon='tower-broadcast', prefix='fa'),
    ).add_to(m)
    return m._repr_html_()


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def autofill_from_map_click(lat, lon, all_areas, all_cities, all_regions, all_types):
    df = speed_df.dropna(subset=['latitude', 'longitude']).copy()
    dists = df.apply(lambda r: haversine_km(lat, lon, r['latitude'], r['longitude']), axis=1)
    nearest = df.loc[dists.idxmin()]

    towers = active_towers.dropna(subset=['Latitude', 'Longitude']).copy()
    if len(towers) > 0:
        td = towers.apply(lambda r: haversine_km(lat, lon, r['Latitude'], r['Longitude']), axis=1)
        nearest_tower_distance = float(td.min())
        tower_count_1km = int((td <= 1).sum())
        tower_count_2km = int((td <= 2).sum())
        tower_count_5km = int((td <= 5).sum())
    else:
        nearest_tower_distance = 0.0
        tower_count_1km = 0
        tower_count_2km = 0
        tower_count_5km = 0

    area_name = nearest.get('area', None)
    growth = 0.0
    if area_name is not None:
        area_hist = (
            speed_df[speed_df['area'] == area_name]
            .groupby('year', as_index=False)['tests']
            .sum().sort_values('year')
        )
        if len(area_hist) >= 2:
            prev_tests = area_hist.iloc[-2]['tests']
            curr_tests = area_hist.iloc[-1]['tests']
            if prev_tests > 0:
                growth = ((curr_tests - prev_tests) / prev_tests) * 100

    st.session_state['map_lat']    = round(lat, 6)
    st.session_state['map_lon']    = round(lon, 6)
    st.session_state['sel_area']   = nearest.get('area',      all_areas[0])
    st.session_state['sel_city']   = nearest.get('city',      all_cities[0])
    st.session_state['sel_region'] = nearest.get('region',    all_regions[0])
    st.session_state['sel_type']   = nearest.get('typeOfArea',all_types[0])
    st.session_state['input_lat']  = round(lat, 6)
    st.session_state['input_lon']  = round(lon, 6)
    st.session_state['inp_tc1']    = float(tower_count_1km)
    st.session_state['inp_tc2']    = float(tower_count_2km)
    st.session_state['inp_tc5']    = float(tower_count_5km)
    st.session_state['inp_dist']   = round(nearest_tower_distance, 3)
    dem = nearest.get('digital_elevation_model', 5)
    st.session_state['inp_dem']    = 5 if pd.isna(dem) else int(dem)
    st.session_state['inp_year']   = datetime.datetime.now().year
    st.session_state['inp_quarter']= (datetime.datetime.now().month - 1) // 3 + 1
    tests_val = nearest.get('tests', 500)
    st.session_state['inp_tests']  = int(tests_val) if not pd.isna(tests_val) else 500
    st.session_state['inp_growth'] = round(growth, 2)

def build_feature_dict(lat, lon, area, city, region, type_of_area,
                       nearest_dist, tc1, tc2, tc5,
                       dem, demand_growth, year, quarter, tests):

    region_enc     = encoders['region'].get(region, 0)
    typeOfArea_enc = encoders['typeOfArea'].get(type_of_area, 0)
    city_enc       = encoders['city'].get(city, 0)

    area_raw = speed_df[speed_df['area'] == area].copy()
    if len(area_raw) > 0:
        area_raw['date']   = pd.to_datetime(area_raw['date'], errors='coerce')
        area_raw['year_q'] = area_raw['date'].dt.year.astype(str) + '_' + \
                             area_raw['date'].dt.quarter.astype(str)
        area_q = area_raw.groupby('year_q').agg(
            avg_d_mbps=('avg_d_mbps','mean'),
            avg_u_mbps=('avg_u_mbps','mean'),
            avg_lat_ms=('avg_lat_ms','mean'),
            tests=('tests','sum'),
        )
        area_median_d   = float(area_q['avg_d_mbps'].median())
        area_median_u   = float(area_q['avg_u_mbps'].median())
        area_median_lat = float(area_q['avg_lat_ms'].median())
        area_test_count = float(area_q['tests'].sum())
        rolling_d       = float(area_q['avg_d_mbps'].mean())
    else:
        area_median_d   = float(speed_df['avg_d_mbps'].mean())
        area_median_u   = float(speed_df['avg_u_mbps'].mean())
        area_median_lat = float(speed_df['avg_lat_ms'].mean())
        area_test_count = float(speed_df['tests'].median())
        rolling_d       = float(speed_df['avg_d_mbps'].mean())

    year_trend          = year - int(speed_df['year'].min())
    tower_density_ratio = tc1 / (tc5 + 1)
    distance_x_density  = nearest_dist * tower_density_ratio

    return {
        'latitude': lat, 'longitude': lon,
        'nearest_tower_distance_km': nearest_dist,
        'tower_count_1km': tc1, 'tower_count_2km': tc2, 'tower_count_5km': tc5,
        'tower_density_ratio': tower_density_ratio,
        'digital_elevation_model': dem,
        'region_enc': region_enc,
        'typeOfArea_enc': typeOfArea_enc,
        'city_enc': city_enc,
        'demand_growth_pct': demand_growth / 100.0,
        'year_trend': year_trend,
        'quarter': quarter,
        'area_rolling_d_mbps': rolling_d,
        'tests': tests,
        'area_median_d': area_median_d,
        'area_median_u': area_median_u,
        'area_median_lat': area_median_lat,
        'area_test_count': area_test_count,
        'distance_x_density': distance_x_density,
    }

if 'map_lat' not in st.session_state:
    st.session_state['map_lat'] = 26.2154
if 'map_lon' not in st.session_state:
    st.session_state['map_lon'] = 50.5832



st.markdown(f"""
<div class="top-nav">
    <div class="nav-brand">
        <div class="nav-icon">📡</div>
        <div>
            <div class="nav-title">Bahrain Network Intelligence</div>
            <div class="nav-subtitle">AI-Powered Infrastructure & Performance Platform</div>
        </div>
    </div>
    <div class="nav-badge">🟢 Live Dashboard</div>
</div>
""", unsafe_allow_html=True)



section_header(" ","Network Overview")

c1, c2, c3, c4 = st.columns(4)
kpi_card(c1, "⬇️", "Avg Download Speed",
         f"{speed_df['avg_d_mbps'].mean():.1f}", " Mbps", "kpi-accent-gold")
kpi_card(c2, "⬆️", "Avg Upload Speed",
         f"{speed_df['avg_u_mbps'].mean():.1f}", " Mbps", "kpi-accent-navy")
kpi_card(c3, "📶", "Avg Latency",
         f"{speed_df['avg_lat_ms'].mean():.0f}", " ms", "kpi-accent-red")
kpi_card(c4, "🗼", "Active Towers",
         f"{len(active_towers):,}", "", "kpi-accent-mixed")

st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

st.markdown("")

section_header(" ","Network Coverage Maps")

map_col1, map_col2 = st.columns(2)

with map_col1:
    st.markdown('<div class="map-card">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="map-card-title">
        <span class="map-dot"></span> Speed Distribution Map
    </div>
    <div class="map-hint">Each cell = one speed-test grid block · Click for details</div>
    """, unsafe_allow_html=True)
    year_options = sorted(speed_df['year'].dropna().unique(), reverse=True)
    sel_year = st.selectbox('Select year', year_options, index=0, key='overview_year')
    html = build_speed_grid_map(int(sel_year))
    if html:
        st.components.v1.html(html, height=400, scrolling=False)
    else:
        st.warning('No data for selected year.')
    st.markdown('</div>', unsafe_allow_html=True)

with map_col2:
    st.markdown('<div class="map-card">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="map-card-title">
        <span style= "margin-top: 4em;"class="map-dot"></span> Active Tower Distribution
    </div>
    <div class="map-hint">Zain · STC · Batelco &nbsp;|&nbsp; Click marker for details</div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="margin-top:38px;"></div>', unsafe_allow_html=True)
    tower_html = build_tower_map()
    st.components.v1.html(tower_html, height=400, scrolling=False)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)


st.markdown(" ")
st.markdown(" ")
st.markdown(" ")


section_header("", "Speed Trends & Area Distribution")

chart1, chart2 = st.columns(2)

with chart1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">⬇️ Yearly Average Download Speed</div>', unsafe_allow_html=True)
    yearly = speed_df.groupby("year")["avg_d_mbps"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    ax.fill_between(yearly["year"], yearly["avg_d_mbps"],
                    alpha=0.12, color=NAVY)
    ax.plot(yearly["year"], yearly["avg_d_mbps"],
            marker="o", color=NAVY, linewidth=2.2,
            markerfacecolor=GOLD, markeredgecolor=NAVY, markersize=7)
    ax.set_ylabel("Avg Download (Mbps)", color=GREY_TEXT, fontsize=9)
    ax.set_xlabel("Year", color=GREY_TEXT, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.25, color=GREY_MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GREY_BORDER)
    ax.spines["bottom"].set_color(GREY_BORDER)
    ax.tick_params(colors=GREY_TEXT, labelsize=8)
    plt.tight_layout()
    st.pyplot(fig, transparent=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

with chart2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📶 Average Speed by Area Type</div>', unsafe_allow_html=True)
    type_avg = speed_df.groupby("typeOfArea")["avg_d_mbps"].mean().sort_values()
    type_avg.index = [TYPE_LABELS.get(x, x) for x in type_avg.index]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    bars = ax.barh(type_avg.index, type_avg.values,
                   color=NAVY, edgecolor='none', height=0.6)
    # Gold accent on highest bar
    bars[-1].set_color(GOLD)
    bars[-1].set_edgecolor(NAVY)
    ax.set_xlabel("Avg Download Speed (Mbps)", color=GREY_TEXT, fontsize=9)
    ax.grid(True, axis="x", linestyle='--', alpha=0.25, color=GREY_MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GREY_BORDER)
    ax.spines["bottom"].set_color(GREY_BORDER)
    ax.tick_params(colors=GREY_TEXT, labelsize=8)
    plt.tight_layout()
    st.pyplot(fig, transparent=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)


st.markdown(" ")
st.markdown(" ")
st.markdown(" ")


st.markdown(f"""
<div class="predict-header">
    <div class="predict-header-icon"></div>
    <div>
        <div class="predict-header-title">AI-Powered Network Performance Prediction</div>
        <div class="predict-header-sub">Click anywhere on the map to auto-fill coordinates and infrastructure fields</div>
    </div>
</div>
""", unsafe_allow_html=True)

all_areas   = sorted(speed_df['area'].dropna().unique())
all_cities  = sorted(speed_df['city'].dropna().unique())
all_regions = sorted(speed_df['region'].dropna().unique())
all_types   = sorted(speed_df['typeOfArea'].dropna().unique())

for k, v in [
    ('sel_area',   all_areas[0]),
    ('sel_city',   all_cities[0]),
    ('sel_region', all_regions[0]),
    ('sel_type',   all_types[0]),
    ('input_lat',  st.session_state['map_lat']),
    ('input_lon',  st.session_state['map_lon']),
    ('inp_tc1',    2.0), ('inp_tc2', 5.0), ('inp_tc5', 12.0),
    ('inp_dist',   0.5), ('inp_dem', 5),
    ('inp_year',   datetime.datetime.now().year),
    ('inp_quarter',(datetime.datetime.now().month - 1) // 3 + 1),
    ('inp_tests',  500), ('inp_growth', 5.0),
]:
    if k not in st.session_state:
        st.session_state[k] = v

map_section, form_section = st.columns([1, 1])

with map_section:
    st.markdown('<div class="map-card">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="map-card-title">
        <span class="map-dot"></span> Click Map to Set Location
    </div>
    <div class="map-hint">Click anywhere on Bahrain — all fields auto-fill instantly</div>
    """, unsafe_allow_html=True)
    input_map  = build_input_map(st.session_state['map_lat'], st.session_state['map_lon'])
    map_result = st_folium(input_map, width=None, height=560, key='input_map')

    if map_result and map_result.get('last_clicked'):
        clicked     = map_result['last_clicked']
        clicked_lat = round(clicked['lat'], 6)
        clicked_lon = round(clicked['lng'], 6)
        if (
            clicked_lat != st.session_state['map_lat'] or
            clicked_lon != st.session_state['map_lon']
        ):
            autofill_from_map_click(
                clicked_lat, clicked_lon,
                all_areas, all_cities, all_regions, all_types
            )
            st.rerun()

    st.info(f"📍 Selected: **{st.session_state['map_lat']}, {st.session_state['map_lon']}**")
    st.markdown('</div>', unsafe_allow_html=True)

with form_section:
    st.markdown('<div class="form-panel">', unsafe_allow_html=True)
    with st.form('prediction_form'):

        st.markdown('<div class="form-group-label">📍 Location</div>', unsafe_allow_html=True)
        f1, f2 = st.columns(2)
        sel_area   = f1.selectbox('Area',   all_areas,   key='sel_area')
        sel_city   = f2.selectbox('City',   all_cities,  key='sel_city')
        f3, f4 = st.columns(2)
        sel_region = f3.selectbox('Region', all_regions, key='sel_region')
        sel_type   = f4.selectbox(
            'Area Type', all_types, key='sel_type',
            format_func=lambda x: f"{x} — {TYPE_LABELS.get(x, x)}"
        )

        st.markdown('<div class="form-group-label">🌐 Coordinates</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        inp_lat = c1.number_input('Latitude',  key='input_lat', format='%.6f')
        inp_lon = c2.number_input('Longitude', key='input_lon', format='%.6f')

        st.markdown('<div class="form-group-label">🗼 Tower Infrastructure</div>', unsafe_allow_html=True)
        t1, t2, t3 = st.columns(3)
        inp_tc1  = t1.number_input('Towers 1 km',        min_value=0.0, step=1.0, key='inp_tc1')
        inp_tc2  = t2.number_input('Towers 2 km',        min_value=0.0, step=1.0, key='inp_tc2')
        inp_tc5  = t3.number_input('Towers 5 km',        min_value=0.0, step=1.0, key='inp_tc5')
        t4, t5 = st.columns(2)
        inp_dist = t4.number_input('Nearest Tower (km)', min_value=0.0, step=0.1,  key='inp_dist')
        inp_dem  = t5.number_input('Elevation (m)',      step=1,                   key='inp_dem')

        st.markdown('<div class="form-group-label">📅 Temporal & Demand</div>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        inp_year    = d1.selectbox('Year',           list(range(2019, 2027)), key='inp_year')
        inp_quarter = d2.selectbox('Quarter',        [1, 2, 3, 4],           key='inp_quarter')
        inp_tests   = d3.number_input('Test Count',  min_value=1, step=50,   key='inp_tests')
        inp_growth  = d4.number_input('Demand Growth (%)', step=1.0,         key='inp_growth')

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button('🔮 Predict Network Speed', use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    try:
        feat = build_feature_dict(
            lat=inp_lat, lon=inp_lon,
            area=sel_area, city=sel_city,
            region=sel_region, type_of_area=sel_type,
            nearest_dist=inp_dist, tc1=inp_tc1, tc2=inp_tc2, tc5=inp_tc5,
            dem=inp_dem, demand_growth=inp_growth,
            year=inp_year, quarter=inp_quarter, tests=inp_tests,
        )
        preds = predict_speeds(feat)
        d_val = preds["avg_d_mbps"]

        if d_val >= 200:
            q_color, q_text, q_icon, q_bg, q_border = (
                "#16a34a", "Excellent — 5G-class speeds", "✓",
                "#f0fdf4", "#bbf7d0"
            )
        elif d_val >= 100:
            q_color, q_text, q_icon, q_bg, q_border = (
                "#d97706", "Good — Solid LTE-A performance", "✓",
                "#fffbeb", "#fde68a"
            )
        elif d_val >= 50:
            q_color, q_text, q_icon, q_bg, q_border = (
                "#ea580c", "Fair — Standard LTE", "!",
                "#fff7ed", "#fed7aa"
            )
        else:
            q_color, q_text, q_icon, q_bg, q_border = (
                RED_LIGHT, "Poor — Tower investment recommended", "!",
                "#fef2f2", "#fecaca"
            )

        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
        section_header("🔮", "Predicted Network Performance")

        r1, r2, r3 = st.columns(3)
        for col, icon, label, val, unit, bg in [
            (r1, "⬇️", "Download Speed", preds['avg_d_mbps'], "Mbps", f"rgba(14,42,92,0.08)"),
            (r2, "⬆️", "Upload Speed",   preds['avg_u_mbps'], "Mbps", f"rgba(139,26,26,0.08)"),
            (r3, "◌",  "Latency",        preds['avg_lat_ms'], "ms",   f"rgba(255,192,0,0.12)"),
        ]:
            col.markdown(f"""
            <div class="result-card">
                <div class="result-icon" style="background:{bg}">{icon}</div>
                <div class="result-label">{label}</div>
                <div class="result-value">{val}<span class="result-unit"> {unit}</span></div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="quality-banner" style="background:{q_bg};border-color:{q_border};">
            <div class="quality-circle" style="background:{q_color};">{q_icon}</div>
            <div>
                <div class="quality-label">Network Quality Assessment</div>
                <div class="quality-text">{q_text}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

st.markdown(" ")


st.markdown(" ")
st.markdown(" ")
st.markdown(" ")
st.markdown(" ")


from recommend import run_tower_impact_agent, simulate_tower_impact

st.markdown(f"""
<div class="rec-header">
    <div class="rec-header-icon">🏗️</div>
    <div>
        <div class="rec-header-title">Top Tower Recommendation Areas</div>
        <div class="rec-header-sub">AI-scored priority ranking for new infrastructure deployment</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(" ")

@st.cache_data
def cached_recommendations():
    return get_top_recommendations(top_n=5)

recs = cached_recommendations()

tab_labels = [f"#{r['rank']}  {r['area']}" for r in recs]
inner_tabs  = st.tabs(tab_labels)

for tab, rec in zip(inner_tabs, recs):
    with tab:
        left, right = st.columns([3, 2], gap="large")

        with left:
            rank_color = [GOLD, '#C0C0C0', '#CD7F32', NAVY, NAVY][rec['rank'] - 1]
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:14px;margin-bottom:1px;">
                <div style="
                    min-width:48px;height:48px;
                    background:{rank_color};
                    border-radius:12px;
                    display:flex;align-items:center;margin-top:0.98em;justify-content:center;
                    font-size:1.4rem;font-weight:800;
                    color:{NAVY_DARK if rec['rank']==1 else 'white'};
                ">#{rec['rank']}</div>
                <div>
                    <br>
                    <div class="rec-area-title">{rec['area']}</div>
                    <div style="margin-top:5px;">
                        <span class="rec-location-tag">{rec['city']}</span>
                        <span class="rec-location-tag">{rec['region']}</span>
                        <span class="rec-location-tag">{rec['typeOfArea_label']}</span>
                    </div>
                </div>
            </div>
            <br>
            <div class="rec-coords">📍 {rec['latitude']}, {rec['longitude']}</div>
            <div class="rec-summary">{rec['recommendation_summary']}</div>
            """, unsafe_allow_html=True)


            st.markdown(" ")
            
            st.markdown('<div class="infra-group-title">Current Performance</div>',
                        unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            metric_mini(m1, "Download",  f"{rec['avg_d_mbps']} Mbps")
            metric_mini(m2, "Upload",    f"{rec['avg_u_mbps']} Mbps")
            metric_mini(m3, "Latency",   f"{rec['avg_lat_ms']} ms")
            metric_mini(m4, "Tests",     f"{rec['tests']:,}")

            st.markdown('')

            st.markdown('<div class="infra-group-title">Infrastructure Context</div>',
                        unsafe_allow_html=True)
            i1, i2, i3, i4 = st.columns(4)
            metric_mini(i1, "Nearest Tower", f"{rec['nearest_tower_distance_km']} km")
            metric_mini(i2, "Towers 1 km",   str(rec['tower_count_1km']))
            metric_mini(i3, "Towers 2 km",   str(rec['tower_count_2km']))
            metric_mini(i4, "Towers 5 km",   str(rec['tower_count_5km']))

        with right:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {NAVY_DARK}, {NAVY});
                border-radius: 14px;
                padding: 18px;
                margin-bottom: 14px;
                text-align: center;
                border-bottom: 3px solid {GOLD};
            ">
                <div style="font-size:0.7rem;font-weight:600;color:rgba(255,255,255,0.5);
                            text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
                    Priority Score
                </div>
                <span class="score-pill">{rec['priority_score']:.1f} / 100</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.78rem;
                        font-weight:700;color:{NAVY_DARK};text-transform:uppercase;
                        letter-spacing:0.5px;margin-bottom:8px;">
                🗺️ Location
            </div>""", unsafe_allow_html=True)
            rec_map_html = build_recommendation_map(
                rec['latitude'], rec['longitude'],
                rec['area'], rec['priority_score']
            )
            st.components.v1.html(rec_map_html, height=260, scrolling=False)

st.markdown(" ")
st.markdown(" ")
st.markdown(" ")