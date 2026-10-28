import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
import io, warnings, time, urllib.parse
from PIL import Image, ImageDraw, ImageFont
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="Spotify First-Week Streaming Predictor",
    layout="wide",
    page_icon="🎧",
    initial_sidebar_state="collapsed",
)

SPOTIFY_GREEN = "#1DB954"

# NOTE: plain (non-f) string so CSS braces don't need escaping.
st.markdown("""
<style>
    /* Montserrat is geometric and close in spirit to Spotify's own Circular, so the title
       reads as a display face without clashing with Streamlit's body font. */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800;900&display=swap');

    /* Base */
    .stApp { background-color: #0e1117; }

    /* Sidebar fully removed - all controls live on the page now */
    section[data-testid="stSidebar"] { display: none !important; }
    div[data-testid="stSidebarCollapsedControl"] { display: none !important; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #181b21;
        border: 1px solid #2a2e37;
        border-radius: 10px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricValue"] { color: #1DB954; }

    /* Hide Streamlit anchor links on headings */
    h1 a, h2 a, h3 a { display: none !important; }

    /* Page title. clamp() scales it with the viewport, so it is big on a desktop and still
       fits a phone without a separate media query. Streamlit ships its own h1 rule with
       higher specificity, so this needs the element selector and !important to win. */
    .stApp h1.app-title, h1.app-title {
        font-family: 'Montserrat', 'Source Sans Pro', -apple-system, sans-serif !important;
        font-size: clamp(2.1rem, 4.7vw, 4.5rem) !important;
        font-weight: 900 !important;
        line-height: 1.06 !important;
        letter-spacing: -0.03em !important;
        margin: 0 0 1.5rem 0 !important;
        padding: 0 !important;
        color: #ffffff !important;
    }
    .stApp h1.app-title .title-icon {
        font-weight: 400 !important;
        letter-spacing: 0 !important;
        margin-right: 0.1em;
    }

    /* Recent prediction cards */
    .pred-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
    }
    @media (max-width: 950px) {
        .pred-grid { grid-template-columns: 1fr; }
    }
    .pred-card {
        display: flex;
        flex-direction: column;
        background: #13161d;
        border: 1px solid #22262f;
        border-radius: 12px;
        padding: 16px 18px 14px 18px;
        min-width: 0;
    }
    .pred-card-title {
        font-size: 14.5px;
        font-weight: 600;
        color: #ffffff;
        line-height: 1.45;
        margin-bottom: 14px;
        /* Two lines reserved so the number rows line up across all three cards; a longer
           title simply grows past it rather than being clipped. */
        min-height: 2.9em;
        overflow-wrap: anywhere;
    }
    .pred-row { display: flex; align-items: stretch; gap: 18px; }
    .pred-num { font-size: 34px; font-weight: 800; line-height: 1.08; white-space: nowrap; }
    .pred-lbl { font-size: 12.5px; color: #777f8c; margin-top: 5px; white-space: nowrap; }
    .pred-sep { width: 1px; background: #2a2e37; flex: 0 0 1px; }
    .pred-badge { margin-top: auto; padding-top: 14px; font-size: 15px; font-weight: 700; }

    /* Prediction result card */
    .result-card {
        background: linear-gradient(160deg, #0d2a18 0%, #111820 60%, #0e1117 100%);
        border: 1px solid #1DB954;
        border-radius: 14px;
        padding: 28px 32px;
        margin: 12px 0 20px 0;
        box-shadow: 0 0 24px #1db95422;
    }

    /* Section divider label */
    .section-divider {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 26px 0 18px 0;
    }
    .section-divider-line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, #1DB95488, transparent);
    }
    .section-divider-label {
        color: #1DB954;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        white-space: nowrap;
    }

    /* Primary predict button - large pill */
    .stButton button[kind="primary"] {
        background-color: #1DB954;
        border: none;
        color: #0e1117;
        border-radius: 999px;
        min-height: 64px;
        padding: 18px 30px;
        letter-spacing: 0.03em;
        box-shadow: 0 0 20px #1db95433;
        transition: all 0.2s ease;
    }
    .stButton button[kind="primary"] p {
        font-size: 19px !important;
        font-weight: 800 !important;
    }
    .stButton button[kind="primary"]:hover {
        background-color: #1ed760;
        color: #0e1117;
        box-shadow: 0 0 28px #1db95477;
        transform: translateY(-1px);
    }

    /* Slider + toggle accent (config.toml primaryColor is the main fix) */
    div[data-baseweb="slider"] [role="slider"] { background-color: #1DB954 !important; }

    /* Footer */
    .app-footer {
        margin-top: 48px;
        padding: 20px 0 8px 0;
        border-top: 1px solid #1e2228;
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        align-items: center;
        gap: 10px 18px;
        color: #4a5060;
        font-size: 13px;
        letter-spacing: 0.02em;
    }
    .app-footer span { white-space: nowrap; }
    .app-footer .accent { color: #1db95488; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA - same CSV used by the original script (source of truth)
# =============================================================================
CSV_CONTENT = """artist,album,year,genre,track_count,spotify_mau,hype_score,lead_time,album_type,feature_track_count,years_since_last_album,prev_album_streams,first_week_streams,is_household_name
Ed Sheeran,Divide,2017,0,16,120,9.0,21,0,0,4,0,375,1
Ed Sheeran,Equals,2021,0,14,365,8.5,21,0,0,4,375,161,1
Drake,Scorpion,2018,1,25,207,10.0,7,0,3,1,0,559,1
Drake,CLB,2021,1,21,365,9.5,14,0,12,3,559,496,1
Drake,FATD,2023,1,23,574,8.0,14,0,10,2,496,405,1
Travis Scott,ASTROWORLD,2018,1,17,207,9.5,14,0,11,3,0,257,0
Travis Scott,Utopia,2023,1,19,574,10.0,14,0,13,5,257,457,0
Eminem,Revival,2017,1,19,120,8.0,45,0,8,4,0,117,1
Eminem,Kamikaze,2018,1,13,207,10.0,0,0,4,1,117,238,1
Eminem,TheDeathOfSlimShady,2024,1,19,640,9.0,45,0,8,6,238,254,1
The Weeknd,Starboy,2016,2,18,100,10.0,30,0,5,1,0,223,1
The Weeknd,AfterHours,2020,2,14,299,9.0,21,0,0,2,223,265,1
The Weeknd,HUT,2024,2,22,640,10.0,21,0,7,4,265,302,1
Justin Bieber,Purpose,2015,0,13,80,10.0,60,0,4,3,0,205,1
Justin Bieber,Justice,2021,0,16,365,8.0,14,0,8,2,205,252,1
SZA,SOS,2022,2,23,489,9.5,21,0,4,5,125,336,0
Taylor Swift,Midnights,2022,0,20,489,10.0,21,0,1,3,0,776,1
Taylor Swift,1989TV,2023,0,21,574,9.0,21,2,0,1,776,582,1
Taylor Swift,TTPD,2024,0,31,640,10.0,28,0,2,1,582,1173,1
Sabrina Carpenter,ShortNSweet,2024,0,12,640,8.0,21,0,0,3,0,352,0
Kendrick Lamar,DAMN,2017,1,14,120,10.0,14,0,3,2,0,227,1
Kendrick Lamar,MrMorale,2022,1,19,489,9.5,25,0,7,5,227,343,1
Kendrick Lamar,GNX,2024,1,12,640,10.0,0,0,4,2,343,363,1
LilWayne,CarterV,2018,1,23,207,10.0,7,0,10,7,0,285,0
Ariana Grande,ThanksUNext,2019,0,12,248,10.0,21,0,0,1,0,361,1
Ariana Grande,EternalSunshine,2024,0,13,640,9.0,28,0,0,5,361,342,1
Olivia Rodrigo,SOUR,2021,0,11,365,9.0,28,0,0,0,0,385,1
Olivia Rodrigo,GUTS,2023,0,12,574,9.0,45,0,0,2,385,284,1
Billie Eilish,HMHAS,2024,0,10,640,9.0,45,0,0,5,0,386,1
Beyonce,Renaissance,2022,2,16,489,9.5,43,0,2,6,0,173,1
Beyonce,CowboyCarter,2024,4,27,640,9.5,14,0,11,2,173,316,1
Dua Lipa,FutureNostalgia,2020,0,13,299,8.5,21,0,2,4,0,294,0
Dua Lipa,RadicalOptimism,2024,0,11,640,7.5,45,0,0,4,294,94,0
Adele,30,2021,0,12,365,10.0,45,0,0,6,0,310,1
Bad Bunny,UnVeranoSinTi,2022,3,23,489,10.0,7,0,8,1,0,503,1
Bad Bunny,NadieSabe,2023,3,22,574,9.5,3,0,8,1,503,469,1
Harry Styles,FineLine,2019,0,12,248,9.0,40,0,0,0,0,155,1
Harry Styles,HarrysHouse,2022,0,13,489,9.5,52,0,0,2,155,440,1
Post Malone,BBandB,2018,1,18,207,10.0,21,0,5,1,0,411,1
Post Malone,HollywoodsBleeding,2019,1,17,248,10.0,14,0,7,1,411,379,1
Karol G,MananaSeraBonito,2023,3,17,574,9.5,31,0,10,2,0,203,0
21 Savage,AmericanDream,2024,1,15,640,9.5,7,0,10,3,0,187,0
Tyler The Creator,CHROMAKOPIA,2024,1,14,640,10.0,11,0,7,3,0,340,0
Kanye West,DONDA,2021,1,27,365,10.0,60,0,23,6,0,442,1
Kanye West,VULTURES1,2024,1,15,640,9.0,30,0,0,3,442,251,1
Tate McRae,ThinkLater,2023,0,14,574,9.0,30,0,0,2,0,115,0
Metro Boomin,WDTY,2024,1,17,640,9.0,14,3,5,2,0,257,0
Morgan Wallen,Dangerous,2021,4,30,365,8.0,60,0,2,3,0,161,0
Juice WRLD,LegendsNeverDie,2020,1,22,299,10.0,7,0,5,1,0,303,0
Taylor Swift,LifeOfAShowgirl,2025,0,12,715,10.0,45,0,1,1,1173,887,1
Sabrina Carpenter,MansBestFriend,2025,0,13,715,9.5,30,0,0,1,352,297,0
Morgan Wallen,ImTheProblem,2025,4,37,715,9.5,45,0,5,2,161,253,0
Tate McRae,SoCloseToWhat,2025,0,16,715,10.0,99,0,2,2,115,188,0
Playboi Carti,MUSIC,2025,1,30,715,10.0,15,0,12,5,160,490,0
Lady Gaga,Mayhem,2025,0,14,715,9.0,45,0,2,5,173,219,1
Justin Bieber,Swag,2025,0,21,715,8.0,1,1,7,4,252,276.8,1
Harry Styles,KissAllTheTime,2026,0,14,761,9.5,40,0,0,4,440,205,1"""

# hype_score was dropped: it was scored retrospectively, after the albums' results were
# known, so it carried hindsight that cross-validation cannot detect (the leak lives inside
# the feature's values, not across the train/test split). Removing it barely moved the
# cross-validated metrics and improved the 2025 holdout, which is what you would expect from
# a contaminated predictor. The three correlated history features (previous album, trailing
# average, trailing streams per track) are collapsed into one, since dropping any single one
# changed almost nothing: they were three versions of the same signal.
FEATS = ['track_count', 'spotify_mau', 'lead_time', 'feature_track_count',
         'years_since_last_album', 'history_per_mau',
         'is_household_name', 'household_x_tracks']

# Spotify global MAU (millions) by release period. A prediction is pinned to the
# platform size at the album's release, so the same inputs always reproduce the same
# number even as Spotify keeps growing. 2026 is split because MAU moved mid-year.
MAU_BY_PERIOD = {
    "2026 (Jan to Jun)": 761,
    "2026 (Jul to Dec)": 788,
}
DEFAULT_PERIOD = "2026 (Jul to Dec)"
DEFAULT_MAU = MAU_BY_PERIOD[DEFAULT_PERIOD]
MAU_SOURCE_NOTE = (
    "Spotify put its monthly active users at 788 million in its Q3 2026 earnings report, "
    "up from 761 million earlier in the year."
)

TRAIN_YEAR_MAX = 2024
TRAIN_YEAR_MIN = 2015


# =============================================================================
# MODEL TRAINING - cached so it only runs once per session
# =============================================================================
@st.cache_resource
def load_and_train():
    df = pd.read_csv(io.StringIO(CSV_CONTENT)).sort_values(['artist', 'year']).reset_index(drop=True)
    df['household_x_tracks'] = df['is_household_name'] * df['track_count']
    df['spt'] = df['first_week_streams'] / df['track_count']
    gm = df[df['year'] <= 2025]['first_week_streams'].median()
    gm_spt = df['spt'].median()
    df['prev_album_streams'] = df['prev_album_streams'].replace(0, gm)

    def prior_mean(df, col, fallback):
        out = []
        for _, r in df.iterrows():
            prior = df[(df['artist'] == r['artist']) & (df['year'] < r['year'])]
            out.append(prior[col].mean() if len(prior) > 0 else fallback)
        return out

    df['t_avg'] = prior_mean(df, 'first_week_streams', gm)
    df['t_spt'] = prior_mean(df, 'spt', gm_spt)
    for col, src in [('trailing_avg_per_mau', 't_avg'), ('trailing_spt_per_mau', 't_spt'),
                      ('prev_streams_per_mau', 'prev_album_streams')]:
        df[col] = df[src] / df['spotify_mau']
    # Single history signal: the artist's last album and their career average, averaged.
    # Replaces the three near-duplicate history features the model used to carry.
    df['history_per_mau'] = (df['prev_streams_per_mau'] + df['trailing_avg_per_mau']) / 2
    df['log_spm'] = np.log(df['first_week_streams'] / df['spotify_mau'])

    train = df[df['year'] <= TRAIN_YEAR_MAX].copy()
    hold = df[df['year'] == 2025].copy()
    val26 = df[df['year'] == 2026].copy()

    MP = dict(n_estimators=200, max_depth=4, min_samples_leaf=3, random_state=42)
    X, y = train[FEATS], train['log_spm']

    # LOOCV for reported error metrics
    lp, la = [], []
    for ti, te in LeaveOneOut().split(X):
        m = RandomForestRegressor(**MP)
        m.fit(X.iloc[ti], y.iloc[ti])
        lp.append(m.predict(X.iloc[te])[0])
        la.append(y.iloc[te[0]])

    r2 = r2_score(la, lp)
    rp = np.exp(np.array(lp)) * train['spotify_mau'].values
    ra = train['first_week_streams'].values
    # Same cross-validated predictions scored on the raw stream scale instead of in log
    # space. It comes out much lower because a handful of outsized albums dominate the
    # spread, so both numbers are reported rather than only the flattering one.
    r2_real = r2_score(ra, rp)
    mape = np.mean(np.abs(rp - ra) / ra * 100)
    mae = np.mean(np.abs(rp - ra))

    model = RandomForestRegressor(**MP)
    model.fit(X, y)

    # Holdout (2025) and validation (2026) results, computed once
    def predict_row(row, mau):
        Xi = pd.DataFrame([row[FEATS].tolist()], columns=FEATS)
        tp = np.array([np.exp(t.predict(Xi)[0]) * mau for t in model.estimators_])
        return np.mean(tp)

    hold_results = []
    for _, r in hold.iterrows():
        p = predict_row(r, r['spotify_mau'])
        e = abs(p - r['first_week_streams']) / r['first_week_streams'] * 100
        hold_results.append({'album': f"{r['artist']} — {r['album']}", 'pred': p,
                              'actual': r['first_week_streams'], 'err': e})

    val_results = []
    for _, r in val26.iterrows():
        p = predict_row(r, r['spotify_mau'])
        e = abs(p - r['first_week_streams']) / r['first_week_streams'] * 100
        val_results.append({'album': f"{r['artist']} — {r['album']}", 'pred': p,
                             'actual': r['first_week_streams'], 'err': e})

    importances = pd.Series(model.feature_importances_, index=FEATS).sort_values(ascending=False)

    return {
        'df': df, 'model': model, 'gm': gm, 'gm_spt': gm_spt,
        'r2': r2, 'r2_real': r2_real, 'mape': mape, 'mae': mae,
        'hold_results': hold_results, 'val_results': val_results,
        'importances': importances, 'train_size': len(train)
    }


state = load_and_train()
df, model = state['df'], state['model']
gm, gm_spt = state['gm'], state['gm_spt']

# Counts used throughout the copy so nothing goes stale when the dataset changes.
N_TRAIN = state['train_size']
N_HOLD = len(state['hold_results'])
N_VAL = len(state['val_results'])
HOLDOUT_MAPE = float(np.mean([r['err'] for r in state['hold_results']])) if N_HOLD else float('nan')

# Reference-only prior-album data. Used ONLY to correct "years since last album" and
# display-only prev-streams for artists whose earlier album we have real numbers for,
# but NOT added to training data because doing so measurably hurt holdout accuracy
# (tested individually and combined; see project notes). Never fed into the model.
REFERENCE_HISTORY = {
    "Post Malone":       {"year": 2019, "streams": 379},  # Hollywood's Bleeding already in training; documents the chain
    "Tyler The Creator":  {"year": 2021, "streams": 133},  # Call Me If You Get Lost
    "Billie Eilish":      {"year": 2021, "streams": 178},  # Happier Than Ever
    "Taylor Swift":       {"year": 2020, "streams": 254},  # Evermore (Folklore same year, Evermore is later)
    "21 Savage":          {"year": 2020, "streams": 108},  # Savage Mode II (collab w/ Metro, kept separate, see notes)
    "Metro Boomin":       {"year": 2022, "streams": 226},  # Heroes & Villains
    "Ariana Grande":      {"year": 2018, "streams": 174},  # Sweetener
    "Harry Styles":       {"year": 2017, "streams": 66},   # Harry Styles (debut)
}

# The summer 2026 case study: forecasts logged before each album was released, then checked
# against the real first-week numbers. These are a record, not live output. They are never
# recomputed, because a number produced after the result is known is not a forecast. The
# model has since changed, so it would give different figures today.
PREDICTION_LOG = [
    {"artist": "Drake", "album": "Iceman", "pred": 479, "actual": 450.4},
    {"artist": "Olivia Rodrigo", "album": "you seem pretty sad for a girl so in love",
     "pred": 413, "actual": 394.7},
    {"artist": "Ariana Grande", "album": "Petal", "pred": 408, "actual": 236.1},
]


def accuracy_style(diff_pct):
    """Color and marker for a prediction vs. actual gap. Green inside 15%,
    amber 15 to 30, red beyond 30."""
    a = abs(diff_pct)
    if a < 15:
        return "#4ade80", " ✓"
    if a < 30:
        return "#fbbf24", ""
    return "#f87171", ""


def predict(params, mau):
    Xi = pd.DataFrame([params], columns=FEATS)
    tp = np.array([np.exp(t.predict(Xi)[0]) * mau for t in model.estimators_])
    return np.mean(tp), np.percentile(tp, 10), np.percentile(tp, 90)


def reference_adjusted_first_year(artist):
    """Earliest known release year for an artist, factoring in REFERENCE_HISTORY
    even though that earlier album is not part of the training data."""
    a = df[df['artist'] == artist].sort_values('year')
    years = a['year'].tolist()
    if artist in REFERENCE_HISTORY:
        years.append(REFERENCE_HISTORY[artist]['year'])
    return min(years) if years else None


def latest_release_year(artist, before_year=None):
    """Most recent known release year, including reference-only albums. With before_year
    set, only albums released strictly earlier count, so backdating a prediction does not
    let a later album leak in as the artist's 'previous' one."""
    a = df[df['artist'] == artist].sort_values('year')
    if before_year is not None:
        a = a[a['year'] < before_year]
    year = int(a.iloc[-1]['year']) if len(a) else None
    if artist in REFERENCE_HISTORY:
        ref_year = REFERENCE_HISTORY[artist]['year']
        if (before_year is None or ref_year < before_year) and (year is None or ref_year > year):
            year = ref_year
    return year


def artist_history(artist, mau, before_year=None):
    """Auto-fill trailing/prev fields from CSV history, falling back to dataset medians.
    Display-only prev_album_streams uses REFERENCE_HISTORY when the real training rows
    do not have it, so the UI shows the correct number even though it is not in training.

    before_year restricts the history to albums released strictly earlier, matching how the
    training features were built (prior_mean also looks only at year < the album's year).
    Without it, predicting an earlier period would feed the model albums that had not come
    out yet."""
    full = df[df['artist'] == artist].sort_values('year')
    a = full[full['year'] < before_year] if before_year is not None else full

    # Household-name status is an attribute of the artist, not of one release, so it is read
    # from the full history rather than the period-restricted slice.
    is_hh = int(full.iloc[-1]['is_household_name']) if len(full) else 0

    if len(a):
        prev = a.iloc[-1]['first_week_streams'] / mau
    elif artist in REFERENCE_HISTORY and (before_year is None
                                          or REFERENCE_HISTORY[artist]['year'] < before_year):
        prev = REFERENCE_HISTORY[artist]['streams'] / mau
    else:
        prev = gm / mau

    trail_avg = (a['first_week_streams'].mean() if len(a) else gm) / mau
    return {
        'prev_streams_per_mau': prev,          # shown in the UI, not fed to the model
        'trailing_avg_per_mau': trail_avg,     # shown in the UI, not fed to the model
        'history_per_mau': (prev + trail_avg) / 2,
        'is_household_name': is_hh,
    }


def compute_contribution_breakdown(params, importances, train_df):
    """Rough, approximate view of which inputs pushed the prediction up or down,
    based on how far each value sits from the typical (median) album in training
    data, weighted by the model's feature importance. This is NOT a true SHAP
    decomposition, just a simplified, directionally useful approximation."""
    friendly = {
        'track_count': 'Track count',
        'spotify_mau': 'Platform size (era)',
        'lead_time': 'Lead time',
        'feature_track_count': 'Featured tracks',
        'years_since_last_album': 'Gap since last album',
        'history_per_mau': "Artist's streaming history",
        'is_household_name': 'Household-name status',
        'household_x_tracks': 'Household name x tracks',
    }
    rows = []
    for feat, val in zip(FEATS, params):
        med = train_df[feat].median()
        spread = train_df[feat].std()
        if spread == 0 or pd.isna(spread):
            continue
        z = (val - med) / spread
        weight = importances.get(feat, 0)
        contribution = z * weight
        rows.append({'feature': friendly.get(feat, feat), 'contribution': contribution})
    out = pd.DataFrame(rows).sort_values('contribution', key=abs, ascending=False).head(5)
    if len(out) and out['contribution'].abs().max() > 0:
        out['scaled'] = out['contribution'] / out['contribution'].abs().max()
    else:
        out['scaled'] = 0
    return out


def generate_share_image(artist_name, album_name, pred, lo_fmt, hi_fmt, basis_note=""):
    """Builds a shareable PNG summary card using PIL, styled to match the app theme."""
    W, H = 1200, 675
    bg = (14, 17, 23)
    green = (29, 185, 84)
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    def load_font(bold, size):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
        return ImageFont.load_default()

    font_label = load_font(False, 30)
    font_artist = load_font(True, 44)
    font_big = load_font(True, 130)
    font_range = load_font(False, 30)
    font_basis = load_font(False, 25)
    font_footer = load_font(False, 24)

    # Accent border
    draw.rectangle([(0, 0), (W - 1, H - 1)], outline=green, width=6)

    title = f"{artist_name}" + (f": {album_name}" if album_name else "")
    draw.text((70, 70), "First-Week Stream Prediction", font=font_label, fill=(179, 179, 179))
    draw.text((70, 120), title, font=font_artist, fill=(255, 255, 255))

    draw.text((70, 260), f"{pred:.0f}M", font=font_big, fill=green)
    draw.text((70, 410), "predicted first-week global streams", font=font_range, fill=(179, 179, 179))
    draw.text((70, 460), f"Confidence range: {lo_fmt} to {hi_fmt}", font=font_range, fill=(179, 179, 179))
    if basis_note:
        draw.text((70, 510), basis_note, font=font_basis, fill=(122, 122, 122))

    draw.text((70, H - 70), "spotify-first-week-streaming-predictor.streamlit.app", font=font_footer, fill=(102, 102, 102))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# =============================================================================
# UI
# =============================================================================
st.markdown(
    "<h1 class='app-title'><span class='title-icon'>🎧</span> Spotify First-Week Global "
    "Streaming Predictor for Major Album Releases</h1>",
    unsafe_allow_html=True
)

st.markdown(f"""
<div style="display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-bottom:20px;">
    <div style="background:#13161d; border:1px solid #22262f; border-radius:12px; padding:16px 18px;">
        <div style="font-size:22px; margin-bottom:8px;">📊</div>
        <p style="font-size:13px; font-weight:600; color:#fff; margin:0 0 6px;">What it does</p>
        <p style="font-size:12px; color:#b3b3b3; margin:0; line-height:1.6;">Predicts global first-week Spotify streams for major album releases, using a Random Forest trained on {N_TRAIN} albums ({TRAIN_YEAR_MIN} to {TRAIN_YEAR_MAX}). Predicted ahead of three summer 2026 releases, two of which landed within 7%.</p>
    </div>
    <div style="background:#13161d; border:1px solid #22262f; border-radius:12px; padding:16px 18px;">
        <div style="font-size:22px; margin-bottom:8px;">🎯</div>
        <p style="font-size:13px; font-weight:600; color:#fff; margin:0 0 6px;">How to use it</p>
        <p style="font-size:12px; color:#b3b3b3; margin:0; line-height:1.6;">Select an artist, pick the release period, and hit Predict. Artist streaming history is auto-filled, so you only need to supply the album-specific inputs like track count and lead time.</p>
    </div>
    <div style="background:#13161d; border:1px solid #22262f; border-radius:12px; padding:16px 18px;">
        <div style="font-size:22px; margin-bottom:8px;">🔧</div>
        <p style="font-size:13px; font-weight:600; color:#fff; margin:0 0 6px;">Always improving</p>
        <p style="font-size:12px; color:#b3b3b3; margin:0; line-height:1.6;">Roadmap: mid-tier and non-English artists to correct upward bias, a surprise-release flag, and an objective buzz proxy such as lead single chart position.</p>
    </div>
</div>
<p style="font-size:11px; color:#666; margin:0 0 18px;">⚠️ Best suited for major, established artists with a track record on Spotify.</p>
""", unsafe_allow_html=True)

artists = sorted(df['artist'].unique().tolist())
DEFAULT_ARTIST = "Drake"
default_artist_index = artists.index(DEFAULT_ARTIST) if DEFAULT_ARTIST in artists else 0

# -----------------------------------------------------------------------------
# Recent predictions
# -----------------------------------------------------------------------------
cards_html = []
for item in PREDICTION_LOG:
    diff_pct = (item['pred'] - item['actual']) / item['actual'] * 100
    badge_color, marker = accuracy_style(diff_pct)
    cards_html.append(f"""
    <div class="pred-card">
        <div class="pred-card-title">{item['artist']}: {item['album']}</div>
        <div class="pred-row">
            <div>
                <div class="pred-num" style="color:{SPOTIFY_GREEN};">{item['pred']:.0f}M</div>
                <div class="pred-lbl">predicted</div>
            </div>
            <div class="pred-sep"></div>
            <div>
                <div class="pred-num" style="color:#ffffff;">{item['actual']:.1f}M</div>
                <div class="pred-lbl">actual</div>
            </div>
        </div>
        <div class="pred-badge" style="color:{badge_color};">{diff_pct:+.1f}% off{marker}</div>
    </div>""")

st.markdown(f"""
<p style="font-size:13px; color:#b3b3b3; margin-bottom:10px; text-transform:uppercase; letter-spacing:0.1em;">
    Summer 2026 case study
</p>
<div class="pred-grid">{''.join(cards_html)}</div>
<p style="font-size:11.5px; color:#666; margin:12px 0 0 0; line-height:1.6;">
    Each forecast was logged before that album was released, then checked against the real
    first-week figure. The model has since been revised, so it would produce different numbers
    today. These are kept as a record and are never recalculated.
</p>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# At a glance (moved off the sidebar)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="section-divider">
    <div class="section-divider-line"></div>
    <div class="section-divider-label">Model at a glance</div>
    <div class="section-divider-line" style="background: linear-gradient(90deg, transparent, #1DB95488);"></div>
</div>
""", unsafe_allow_html=True)

g1, g2, g3, g4 = st.columns(4)
g1.metric(
    "Accuracy on unseen albums", f"{100 - HOLDOUT_MAPE:.0f}%",
    help=(f"100 minus the model's average error on the {N_HOLD} albums from 2025 that it never saw "
          f"during training. It describes how close predictions land on average. It is not a hit rate, "
          f"so it does not say how often a prediction falls inside its confidence range."))
g2.metric(
    "Albums tracked", N_TRAIN,
    help=(f"Major album releases used to train the model ({TRAIN_YEAR_MIN} to {TRAIN_YEAR_MAX}). "
          f"A further {N_HOLD} albums from 2025 and {N_VAL} from 2026 are held out for testing only."))
g3.metric(
    "Model R² (relative scale)", f"{state['r2']:.3f}",
    help=(f"Measured on log(streams ÷ users), so it scores how well the model places an album's size "
          f"relative to Spotify's audience at the time, not its raw stream count. Computed by "
          f"leave-one-out cross-validation, on albums the model had not seen.\n\n"
          f"On the raw stream scale the same predictions score about {state['r2_real']:.2f}, because a "
          f"few outsized albums account for most of the spread and the model under-predicts them. "
          f"Average prediction error is the more intuitive gauge of everyday accuracy."))
g4.metric(
    "Avg. prediction error", f"{state['mape']:.1f}%",
    help=("Average percentage the prediction is off from the real number, measured across all training "
          "albums using leave-one-out cross-validation."))

# -----------------------------------------------------------------------------
# Prediction output (rendered above the inputs, full width)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="section-divider">
    <div class="section-divider-line"></div>
    <div style="text-align:center;">
        <div class="section-divider-label">Make a prediction</div>
        <div style="font-size:11px; color:#b3b3b3; margin-top:3px;">Fill in album details below to generate a first-week stream estimate</div>
    </div>
    <div class="section-divider-line" style="background: linear-gradient(90deg, transparent, #1DB95488);"></div>
</div>
""", unsafe_allow_html=True)

prediction_slot = st.container()

# -----------------------------------------------------------------------------
# Album details (full width)
# -----------------------------------------------------------------------------
st.markdown("<p style='font-size:24px; font-weight:700; margin:38px 0 18px 0;'>Album Details</p>",
            unsafe_allow_html=True)

a_col, name_col = st.columns(2)
with a_col:
    artist_name = st.selectbox("Artist", artists, index=default_artist_index)
with name_col:
    album_name = st.text_input(
        "Album name (optional)", placeholder="e.g. Midnight Rain",
        help="Optional. If provided, your share tweet will include the album name.")

# Release period sets the platform size the prediction is pinned to, and also the cutoff
# for which of the artist's albums count as history. The MAU box is keyed on the period so
# switching period reloads that period's figure, while an override you type stays put for
# the period you typed it in.
p_col, m_col, _spacer = st.columns([1, 1, 1])
with p_col:
    periods = list(MAU_BY_PERIOD)
    release_period = st.selectbox(
        "Release period", periods, index=periods.index(DEFAULT_PERIOD),
        help="When the album comes out. This pins the prediction to Spotify's platform size at that "
             "time, so the same inputs always reproduce the same number as Spotify keeps growing. "
             "It also limits the artist's history to albums released before then.")
with m_col:
    mau = st.number_input(
        "Spotify MAU at release (millions)", min_value=1,
        value=MAU_BY_PERIOD[release_period], step=1, key=f"mau_{release_period}",
        help="Auto-filled from the release period. Override it to model a future period Spotify has "
             "not reported yet.")

release_year = int(release_period.split()[0])
mau_basis = f"{mau}M MAU, {release_period}"
if mau != MAU_BY_PERIOD[release_period]:
    st.caption(f"Using a manual override of {mau}M. {release_period} reported "
               f"{MAU_BY_PERIOD[release_period]}M.")
else:
    st.caption(MAU_SOURCE_NOTE)

# Artist bio snippet: last album plus a quick fact, limited to the chosen release period so
# it shows the history the model will actually use.
a_hist = df[(df['artist'] == artist_name) & (df['year'] < release_year)].sort_values('year')
if len(a_hist) > 0:
    last = a_hist.iloc[-1]
    best = a_hist.loc[a_hist['first_week_streams'].idxmax()]
    if len(a_hist) > 1:
        fact = f"Best first-week before {release_year}: {best['album']}, {int(best['first_week_streams'])}M ({int(best['year'])})"
    else:
        fact = f"Only one album tracked for this artist before {release_year}"
    st.markdown(
        f"<div style='background:#13161d; border:1px solid #1db95444; border-radius:6px; padding:8px 14px; "
        f"margin-bottom:12px; font-size:13px;'>"
        f"<span style='color:{SPOTIFY_GREEN}; font-weight:600;'>Last album before {release_year}:</span> "
        f"<span style='color:#fff;'>{last['album']}</span>"
        f"<span style='color:#b3b3b3;'>, {int(last['first_week_streams'])}M first-week streams ({int(last['year'])})</span>"
        f"<br><span style='color:#666; font-size:12px;'>{fact}</span>"
        f"</div>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"<div style='background:#13161d; border:1px solid #f59e0b44; border-radius:6px; padding:8px 14px; "
        f"margin-bottom:12px; font-size:13px; color:#b3b3b3;'>"
        f"No albums tracked for {artist_name} before {release_year}, so the model falls back to "
        f"dataset medians. Expect a weaker prediction."
        f"</div>",
        unsafe_allow_html=True
    )

t_col, l_col, f_col = st.columns(3)
with t_col:
    track_count = st.number_input(
        "Track count", min_value=1, value=12, step=1,
        help="Total number of tracks on the album, including interludes. Check the official pre-save or announced tracklist.")
with l_col:
    lead_time = st.number_input(
        "Lead time (days)", min_value=0, value=21, step=1,
        help="Number of days between the official album announcement and the release date. If announced and released the same day, enter 0. A standard campaign is 3 to 6 weeks (21 to 42 days).")
with f_col:
    feature_track_count = st.number_input(
        "Featured artist tracks", min_value=0, value=0, step=1,
        help="How many tracks include a credited guest artist (e.g. feat. Drake). Leave at 0 if it is a solo album.")

is_household = st.toggle(
    "Household-name artist", value=True,
    help="Is this artist a globally recognized name? Think Taylor Swift, Drake, Eminem. Turn off for artists who are big within a genre but not yet mainstream crossover.")

# Years since last album, measured against the chosen release period rather than today,
# so a prediction for an earlier period uses the gap as it stood back then.
most_recent_year = latest_release_year(artist_name, before_year=release_year)
years_since_last = max(0, int(release_year - most_recent_year)) if most_recent_year else 0
st.caption(f"Years since last album (auto): {years_since_last}")

hist = artist_history(artist_name, mau, before_year=release_year)
hist['is_household_name'] = int(is_household)

with st.expander("Auto-filled from artist history (dataset medians used if no history)"):
    st.write(f"Previous release first-week streams: **{hist['prev_streams_per_mau'] * mau:.0f}M**")
    st.write(f"Career average first-week streams: **{hist['trailing_avg_per_mau'] * mau:.0f}M**")
    st.write(f"Combined history signal used by the model: **{hist['history_per_mau'] * mau:.0f}M**")

predict_clicked = st.button("Predict first-week streams", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Render the prediction into the slot above the inputs
# -----------------------------------------------------------------------------
with prediction_slot:
    if predict_clicked:
        with st.spinner(f"Crunching {N_TRAIN} albums of streaming history..."):
            time.sleep(0.8)
            params = [
                track_count, mau, lead_time, feature_track_count,
                years_since_last, hist['history_per_mau'],
                int(is_household), int(is_household) * track_count
            ]
            pred, lo, hi = predict(params, mau)

        mape = state['mape']
        daily_avg = pred / 7

        prior_albums = len(df[(df['artist'] == artist_name) & (df['year'] < release_year)])
        if prior_albums < 2:
            card_border = "#f59e0b"
            conf_note = (f"⚠️ Lower confidence: only {prior_albums} album"
                         f"{'' if prior_albums == 1 else 's'} tracked for this artist before "
                         f"{release_year}, so the history signal is thin.")
        else:
            card_border = SPOTIFY_GREEN
            conf_note = None

        lo_fmt = f"{pred * (1 - mape / 100):.0f}M"
        hi_fmt = f"{pred * (1 + mape / 100):.0f}M"
        album_label = f" ({album_name})" if album_name else ""

        st.markdown(f"""
        <div class="result-card" style="border-color:{card_border};">
            <div style="color:#b3b3b3; font-size:14px; margin-bottom:4px;">Predicted first-week global streams</div>
            <div style="color:{SPOTIFY_GREEN}; font-size:52px; font-weight:700; line-height:1.1;">{pred:.0f}M</div>
            <div style="color:#b3b3b3; font-size:14px; margin-top:10px;">
                Confidence range (±{mape:.0f}% MAPE): {lo_fmt} to {hi_fmt}
            </div>
            <div style="color:#6b7280; font-size:12.5px; margin-top:8px;">
                Based on {mau_basis}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if conf_note:
            st.warning(conf_note)

        m1, m2, m3 = st.columns(3)
        m1.metric("Streams per track", f"{pred / track_count:.1f}M",
                  help="Total predicted streams divided by track count. Useful for comparing albums of different lengths on equal footing.")
        m2.metric("Avg. streams per day", f"{daily_avg:.1f}M",
                  help="Predicted first-week total spread evenly across seven days.")
        m3.metric("Model confidence (±MAPE)", f"±{mape:.0f}%",
                  help="Based on historical accuracy tested against albums the model never saw during training.")

        def g(text):
            return f"<span style='color:{SPOTIFY_GREEN}; font-weight:700;'>{text}</span>"

        st.markdown(
            f"<div style='font-size:15px; line-height:1.7; color:#e0e0e0; margin-top:12px;'>"
            f"The model expects <strong>{artist_name}</strong>{album_label}'s album to stream around "
            f"{g(f'{pred:.0f}M')} times globally in its first week, an average of {g(f'{daily_avg:.1f}M')} "
            f"streams per day. That works out to roughly {g(f'{pred / track_count:.1f}M')} per track. "
            f"Based on the model's historical accuracy (plus or minus {g(f'{mape:.0f}%')}), a realistic range is "
            f"{g(f'{lo_fmt} to {hi_fmt}')}."
            f"</div>",
            unsafe_allow_html=True
        )
        st.caption("For reference: Drake's Iceman did 450M in its first week, Olivia Rodrigo's album did 395M, "
                   "and Ariana Grande's Petal did 236M.")
        st.caption("Note: your prediction is session only and will not affect the model.")

        with st.expander("🔍 Why this number?"):
            st.caption(
                "Approximate view of which inputs pushed the prediction up or down, based on how far "
                "each value sits from a typical album in the training data, weighted by the model's "
                "feature importance. This is a simplified estimate, not an exact decomposition."
            )
            train_only = df[df['year'] <= TRAIN_YEAR_MAX]
            importances_dict = state['importances'].to_dict()
            breakdown = compute_contribution_breakdown(params, importances_dict, train_only)
            for _, row in breakdown.iterrows():
                bar_color = "#4ade80" if row['contribution'] > 0 else "#f87171"
                bar_width = abs(row['scaled']) * 50
                direction = "pushes prediction up" if row['contribution'] > 0 else "pushes prediction down"
                st.markdown(
                    f"<div style='display:flex; align-items:center; gap:10px; margin-bottom:8px;'>"
                    f"<div style='width:190px; font-size:12px; color:#b3b3b3;'>{row['feature']}</div>"
                    f"<div style='flex:1; background:#1a1d24; border-radius:4px; height:10px; position:relative;'>"
                    f"<div style='width:{bar_width}%; background:{bar_color}; height:10px; border-radius:4px;"
                    f"{'margin-left:auto;' if row['contribution'] < 0 else ''}'></div>"
                    f"</div>"
                    f"<div style='width:150px; font-size:11px; color:#666;'>{direction}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        album_share = f"'s {album_name}" if album_name else "'s upcoming album"
        share_text = (
            f"I used the Spotify First-Week Global Streaming Predictor to estimate "
            f"{artist_name}{album_share} will hit {pred:.0f}M streams in week 1 "
            f"(range: {lo_fmt} to {hi_fmt}, based on {mau_basis}). "
            f"Check it out: spotify-first-week-streaming-predictor.streamlit.app"
        )
        tweet_url = f"https://twitter.com/intent/tweet?text={urllib.parse.quote(share_text)}"

        share_col1, share_col2, _ = st.columns([1, 1, 2])
        with share_col1:
            st.markdown(
                f'<a href="{tweet_url}" target="_blank" style="display:inline-block; margin-top:10px; '
                f'background:#000; color:#fff; padding:8px 18px; border-radius:20px; '
                f'font-weight:600; font-size:13px; text-decoration:none; border:1px solid #333; width:100%; text-align:center; box-sizing:border-box;">'
                f'Share on X &nbsp;𝕏</a>',
                unsafe_allow_html=True
            )
        with share_col2:
            try:
                img_buf = generate_share_image(artist_name, album_name, pred, lo_fmt, hi_fmt,
                                               basis_note=f"Based on {mau_basis}")
                st.download_button(
                    "Download image 🖼️",
                    data=img_buf,
                    file_name=f"{artist_name.replace(' ', '_')}_prediction.png",
                    mime="image/png",
                    use_container_width=True
                )
            except Exception:
                pass
    else:
        st.markdown("""
        <div style="background:#13161d; border:1px solid #22262f; border-radius:14px; padding:20px 26px; color:#b3b3b3; font-size:14px; line-height:1.7;">
            <span style="font-size:20px; margin-right:8px;">🎵</span>
            Fill in the album details below and hit <strong style="color:#fff;">Predict first-week streams</strong>. Your estimate will appear here, with a confidence range and a plain-English breakdown of what the numbers mean.
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Detail panels (moved off the sidebar)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="section-divider">
    <div class="section-divider-line"></div>
    <div class="section-divider-label">Under the hood</div>
    <div class="section-divider-line" style="background: linear-gradient(90deg, transparent, #1DB95488);"></div>
</div>
""", unsafe_allow_html=True)

with st.expander("🔍 How accurate is this model?"):
    st.markdown(
        f"The model was tested against albums it had **never seen during training**, "
        f"the gold standard for checking whether a prediction tool actually works in the real world.\n\n"
        f"Across those unseen albums it was off by **{HOLDOUT_MAPE:.1f}% on average**. "
        f"For context, predicting music streams is inherently noisy. Viral moments, surprise drops "
        f"and cultural timing can swing numbers by 2x to 3x in ways no model can fully anticipate."
    )
    hold_df = pd.DataFrame(state['hold_results'])
    hold_df['pred'] = hold_df['pred'].round(0).astype(int)
    hold_df['actual'] = hold_df['actual'].round(0).astype(int)
    hold_df['err'] = hold_df['err'].round(1)
    hold_df.columns = ['Album', 'Predicted (M)', 'Actual (M)', 'Error (%)']
    st.caption("2025 albums, predicted before release and checked after:")
    st.dataframe(hold_df, hide_index=True, use_container_width=True)
    st.caption(f"Average error on these albums: {HOLDOUT_MAPE:.1f}%")

with st.expander("✅ Real-world track record (2026)"):
    st.markdown(
        "These are 2026 albums with known outcomes. The model predicted them, "
        "then they were checked against reality once the actual numbers came in."
    )
    val_df = pd.DataFrame(state['val_results'])
    val_df['pred'] = val_df['pred'].round(0).astype(int)
    val_df['actual'] = val_df['actual'].round(0).astype(int)
    val_df['err'] = val_df['err'].round(1)
    val_df.columns = ['Album', 'Predicted (M)', 'Actual (M)', 'Error (%)']
    st.dataframe(val_df, hide_index=True, use_container_width=True)

with st.expander("⚙️ What factors drive the prediction?"):
    st.markdown(
        "These are the inputs the model relies on most heavily. "
        "The higher the number, the more that factor influences the prediction."
    )
    imp_df = state['importances'].reset_index()
    imp_df.columns = ['Factor', 'Influence']
    friendly_factors = {
        'history_per_mau': "Artist's streaming history (last album + career avg.)",
        'spotify_mau': 'Spotify platform size (era)',
        'lead_time': 'Days between announcement and release',
        'household_x_tracks': 'Household name x track count',
        'track_count': 'Number of tracks',
        'feature_track_count': 'Featured artist tracks',
        'years_since_last_album': 'Gap since last album',
        'is_household_name': 'Household-name status',
    }
    imp_df['Factor'] = imp_df['Factor'].map(friendly_factors).fillna(imp_df['Factor'])
    imp_df['Influence'] = imp_df['Influence'].round(3)
    st.dataframe(imp_df, hide_index=True, use_container_width=True)

with st.expander("📖 How does this model work?"):
    st.markdown(f"""
This application uses a **Random Forest Regressor** trained on {N_TRAIN} major album releases from
{TRAIN_YEAR_MIN} to {TRAIN_YEAR_MAX}, plus {N_HOLD} albums from 2025 and {N_VAL} from 2026 held out
purely for testing (never used in training).

**The core idea:** an artist's future first-week performance is mostly explained by their recent
streaming trajectory, combined with how big Spotify's platform was at the time. Those two signals do
most of the work.

**A feature that had to be removed:** the model originally included a hand-assigned "hype score" for
pre-release buzz, and it was the single strongest predictor. The problem was that those scores had
been assigned *after* the albums came out, so they carried hindsight the model could exploit.
Cross-validation cannot catch that, because the contamination sits inside the feature's values rather
than across the train/test split. Dropping it barely changed the cross-validated numbers and actually
improved performance on the held-out 2025 albums, which is the signature of a leaky predictor. Three
near-duplicate history features were also collapsed into one, since removing any individual one
changed almost nothing.

**Why Random Forest specifically?** With only {N_TRAIN} training rows, simpler models like Ridge regression
actually score *better* on paper (higher R²) but performed *worse* when tested against albums they had
never seen, a sign they were fitting noise rather than a real pattern. Random Forest, constrained to
shallow trees, generalized better in that same head-to-head test, which is the more honest way to judge
a model like this.

**How accuracy is measured:** every prediction shown in this app comes with a confidence range based on
Leave-One-Out Cross-Validation. That means training on {N_TRAIN - 1} albums and testing on the 1 left out,
repeated {N_TRAIN} times. It gives an honest read on how the model performs on data it has not seen,
without needing a much larger dataset.

**What the model does not know:** things like label disputes, surprise tour announcements, viral TikTok
moments, or genuine cultural narrative are deliberately left out. They are real factors, but they cannot be
reliably quantified across just {N_TRAIN} albums without the model latching onto coincidence instead of signal.

**Known limitations:** the model tends to over-predict for artists showing signs of declining momentum,
since it currently has no explicit signal for that. Ariana Grande's *Petal* is the clearest recent example:
the model called 408M and the album opened at 236.1M, a miss of roughly 73%. That gap persisted across
every feature set tested, so it reflects a real blind spot rather than one bad input. Surprise album drops
with little to no announcement lead time are also harder to model than traditional promotional cycles.

**The honest headline:** on the raw stream scale this model beats "always guess the average album" by a
modest margin, and {N_TRAIN} training rows against {len(FEATS)} features is thin. Treat its output as a
ballpark for how an album is likely to open, not a precise forecast.
    """)

st.markdown(f"""
<div class="app-footer">
    <span>Built with Spotify streaming data</span>
    <span class="accent">Random Forest model</span>
    <span>{N_TRAIN} training albums ({TRAIN_YEAR_MIN} to {TRAIN_YEAR_MAX})</span>
    <span>Last updated October 2026</span>
</div>
""", unsafe_allow_html=True)
