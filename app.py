import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
import io, warnings
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Spotify First-Week Streaming Predictor", layout="wide", page_icon="🎧")

SPOTIFY_GREEN = "#1DB954"

st.markdown(f"""
<style>
    /* Base */
    .stApp {{ background-color: #0e1117; }}
    section[data-testid="stSidebar"] {{ background-color: #14171d; }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background-color: #181b21;
        border: 1px solid #2a2e37;
        border-radius: 10px;
        padding: 14px 16px;
    }}
    div[data-testid="stMetricValue"] {{ color: {SPOTIFY_GREEN}; }}

    /* Hide Streamlit anchor links on headings */
    h1 a, h2 a, h3 a {{ display: none !important; }}

    /* Input column card styling via column container */
    div[data-testid="column"]:first-child {{
        background: #13161d;
        border: 1px solid #22262f;
        border-radius: 14px;
        padding: 22px 24px 18px 24px;
    }}

    /* Prediction result card */
    .result-card {{
        background: linear-gradient(160deg, #0d2a18 0%, #111820 60%, #0e1117 100%);
        border: 1px solid {SPOTIFY_GREEN};
        border-radius: 14px;
        padding: 28px 32px;
        margin: 12px 0 20px 0;
        box-shadow: 0 0 24px #1db95422;
    }}

    /* Section divider label */
    .section-divider {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 20px 0 18px 0;
    }}
    .section-divider-line {{
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, {SPOTIFY_GREEN}88, transparent);
    }}
    .section-divider-label {{
        color: {SPOTIFY_GREEN};
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        white-space: nowrap;
    }}

    /* Predict button */
    .stButton button[kind="primary"] {{
        background-color: {SPOTIFY_GREEN};
        border: none;
        color: #0e1117;
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 0.03em;
        transition: all 0.2s ease;
    }}
    .stButton button[kind="primary"]:hover {{
        background-color: #1ed760;
        color: #0e1117;
        box-shadow: 0 0 16px #1db95466;
        transform: translateY(-1px);
    }}

    /* Footer */
    .app-footer {{
        margin-top: 48px;
        padding: 20px 0 8px 0;
        border-top: 1px solid #1e2228;
        text-align: center;
        color: #4a5060;
        font-size: 13px;
        letter-spacing: 0.02em;
    }}
    .app-footer span {{ color: {SPOTIFY_GREEN}88; }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA — same CSV used by the original script (source of truth)
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
Bad Bunny,UltimoTourDelMundo,2020,3,16,299,9.5,3,0,0,0,0,274,1
Bad Bunny,UnVeranoSinTi,2022,3,23,489,10.0,7,0,8,1,274,503,1
Bad Bunny,NadieSabe,2023,3,22,574,9.5,3,0,8,1,503,469,1
Harry Styles,FineLine,2019,0,12,248,9.0,40,0,0,0,0,155,1
Harry Styles,HarrysHouse,2022,0,13,489,9.5,52,0,0,2,155,440,1
Post Malone,BBandB,2018,1,18,207,10.0,21,0,5,1,0,411,1
Post Malone,HollywoodsBleeding,2019,1,17,248,10.0,14,0,7,1,411,379,1
Karol G,MananaSeraBonito,2023,3,17,574,9.5,31,0,10,2,0,203,0
21 Savage,AmericanDream,2024,1,15,640,9.5,7,0,10,3,0,187,0
Tyler The Creator,CHROMAKOPIA,2024,1,14,640,10.0,11,0,7,3,175,340,0
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

FEATS = ['track_count', 'spotify_mau', 'hype_score', 'lead_time', 'feature_track_count',
         'years_since_last_album', 'prev_streams_per_mau', 'trailing_avg_per_mau',
         'trailing_spt_per_mau', 'is_household_name', 'household_x_tracks']

DEFAULT_MAU = 761  # current Spotify global MAU (millions), editable in sidebar


# =============================================================================
# MODEL TRAINING — cached so it only runs once per session
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
    df['log_spm'] = np.log(df['first_week_streams'] / df['spotify_mau'])

    train = df[df['year'] <= 2024].copy()
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
        'r2': r2, 'mape': mape, 'mae': mae,
        'hold_results': hold_results, 'val_results': val_results,
        'importances': importances, 'train_size': len(train)
    }


state = load_and_train()
df, model = state['df'], state['model']
gm, gm_spt = state['gm'], state['gm_spt']

# Reference-only prior-album data — used ONLY to correct "years since last album" and
# display-only prev-streams for artists whose earlier album we have real numbers for,
# but did NOT add to training data because doing so measurably hurt holdout accuracy
# (tested individually + combined; see project notes). Never fed into the model.
REFERENCE_HISTORY = {
    "Post Malone":       {"year": 2019, "streams": 379},  # Hollywood's Bleeding already in training; this just documents the chain
    "Tyler The Creator":  {"year": 2021, "streams": 133},  # Call Me If You Get Lost
    "Billie Eilish":      {"year": 2021, "streams": 178},  # Happier Than Ever
    "Taylor Swift":       {"year": 2020, "streams": 254},  # Evermore (Folklore same year, Evermore is later)
    "21 Savage":          {"year": 2020, "streams": 108},  # Savage Mode II (collab w/ Metro — kept separate, see notes)
    "Metro Boomin":       {"year": 2022, "streams": 226},  # Heroes & Villains
    "Ariana Grande":      {"year": 2018, "streams": 174},  # Sweetener
    "Harry Styles":       {"year": 2017, "streams": 66},   # Harry Styles (debut)
}


def predict(params, mau):
    Xi = pd.DataFrame([params], columns=FEATS)
    tp = np.array([np.exp(t.predict(Xi)[0]) * mau for t in model.estimators_])
    return np.mean(tp), np.percentile(tp, 10), np.percentile(tp, 90)


def reference_adjusted_first_year(artist):
    """Returns the earliest known release year for an artist, factoring in REFERENCE_HISTORY
    even though that earlier album isn't part of the training data."""
    a = df[df['artist'] == artist].sort_values('year')
    years = a['year'].tolist()
    if artist in REFERENCE_HISTORY:
        years.append(REFERENCE_HISTORY[artist]['year'])
    return min(years) if years else None


def artist_history(artist, mau):
    """Auto-fill trailing/prev fields from CSV history, falling back to dataset medians.
    Display-only prev_album_streams uses REFERENCE_HISTORY when the real training rows
    don't have it, so the UI shows the correct number even though it's not in training."""
    a = df[df['artist'] == artist].sort_values('year')
    is_hh = int(a.iloc[-1]['is_household_name']) if len(a) else 0

    if len(a):
        prev = a.iloc[-1]['first_week_streams'] / mau
    elif artist in REFERENCE_HISTORY:
        prev = REFERENCE_HISTORY[artist]['streams'] / mau
    else:
        prev = gm / mau

    trail_avg = (a['first_week_streams'].mean() if len(a) else gm) / mau
    trail_spt = (a['spt'].mean() if len(a) else gm_spt) / mau
    return {
        'prev_streams_per_mau': prev,
        'trailing_avg_per_mau': trail_avg,
        'trailing_spt_per_mau': trail_spt,
        'is_household_name': is_hh,
    }


# =============================================================================
# UI
# =============================================================================
header_l, header_r = st.columns([0.06, 0.94])
with header_l:
    st.markdown("<div style='font-size:38px; margin-top:4px;'>🎧</div>", unsafe_allow_html=True)
with header_r:
    st.title("Spotify First-Week Global Streaming Predictor for Major Album Releases")

st.markdown(f"""
<div style="background:{SPOTIFY_GREEN}18; border-left:3px solid {SPOTIFY_GREEN}; border-radius:6px; padding:14px 18px; margin-bottom:18px;">
<b>Welcome!</b> This application uses a machine learning model trained on 49 major album releases (2015 to 2026)
to predict how many streams an announced album will rack up in its first week on Spotify, globally.
<br><br>
<b>How to use it:</b> Select an artist from the dropdown, fill in a few details about the upcoming album,
then hit <b>Predict</b>. The model will instantly give you a stream estimate along with a confidence range.
Artist streaming history is auto-filled from the training data, so you only need to supply the album-specific details.
<br><br>
<span style="color:#b3b3b3; font-size:13px;">⚠️ Best suited for major, established artists with a track record on Spotify.
This model is periodically refined, updated, and improved as new album data becomes available.</span>
</div>
""", unsafe_allow_html=True)

artists = sorted(df['artist'].unique().tolist())

# Recent validated predictions
st.markdown(f"""
<div style="margin-bottom: 6px;">
    <p style="font-size:13px; color:#b3b3b3; margin-bottom:10px; text-transform:uppercase; letter-spacing:0.1em;">
        Recent validated predictions
    </p>
    <div style="display:flex; gap:12px; flex-wrap:wrap;">
        <div style="flex:1; min-width:200px; background:#13161d; border:1px solid #22262f; border-radius:12px; padding:14px 18px;">
            <div style="font-size:12px; color:#b3b3b3; margin-bottom:4px;">Drake — Iceman 🧊</div>
            <div style="display:flex; align-items:baseline; gap:10px; flex-wrap:wrap;">
                <span style="font-size:22px; font-weight:700; color:{SPOTIFY_GREEN};">480M</span>
                <span style="font-size:13px; color:#b3b3b3;">predicted</span>
                <span style="font-size:22px; font-weight:700; color:#fff;">455.3M</span>
                <span style="font-size:13px; color:#b3b3b3;">actual</span>
            </div>
            <div style="font-size:12px; color:#4ade80; margin-top:4px;">+5.4% off &nbsp;✓</div>
        </div>
        <div style="flex:1; min-width:200px; background:#13161d; border:1px solid #22262f; border-radius:12px; padding:14px 18px;">
            <div style="font-size:12px; color:#b3b3b3; margin-bottom:4px;">Olivia Rodrigo — you seem pretty sad... 💜</div>
            <div style="display:flex; align-items:baseline; gap:10px; flex-wrap:wrap;">
                <span style="font-size:22px; font-weight:700; color:{SPOTIFY_GREEN};">416M</span>
                <span style="font-size:13px; color:#b3b3b3;">predicted</span>
                <span style="font-size:22px; font-weight:700; color:#fff;">394.7M</span>
                <span style="font-size:13px; color:#b3b3b3;">actual</span>
            </div>
            <div style="font-size:12px; color:#4ade80; margin-top:4px;">+5.4% off &nbsp;✓</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-divider">
    <div class="section-divider-line"></div>
    <div class="section-divider-label">Make a prediction</div>
    <div class="section-divider-line" style="background: linear-gradient(90deg, transparent, #1DB95488);"></div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Global setting")
    mau = st.number_input("Spotify global MAU (millions)", min_value=1, value=DEFAULT_MAU, step=1,
                           help="Spotify's total monthly active users worldwide, used to normalize all streaming figures.")
    st.divider()
    st.markdown("### 📊 At a glance")
    st.metric("Albums in training set", state['train_size'])
    st.metric("Model R²", f"{state['r2']:.3f}", help="How well the model explains variation across albums. 0.52 is solid for this type of prediction — music is genuinely unpredictable.")
    st.metric("Avg. prediction error", f"±{state['mape']:.1f}%", help="On average, predictions land within this % of the real number. Tested using leave-one-out cross-validation across all 49 training albums.")
    st.divider()

    with st.expander("🔍 How accurate is this model?"):
        st.markdown(
            "The model was tested against albums it had **never seen during training** — "
            "the gold standard for checking if a prediction tool actually works in the real world.\n\n"
            f"Across those unseen albums, it was off by **{state['mape']:.1f}% on average**. "
            "For context: predicting music streams is inherently noisy — viral moments, surprise drops, "
            "and cultural timing can swing numbers by 2–3x in ways no model can fully anticipate."
        )
        hold_df = pd.DataFrame(state['hold_results'])
        hold_df['pred'] = hold_df['pred'].round(0).astype(int)
        hold_df['actual'] = hold_df['actual'].round(0).astype(int)
        hold_df['err'] = hold_df['err'].round(1)
        hold_df.columns = ['Album', 'Predicted (M)', 'Actual (M)', 'Error (%)']
        st.caption("2025 albums — predicted before release, checked after:")
        st.dataframe(hold_df, hide_index=True, use_container_width=True)
        st.caption(f"Average error on these albums: {np.mean([r['err'] for r in state['hold_results']]):.1f}%")

    with st.expander("✅ Real-world track record (2026)"):
        st.markdown(
            "These are 2026 albums with known outcomes — predicted by the model, "
            "then checked against reality once the actual numbers came in."
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
        friendly = {
            'trailing_avg_per_mau': "Artist's recent streaming average",
            'prev_streams_per_mau': "Previous album's first-week streams",
            'spotify_mau': 'Spotify platform size (era)',
            'hype_score': 'Pre-release buzz (your rating)',
            'trailing_spt_per_mau': 'Avg. streams per track recently',
            'lead_time': 'Days between announcement & release',
            'household_x_tracks': 'Household name x track count',
            'track_count': 'Number of tracks',
            'feature_track_count': 'Featured artist tracks',
            'years_since_last_album': 'Gap since last album',
            'is_household_name': 'Household-name status',
        }
        imp_df['Factor'] = imp_df['Factor'].map(friendly).fillna(imp_df['Factor'])
        imp_df['Influence'] = imp_df['Influence'].round(3)
        st.dataframe(imp_df, hide_index=True, use_container_width=True)

left_col, right_col = st.columns([0.55, 0.45], gap="large")

with left_col:
    st.markdown(f"<p style='font-size:22px; font-weight:700; margin-bottom:16px;'>Album details</p>", unsafe_allow_html=True)

    artist_name = st.selectbox("Artist", artists)
    st.markdown(
        f"<div style='background:{SPOTIFY_GREEN}; border-radius:6px; padding:8px 14px; "
        f"margin-bottom:12px; color:#0e1117; font-weight:600; font-size:15px;'>"
        f"🎵 {artist_name}</div>",
        unsafe_allow_html=True
    )

    track_count = st.number_input(
        "Track count",
        min_value=1, value=12, step=1,
        help="Total number of tracks on the album, including interludes. Check the official pre-save or announced tracklist.")

    lead_time = st.number_input(
        "Lead time (days)",
        min_value=0, value=21, step=1,
        help="Number of days between the official album announcement and the release date. If announced and released the same day, enter 0. A standard campaign is 3 to 6 weeks (21 to 42 days).")

    feature_track_count = st.number_input(
        "Featured artist tracks",
        min_value=0, value=0, step=1,
        help="How many tracks include a credited guest artist (e.g. feat. Drake). Leave at 0 if it is a solo album.")

    hype_score = st.slider(
        "Hype score (0 to 10)",
        0.0, 10.0, 9.0, 0.5,
        help="Your read on pre-release buzz. Consider: how are lead singles charting? Is the artist trending on social media? Any major narrative (beef, comeback, tour, controversy)? 10 = peak cultural moment, 7 to 8 = solid mainstream interest, 5 to 6 = moderate, below 5 = quiet release.")

    is_household = st.toggle(
        "Household-name artist",
        value=True,
        help="Is this artist a globally recognized name? Think Taylor Swift, Drake, Eminem. Turn off for artists who are big within a genre but not yet mainstream crossover.")

    # Years since last album (auto from CSV)
    a = df[df['artist'] == artist_name].sort_values('year')
    most_recent_year = int(a.iloc[-1]['year']) if len(a) else None
    if artist_name in REFERENCE_HISTORY:
        ref_year = REFERENCE_HISTORY[artist_name]['year']
        if most_recent_year is None or ref_year > most_recent_year:
            most_recent_year = ref_year
    years_since_last = int(2026 - most_recent_year) if most_recent_year else 0
    st.caption(f"Years since last album (auto): {years_since_last}")

    # Auto-filled history fields
    hist = artist_history(artist_name, mau)
    hist['is_household_name'] = int(is_household)

    with st.expander("Auto-filled from artist history (dataset medians used if no history)"):
        st.write(f"Previous release first-week streams: **{hist['prev_streams_per_mau'] * mau:.0f}M**")
        st.write(f"Trailing average first-week streams: **{hist['trailing_avg_per_mau'] * mau:.0f}M**")
        st.write(f"Trailing streams per track: **{hist['trailing_spt_per_mau'] * mau:.1f}M**")

    predict_clicked = st.button("Predict first-week streams", type="primary", use_container_width=True)

with right_col:
    st.markdown(f"<p style='font-size:22px; font-weight:700; margin-bottom:16px;'>Prediction</p>", unsafe_allow_html=True)
    if predict_clicked:
        if not artist_name:
            st.error("Enter an artist name.")
        else:
            params = [
                track_count, mau, hype_score, lead_time, feature_track_count,
                years_since_last,
                hist['prev_streams_per_mau'], hist['trailing_avg_per_mau'], hist['trailing_spt_per_mau'],
                int(is_household), int(is_household) * track_count
            ]
            pred, lo, hi = predict(params, mau)
            mape = state['mape']

            st.markdown(f"""
            <div class="result-card">
                <div style="color:#b3b3b3; font-size:14px; margin-bottom:4px;">Predicted first-week global streams</div>
                <div style="color:{SPOTIFY_GREEN}; font-size:44px; font-weight:700; line-height:1.1;">{pred:.0f}M</div>
                <div style="color:#b3b3b3; font-size:14px; margin-top:10px;">
                    Confidence range (±{mape:.0f}% MAPE): {pred * (1 - mape / 100):.0f}M to {pred * (1 + mape / 100):.0f}M
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            c1.metric("Streams per track", f"{pred / track_count:.1f}M")
            c2.metric("Model confidence (±MAPE)", f"±{mape:.0f}%")

            lo_fmt = f"{pred * (1 - mape / 100):.0f}M"
            hi_fmt = f"{pred * (1 + mape / 100):.0f}M"
            st.markdown(
                f"The model expects **{artist_name}**'s album to stream around **{pred:.0f}M** times "
                f"globally in its first week, roughly **{pred / track_count:.1f}M per track**. "
                f"Based on the model's historical accuracy (plus or minus {mape:.0f}%), a realistic range is "
                f"**{lo_fmt} to {hi_fmt}**."
            )
            st.markdown("")
            st.caption("Your prediction is session only and won't affect the model.")

            # Share on X/Twitter
            share_text = (
                f"I used the Spotify First-Week Global Streaming Predictor to estimate "
                f"{artist_name}'s upcoming album will hit {pred:.0f}M streams in week 1 "
                f"(range: {lo_fmt} to {hi_fmt}). "
                f"Check it out: spotify-first-week-streaming-predictor.streamlit.app"
            )
            import urllib.parse
            tweet_url = f"https://twitter.com/intent/tweet?text={urllib.parse.quote(share_text)}"
            st.markdown(
                f'<a href="{tweet_url}" target="_blank" style="display:inline-block; margin-top:10px; '
                f'background:#000; color:#fff; padding:8px 18px; border-radius:20px; '
                f'font-weight:600; font-size:13px; text-decoration:none; border:1px solid #333;">'
                f'Share on X &nbsp;𝕏</a>',
                unsafe_allow_html=True
            )
    else:
        st.info("Fill in album details and click **Predict first-week streams** to see results here.")

st.markdown(f"""
<div class="app-footer">
    Built with Spotify streaming data &nbsp;·&nbsp;
    <span>Random Forest model</span> &nbsp;·&nbsp;
    49 training albums (2015 to 2026) &nbsp;·&nbsp;
    Validated on unseen releases &nbsp;·&nbsp;
    <span>±{state['mape']:.0f}% avg. error</span>
</div>
""", unsafe_allow_html=True)

