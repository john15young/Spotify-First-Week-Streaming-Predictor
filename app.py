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
    .stApp {{ background-color: #0e1117; }}
    div[data-testid="stMetric"] {{
        background-color: #181b21;
        border: 1px solid #2a2e37;
        border-radius: 10px;
        padding: 14px 16px;
    }}
    div[data-testid="stMetricValue"] {{ color: {SPOTIFY_GREEN}; }}
    .result-card {{
        background: linear-gradient(135deg, #14241a 0%, #181b21 100%);
        border: 1px solid {SPOTIFY_GREEN};
        border-radius: 14px;
        padding: 28px 32px;
        margin: 12px 0 20px 0;
    }}
    .artist-avatar {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: {SPOTIFY_GREEN};
        color: #0e1117;
        font-weight: 700;
        font-size: 16px;
        margin-right: 12px;
        vertical-align: middle;
    }}
    .artist-row {{ display: flex; align-items: center; margin-bottom: 6px; }}
    section[data-testid="stSidebar"] {{ background-color: #14171d; }}
    .stButton button[kind="primary"] {{
        background-color: {SPOTIFY_GREEN};
        border: none;
        color: #0e1117;
        font-weight: 600;
    }}
    .stButton button[kind="primary"]:hover {{
        background-color: #1ed760;
        color: #0e1117;
    }}
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
Bad Bunny,UnVeranoSinTi,2022,3,23,489,10.0,7,0,8,1,0,503,1
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


def predict(params, mau):
    Xi = pd.DataFrame([params], columns=FEATS)
    tp = np.array([np.exp(t.predict(Xi)[0]) * mau for t in model.estimators_])
    return np.mean(tp), np.percentile(tp, 10), np.percentile(tp, 90)


def artist_history(artist, mau):
    """Auto-fill trailing/prev fields from CSV history, falling back to dataset medians."""
    a = df[df['artist'] == artist].sort_values('year')
    is_hh = int(a.iloc[-1]['is_household_name']) if len(a) else 0
    prev = (a.iloc[-1]['first_week_streams'] if len(a) else gm) / mau
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
    st.markdown(f"<div style='font-size:38px; margin-top:4px;'>🎧</div>", unsafe_allow_html=True)
with header_r:
    st.title("Spotify First-Week Global Streaming Predictor for Major Album Releases")
st.caption("Random Forest model trained on 49 albums (2015–2026) · predicts first-week global Spotify streams for an announced album")

artists = sorted(df['artist'].unique().tolist())

with st.sidebar:
    st.markdown("### ⚙️ Global setting")
    mau = st.number_input("Spotify global MAU (millions)", min_value=1, value=DEFAULT_MAU, step=1,
                           help="Spotify's total monthly active users worldwide, used to normalize all streaming figures.")
    st.divider()
    st.markdown("### 📊 At a glance")
    st.metric("Albums in training set", state['train_size'])
    st.metric("Model R²", f"{state['r2']:.3f}")
    st.metric("LOOCV MAPE", f"{state['mape']:.1f}%")

left_col, right_col = st.columns([0.55, 0.45], gap="large")

with left_col:
    st.subheader("Album details")

    artist_choice = st.selectbox("Artist", artists + ["+ New artist"])
    is_new = artist_choice == "+ New artist"

    if is_new:
        artist_name = st.text_input("Artist name")
    else:
        artist_name = artist_choice
        initials = "".join([w[0] for w in artist_name.split()[:2]]).upper()
        st.markdown(
            f"<div class='artist-row'><div class='artist-avatar'>{initials}</div>"
            f"<span style='font-size:15px; color:#b3b3b3;'>Selected artist</span></div>",
            unsafe_allow_html=True
        )

    col1, col2 = st.columns(2)
    with col1:
        track_count = st.number_input("Track count", min_value=1, value=12, step=1)
        feature_track_count = st.number_input("Tracks with guest features", min_value=0, value=0, step=1)
        hype_score = st.slider("Hype score (your judgment, 0–10)", 0.0, 10.0, 9.0, 0.5,
                                help="Subjective buzz rating based on chart activity, single performance, social media, controversy/narrative, etc.")
    with col2:
        lead_time = st.number_input("Lead time (days between announcement and release)", min_value=0, value=21, step=1)
        is_household = st.toggle("Household-name artist", value=True if not is_new else False)

    # Years since last album — only ask directly for new artists; otherwise compute from CSV
    if is_new:
        years_since_last = st.number_input("Years since last album (0 if debut)", min_value=0, value=2, step=1)
    else:
        a = df[df['artist'] == artist_name].sort_values('year')
        years_since_last = int(2026 - a.iloc[-1]['year']) if len(a) else 0
        st.caption(f"Years since last album (auto): {years_since_last}")

    # Auto-filled history fields
    hist = artist_history(artist_name, mau) if artist_name and not is_new else (
        {'prev_streams_per_mau': gm / mau, 'trailing_avg_per_mau': gm / mau,
         'trailing_spt_per_mau': gm_spt / mau, 'is_household_name': int(is_household)}
    )
    if not is_new:
        hist['is_household_name'] = int(is_household)

    with st.expander("Auto-filled from artist history (dataset medians used if no history)"):
        st.write(f"Previous release first-week streams: **{hist['prev_streams_per_mau'] * mau:.0f}M**")
        st.write(f"Trailing average first-week streams: **{hist['trailing_avg_per_mau'] * mau:.0f}M**")
        st.write(f"Trailing streams-per-track: **{hist['trailing_spt_per_mau'] * mau:.1f}M**")

    predict_clicked = st.button("Predict first-week streams", type="primary", use_container_width=True)

with right_col:
    st.subheader("Prediction")
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
                    Range ({mape:.0f}% LOOCV MAPE): {pred * (1 - mape / 100):.0f}M – {pred * (1 + mape / 100):.0f}M
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            c1.metric("Streams per track", f"{pred / track_count:.1f}M")
            c2.metric("Confidence (±MAPE)", f"±{mape:.0f}%")
            st.caption("Prediction is session-only and is not saved back to the training data.")
    else:
        st.info("Fill in album details and click **Predict first-week streams** to see results here.")

st.divider()

# =============================================================================
# MODEL TRANSPARENCY
# =============================================================================
st.subheader("Model transparency")

t1, t2, t3 = st.columns(3)
t1.metric("LOOCV R²", f"{state['r2']:.3f}")
t2.metric("LOOCV MAPE", f"{state['mape']:.1f}%")
t3.metric("Training albums", state['train_size'])

tab1, tab2, tab3 = st.tabs(["2025 holdout", "2026 validation", "Feature importance"])

with tab1:
    hold_df = pd.DataFrame(state['hold_results'])
    hold_df['pred'] = hold_df['pred'].round(0).astype(int)
    hold_df['actual'] = hold_df['actual'].round(0).astype(int)
    hold_df['err'] = hold_df['err'].round(1)
    hold_df.columns = ['Album', 'Predicted (M)', 'Actual (M)', 'Error (%)']
    st.dataframe(hold_df, hide_index=True, use_container_width=True)
    st.caption(f"Holdout MAPE: {np.mean([r['err'] for r in state['hold_results']]):.1f}%")

with tab2:
    val_df = pd.DataFrame(state['val_results'])
    val_df['pred'] = val_df['pred'].round(0).astype(int)
    val_df['actual'] = val_df['actual'].round(0).astype(int)
    val_df['err'] = val_df['err'].round(1)
    val_df.columns = ['Album', 'Predicted (M)', 'Actual (M)', 'Error (%)']
    st.dataframe(val_df, hide_index=True, use_container_width=True)

with tab3:
    imp_df = state['importances'].reset_index()
    imp_df.columns = ['Feature', 'Importance']
    imp_df['Importance'] = imp_df['Importance'].round(3)
    st.dataframe(imp_df, hide_index=True, use_container_width=True)
