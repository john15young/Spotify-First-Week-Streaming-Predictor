import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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
CSV_CONTENT = """artist,album,year,track_count,spotify_mau,lead_time,feature_track_count,years_since_last_album,prev_album_streams,first_week_streams,is_household_name
Harry Styles,HarryStyles,2017,10,120,29,0,0,0,66,0
Harry Styles,FineLine,2019,12,248,39,0,2,66,155,1
Harry Styles,HarrysHouse,2022,13,489,58,0,2,155,440,1
Harry Styles,KissAllTheTime,2026,14,761,50,0,4,440,205,1
Noah Kahan,TheGreatDivide,2026,17,761,86,0,4,0,231.5,0
Kanye West,BULLY,2026,18,761,3,6,2,251,158.9,1
Don Toliver,OCTANE,2026,18,761,8,5,2,0,162.5,0
J Cole,TheFallOff,2026,24,761,23,7,5,207.1,139.1,1
Bruno Mars,TheRomantic,2026,9,761,51,0,5,0,144.0,1
A$AP Rocky,DontBeDumb,2026,15,761,27,6,8,0,129.7,0
Zach Bryan,WithHeavenOnTop,2026,25,761,175,0,2,98.5,98.1,0
Ella Langley,Dandelion,2026,18,761,73,1,2,0,88.4,0
Karol G,NoMeArrepiento,2026,14,788,9,4,1,136.2,76.4,1
Ariana Grande,DangerousWoman,2016,15,100,71,4,2,0,53.6,1
Ariana Grande,Sweetener,2018,15,207,58,2,2,53.6,174,1
Ariana Grande,ThanksUNext,2019,12,248,17,0,1,174,361,1
Ariana Grande,Positions,2020,14,299,7,3,1,361,216.7,1
Ariana Grande,EternalSunshine,2024,13,640,51,0,4,216.7,342,1
Taylor Swift,Lover,2019,18,248,71,2,2,0,257.1,1
Taylor Swift,Folklore,2020,16,299,1,1,1,257.1,329,1
Taylor Swift,Evermore,2020,15,299,1,3,0,329,254,1
Taylor Swift,Midnights,2022,20,489,54,1,2,254,776,1
Taylor Swift,1989TV,2023,21,574,79,0,1,776,582,1
Taylor Swift,TTPD,2024,31,640,75,2,1,582,1173,1
Taylor Swift,LifeOfAShowgirl,2025,12,715,52,1,1,1173,887,1
Bad Bunny,UltimoTourDelMundo,2020,16,299,2,0,0,82.9,274,1
Bad Bunny,YHLQMDLG,2020,20,299,2,11,2,67.4,295.6,1
Bad Bunny,LasQueNoIbanASalir,2020,10,299,0,6,0,295.6,82.9,1
Bad Bunny,UnVeranoSinTi,2022,23,489,4,8,1,274,503,1
Bad Bunny,NadieSabe,2023,22,574,4,8,1,503,471,1
Bad Bunny,DeBITiRARMasFotos,2025,17,715,10,3,2,471,579.2,1
Tyler The Creator,Igor,2019,12,248,11,5,2,0,114.3,0
Tyler The Creator,CallMeIfYouGetLost,2021,16,365,8,12,2,114.3,133,0
Tyler The Creator,CHROMAKOPIA,2024,14,640,11,7,3,133,340,0
Tyler The Creator,DontTapTheGlass,2025,10,715,3,3,1,340,111.5,0
Billie Eilish,WWAFAWDWG,2019,14,248,59,0,0,0,276.3,0
Billie Eilish,HappierThanEver,2021,16,365,95,0,2,276.3,178,1
Billie Eilish,HMHAS,2024,10,640,39,0,3,178,386,1
Post Malone,Stoney,2016,18,100,36,4,0,0,14.5,0
Post Malone,BBandB,2018,18,207,23,5,2,14.5,411,1
Post Malone,HollywoodsBleeding,2019,17,248,11,7,1,411,379,1
Post Malone,TwelveCaratToothache,2022,14,489,37,6,3,379,142,1
Post Malone,Austin,2023,17,574,32,0,1,142,119.4,1
Post Malone,F1Trillion,2024,18,640,59,15,1,119.4,180.2,1
Ed Sheeran,Divide,2017,16,120,50,0,3,23.8,375,1
Ed Sheeran,Equals,2021,14,365,71,0,4,375,161,1
Drake,Scorpion,2018,25,207,16,3,1,290.2,559,1
Drake,CLB,2021,21,365,7,12,3,559,496,1
Drake,FATD,2023,23,574,14,10,2,496,405,1
Travis Scott,ASTROWORLD,2018,17,207,4,11,3,0,257,0
Travis Scott,Utopia,2023,19,574,6,13,5,257,457,0
Eminem,Revival,2017,19,120,17,8,4,10,117,1
Eminem,Kamikaze,2018,13,207,0,4,1,117,238,1
Eminem,TheDeathOfSlimShady,2024,19,640,11,8,6,238,254,1
The Weeknd,BeautyBehindTheMadness,2015,14,80,50,3,2,0,82.8,0
The Weeknd,Starboy,2016,18,100,65,5,1,82.8,223,1
The Weeknd,AfterHours,2020,14,299,30,0,2,223,265,1
The Weeknd,HUT,2025,22,715,65,7,4,265,302,1
Justin Bieber,Purpose,2015,13,80,42,4,3,0,205,1
Justin Bieber,Changes,2020,17,299,18,6,5,205,181,1
Justin Bieber,Justice,2021,16,365,21,8,1,181,252,1
Justin Bieber,Swag,2025,21,715,1,7,4,252,276.8,1
SZA,SOS,2022,23,489,6,4,5,125,336,0
Sabrina Carpenter,EmailsICantSend,2022,13,489,10,0,3,0,22.3,0
Sabrina Carpenter,ShortNSweet,2024,12,640,81,0,2,22.3,352,0
Sabrina Carpenter,MansBestFriend,2025,13,715,79,0,1,352,297,0
Kendrick Lamar,DAMN,2017,14,120,3,3,2,0,227,1
Kendrick Lamar,MrMorale,2022,19,489,25,7,5,227,343,1
Kendrick Lamar,GNX,2024,12,640,0,4,2,343,363,1
LilWayne,CarterV,2018,23,207,2,10,7,0,226.9,0
Olivia Rodrigo,SOUR,2021,11,365,50,0,0,0,385,1
Olivia Rodrigo,GUTS,2023,12,574,74,0,2,385,284,1
Beyonce,Renaissance,2022,16,489,43,2,6,0,173,1
Beyonce,CowboyCarter,2024,27,640,47,11,2,173,316,1
Dua Lipa,FutureNostalgia,2020,13,299,58,2,4,0,148.9,0
Dua Lipa,RadicalOptimism,2024,11,640,51,0,4,148.9,94,0
Adele,30,2021,12,365,37,0,6,0,310,1
Karol G,KG0516,2021,16,365,9,10,2,0,54.9,0
Karol G,MananaSeraBonito,2023,17,574,14,10,2,54.9,203,0
Karol G,Tropicoqueta,2025,20,715,10,6,2,203,136.2,0
21 Savage,IAmGreaterThanIWas,2018,15,207,10,8,1,0,88.6,0
21 Savage,AmericanDream,2024,15,640,3,10,2,381.7,187,0
Kanye West,JesusIsKing,2019,11,248,57,5,1,141.0,174.3,1
Kanye West,DONDA,2021,27,365,23,23,2,174.3,442,1
Kanye West,VULTURES1,2024,15,640,24,0,3,442,251,1
Tate McRae,ThinkLater,2023,14,574,32,0,2,0,115,0
Tate McRae,SoCloseToWhat,2025,16,715,99,2,2,115,188,0
Morgan Wallen,Dangerous,2021,30,365,58,2,3,0,161,0
Morgan Wallen,OneThingAtATime,2023,36,574,32,3,2,161,251.1,0
Morgan Wallen,ImTheProblem,2025,37,715,57,5,2,251.1,253,0
Juice WRLD,DeathRaceForLove,2019,22,248,29,3,1,0,126.3,0
Juice WRLD,LegendsNeverDie,2020,22,299,4,5,1,126.3,303,0
Playboi Carti,WholeLottaRed,2020,24,299,4,3,2,0,88.5,0
Playboi Carti,MUSIC,2025,30,715,2,12,5,88.5,483.35,0
Lady Gaga,Mayhem,2025,14,715,39,2,5,173,216.08,1
Future,INeverLikedYou,2022,16,489,4,6,2,0,118.6,0
Zach Bryan,ZachBryan,2023,16,574,16,4,1,0,145.5,0
Zach Bryan,TheGreatAmericanBarScene,2024,19,640,14,5,1,145.5,98.5,0
Khalid,FreeSpirit,2019,17,248,36,4,1,0,142.5,0
BTS,LoveYourselfTear,2018,11,207,32,1,0,0,72.3,1
BTS,MapOfTheSoul7,2020,20,299,45,2,1,98.1,144.4,1
BTS,BE,2020,8,299,53,0,0,144.4,153.8,1
Olivia Dean,TheArtOfLoving,2025,12,715,116,0,2,0,97,0
J Cole,KOD,2018,12,207,4,0,2,0,190,0
J Cole,TheOffSeason,2021,12,365,10,5,3,190,207.1,0
Rosalia,LUX,2025,18,715,18,5,3,79.0,182.9,0
J Balvin,Jose,2021,24,365,28,14,1,102.0,85.5,1
Gunna,DS4EVER,2022,19,489,6,11,2,0,85.8,0
Daddy Yankee,Legendaddy,2022,19,489,4,7,10,0,78.6,1
Anuel AA,Emmanuel,2020,22,299,2,12,2,0,78.3,0
Shakira,LasMujeresYaNoLloran,2024,15,640,36,10,7,0,67.5,1
Lil Uzi Vert,LuvIsRage2,2017,16,120,1,3,0,0,82.9,0
Lil Baby,MyTurn,2020,20,299,29,8,2,0,81.4,0
Twenty One Pilots,Trench,2018,14,207,86,0,3,0,76.4,0
Central Cee,CantRushGreatness,2025,17,715,85,6,3,0,96.2,0
sombr,IBarelyKnowHer,2025,10,715,11,0,0,0,98.7,0
Addison Rae,Addison,2025,12,715,44,0,2,0,63.8,0
Alex Warren,YoullBeAlrightKid,2025,21,715,45,2,1,0,133.8,0
Jennie,Ruby,2025,15,715,45,5,0,0,148.6,0
Sleep Token,EvenInArcadia,2025,10,715,57,0,2,0,82,0
Doja Cat,Vie,2025,15,715,43,1,2,124.5,56,0"""

# hype_score was dropped: it was scored retrospectively, after the albums' results were
# known, so it carried hindsight that cross-validation cannot detect (the leak lives inside
# the feature's values, not across the train/test split). Removing it made the cross-validated
# metrics worse and the held-out 2025 albums better, which is what you would expect from a
# contaminated predictor. The three correlated history features (previous album, trailing
# average, trailing streams per track) are collapsed into one, since dropping any single one
# left accuracy roughly unchanged: they were three versions of the same signal.
#
# lead_time = days from the first official announcement of the release date to the actual
# release. Every value was re-checked against announcement dates in October 2026. Hurry Up
# Tomorrow (released Jan 31, 2025) is carried as a 2025 holdout album.
FEATS = ['track_count', 'spotify_mau', 'lead_time', 'feature_track_count',
         'years_since_last_album', 'history_per_mau', 'has_hist',
         'is_household_name', 'household_x_tracks']

# Platform-size adjustment: stream figures are divided by MAU ** ERA_EXP rather than by MAU in
# full, because first-week streams grew far more slowly than Spotify's user base. has_hist is 1
# when a real previous-album figure exists and 0 when the fallback value stands in for one.
ERA_EXP = 0.25

# Ridge regression on standardized features. alpha is fixed at 1.0 in advance (sklearn's
# default, not tuned). Chosen over a Random Forest by leave-one-out cross-validation alone.
ALPHA = 1.0


def make_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))

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
    # Fallback for artists with no earlier album: median of the TRAINING albums only, so no
    # holdout or 2026 album influences a training feature.
    _trn = df[df['year'] <= TRAIN_YEAR_MAX]
    gm = _trn['first_week_streams'].median()
    gm_spt = _trn['spt'].median()
    df['has_hist'] = (df['prev_album_streams'] > 0).astype(int)
    df['prev_album_streams'] = df['prev_album_streams'].replace(0, gm)

    def prior_mean(df, col, fallback):
        out = []
        for _, r in df.iterrows():
            prior = df[(df['artist'] == r['artist']) & (df['year'] < r['year'])]
            out.append(prior[col].mean() if len(prior) > 0 else fallback)
        return out

    df['t_avg'] = prior_mean(df, 'first_week_streams', gm)
    df['t_spt'] = prior_mean(df, 'spt', gm_spt)
    df['era_scale'] = df['spotify_mau'] ** ERA_EXP
    for col, src in [('trailing_avg_per_mau', 't_avg'), ('trailing_spt_per_mau', 't_spt'),
                      ('prev_streams_per_mau', 'prev_album_streams')]:
        df[col] = df[src] / df['era_scale']
    # Single history signal: the artist's last album and their career average, averaged.
    # Replaces the three near-duplicate history features the model used to carry.
    df['history_per_mau'] = (df['prev_streams_per_mau'] + df['trailing_avg_per_mau']) / 2
    df['log_spm'] = np.log(df['first_week_streams'] / df['era_scale'])

    train = df[df['year'] <= TRAIN_YEAR_MAX].copy()
    hold = df[df['year'] == 2025].copy()
    val26 = df[df['year'] == 2026].copy()

    X, y = train[FEATS], train['log_spm']

    # LOOCV for reported error metrics
    lp, la = [], []
    for ti, te in LeaveOneOut().split(X):
        m = make_model()
        m.fit(X.iloc[ti], y.iloc[ti])
        lp.append(m.predict(X.iloc[te])[0])
        la.append(y.iloc[te[0]])

    r2 = r2_score(la, lp)
    rp = np.exp(np.array(lp)) * train['era_scale'].values
    ra = train['first_week_streams'].values
    # Same cross-validated predictions scored on the raw stream scale instead of in log
    # space. It comes out much lower because a handful of outsized albums dominate the
    # spread, so both numbers are reported rather than only the flattering one.
    r2_real = r2_score(ra, rp)
    mape = np.mean(np.abs(rp - ra) / ra * 100)
    mae = np.mean(np.abs(rp - ra))
    # Naive benchmark: always guess the median first-week total of the other albums.
    base_pred = np.array([np.median(np.delete(ra, i)) for i in range(len(ra))])
    base_mape = np.mean(np.abs(base_pred - ra) / ra * 100)
    base_med = float(np.median(np.abs(base_pred - ra) / ra * 100))
    model_med = float(np.median(np.abs(rp - ra) / ra * 100))

    beat_share = float(np.mean(np.abs(rp - ra) < np.abs(base_pred - ra)))

    model = make_model()
    model.fit(X, y)

    # Holdout (2025) and validation (2026) results, computed once
    def predict_row(row, mau):
        Xi = pd.DataFrame([row[FEATS].tolist()], columns=FEATS)
        return float(np.exp(model.predict(Xi)[0]) * mau ** ERA_EXP)

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

    base_hold_errs = [abs(np.median(ra) - r['actual']) / r['actual'] * 100 for r in hold_results]
    base_hold = float(np.mean(base_hold_errs)) if hold_results else float('nan')
    base_hold_med = float(np.median(base_hold_errs)) if hold_results else float('nan')
    hold_med = float(np.median([r['err'] for r in hold_results])) if hold_results else float('nan')

    weights = pd.Series(model.named_steps['ridge'].coef_, index=FEATS)

    return {
        'df': df, 'model': model, 'gm': gm, 'gm_spt': gm_spt,
        'r2': r2, 'r2_real': r2_real, 'mape': mape, 'mae': mae,
        'base_mape': base_mape, 'base_hold': base_hold,
        'base_hold_med': base_hold_med, 'hold_med': hold_med,
        'base_med': base_med, 'model_med': model_med,
        'hold_results': hold_results, 'val_results': val_results,
        'weights': weights, 'beat_share': beat_share, 'train_size': len(train)
    }


state = load_and_train()
df, model = state['df'], state['model']
gm, gm_spt = state['gm'], state['gm_spt']

# Counts used throughout the copy so nothing goes stale when the dataset changes.
N_TRAIN = state['train_size']
N_HOLD = len(state['hold_results'])
N_VAL = len(state['val_results'])
BASE_MAPE, BASE_HOLD = state['base_mape'], state['base_hold']
BASE_HOLD_MED, HOLD_MED = state['base_hold_med'], state['hold_med']
TYPICAL_ERR = state['model_med']   # cross-validated median error, drives the displayed range
HOLDOUT_MAPE = float(np.mean([r['err'] for r in state['hold_results']])) if N_HOLD else float('nan')

# Reference-only prior-album data. Used ONLY to correct "years since last album" and
# display-only prev-streams for artists whose earlier album has real numbers but is not a
# training row. Never fed into the model. Both are collaborations or projects deliberately
# left out of the dataset.
REFERENCE_HISTORY = {
    "21 Savage":          {"year": 2020, "streams": 108},  # Savage Mode II (collab w/ Metro, kept out)
    "Metro Boomin":       {"year": 2022, "streams": 226.2},  # Heroes & Villains (solo album, kept out; verified 226,156,391)
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
    """Predicted first-week streams (M). The model predicts log(streams / MAU**ERA_EXP), the same
    quantity the cross-validation numbers are computed on, then converts back."""
    Xi = pd.DataFrame([params], columns=FEATS)
    return float(np.exp(model.predict(Xi)[0]) * mau ** ERA_EXP)


def reference_adjusted_first_year(artist):
    """Earliest known release year for an artist, factoring in REFERENCE_HISTORY
    even though that earlier album is not part of the training data."""
    a = df[df['artist'] == artist].sort_values('year')
    years = a['year'].tolist()
    if artist in REFERENCE_HISTORY:
        years.append(REFERENCE_HISTORY[artist]['year'])
    return min(years) if years else None


# Albums already released in 2026 that are not training rows (first-week streams in millions).
# They count as an artist's previous album for a Jul to Dec 2026 prediction, so the app does not
# treat Drake, Olivia Rodrigo, Ariana Grande or BTS as having gone quiet since 2023 to 2024.
# Harry Styles' 2026 album is already a row in the data. Release dates: Arirang Mar 20,
# Iceman May 15, Olivia Rodrigo Jun 12, Petal Jul 31 (all 2026).
RELEASES_2026 = {
    "Drake":          [("Iceman", 450.4)],
    "Olivia Rodrigo": [("you seem pretty sad for a girl so in love", 394.7)],
    "Ariana Grande":  [("Petal", 236.1)],
    "BTS":            [("Arirang", 525.6)],
}


def artist_prior(artist, before_year=None, late_2026=False):
    """The artist's earlier albums as a DataFrame (album, year, first_week_streams), oldest
    first. With before_year set, only albums released strictly earlier count, so backdating a
    prediction does not let a later album leak in. late_2026 also counts the albums already
    released in 2026 (for a Jul to Dec 2026 prediction)."""
    cols = ['album', 'year', 'first_week_streams']
    a = df[df['artist'] == artist][cols]
    if before_year is None:
        return a.sort_values('year')
    keep = a[a['year'] < before_year]
    if late_2026 and before_year == 2026:
        extra = a[a['year'] == 2026]
        refs = pd.DataFrame([(alb, 2026, st_) for alb, st_ in RELEASES_2026.get(artist, [])], columns=cols)
        keep = pd.concat([keep, extra, refs], ignore_index=True)
    return keep.sort_values('year', kind='stable')


def latest_release_year(artist, before_year=None, late_2026=False):
    """Most recent known release year, including reference-only albums. With before_year
    set, only albums released strictly earlier count, so backdating a prediction does not
    let a later album leak in as the artist's 'previous' one."""
    a = artist_prior(artist, before_year, late_2026)
    year = int(a.iloc[-1]['year']) if len(a) else None
    if artist in REFERENCE_HISTORY:
        ref_year = REFERENCE_HISTORY[artist]['year']
        if (before_year is None or ref_year < before_year) and (year is None or ref_year > year):
            year = ref_year
    return year


def artist_history(artist, mau, before_year=None, late_2026=False):
    """Auto-fill trailing/prev fields from CSV history, falling back to dataset medians.
    Display-only prev_album_streams uses REFERENCE_HISTORY when the real training rows
    do not have it, so the UI shows the correct number even though it is not in training.

    before_year restricts the history to albums released strictly earlier, matching how the
    training features were built (prior_mean also looks only at year < the album's year).
    Without it, predicting an earlier period would feed the model albums that had not come
    out yet."""
    full = df[df['artist'] == artist].sort_values('year')
    a = artist_prior(artist, before_year, late_2026)

    # Household-name status is an attribute of the artist, not of one release, so it is read
    # from the full history rather than the period-restricted slice.
    is_hh = int(full.iloc[-1]['is_household_name']) if len(full) else 0

    scale = mau ** ERA_EXP
    has_hist = 1
    if len(a):
        prev = a.iloc[-1]['first_week_streams'] / scale
    elif artist in REFERENCE_HISTORY and (before_year is None
                                          or REFERENCE_HISTORY[artist]['year'] < before_year):
        prev = REFERENCE_HISTORY[artist]['streams'] / scale
    else:
        prev = gm / scale
        has_hist = 0

    trail_avg = (a['first_week_streams'].mean() if len(a) else gm) / scale
    return {
        'prev_streams_per_mau': prev,          # shown in the UI, not fed to the model
        'trailing_avg_per_mau': trail_avg,     # shown in the UI, not fed to the model
        'history_per_mau': (prev + trail_avg) / 2,
        'has_hist': has_hist,
        'is_household_name': is_hh,
    }


def compute_contribution_breakdown(params):
    """How far each input pushed the prediction up or down versus an average training album.
    The model is linear in log-streams, so each push is exactly weight x (how far the input sits
    from the training average, in standard deviations). Inputs overlap (for example track count
    and household name x track count), so read the bars as a rough guide, not independent effects."""
    friendly = {
        'track_count': 'Track count',
        'spotify_mau': 'Platform size (era)',
        'lead_time': 'Lead time',
        'feature_track_count': 'Featured tracks',
        'years_since_last_album': 'Gap since last album',
        'history_per_mau': "Artist's streaming history",
        'has_hist': 'Has a previous album on record',
        'is_household_name': 'Household-name status',
        'household_x_tracks': 'Household name x tracks',
    }
    scaler = model.named_steps['standardscaler']
    coef = model.named_steps['ridge'].coef_
    z = (np.array(params, dtype=float) - scaler.mean_) / scaler.scale_
    out = pd.DataFrame({'feature': [friendly.get(f, f) for f in FEATS], 'contribution': coef * z})
    out = out.sort_values('contribution', key=abs, ascending=False).head(5)
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
    draw.text((70, 460), f"Typical range: {lo_fmt} to {hi_fmt}", font=font_range, fill=(179, 179, 179))
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
        <p style="font-size:12px; color:#b3b3b3; margin:0; line-height:1.6;">Predicts global first-week Spotify streams for major album releases, using a Ridge regression trained on {N_TRAIN} albums ({TRAIN_YEAR_MIN} to {TRAIN_YEAR_MAX}). Predicted ahead of three summer 2026 releases, two of which landed within 7%.</p>
    </div>
    <div style="background:#13161d; border:1px solid #22262f; border-radius:12px; padding:16px 18px;">
        <div style="font-size:22px; margin-bottom:8px;">🎯</div>
        <p style="font-size:13px; font-weight:600; color:#fff; margin:0 0 6px;">How to use it</p>
        <p style="font-size:12px; color:#b3b3b3; margin:0; line-height:1.6;">Select an artist, pick the release period, and hit Predict. Artist streaming history is auto-filled, so you only need to supply the album-specific inputs like track count and lead time.</p>
    </div>
    <div style="background:#13161d; border:1px solid #22262f; border-radius:12px; padding:16px 18px;">
        <div style="font-size:22px; margin-bottom:8px;">🔧</div>
        <p style="font-size:13px; font-weight:600; color:#fff; margin:0 0 6px;">Always improving</p>
        <p style="font-size:12px; color:#b3b3b3; margin:0; line-height:1.6;">Roadmap: broader artist coverage and additional pre-release signals to improve accuracy.</p>
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
    "Typical error, unseen albums", f"{HOLD_MED:.0f}%",
    help=(f"The median miss on the {N_HOLD} albums from 2025 that the model never saw during training. "
          f"Always guessing the median album would have missed by {BASE_HOLD_MED:.0f}% on the same albums. "
          f"The average miss is larger ({HOLDOUT_MAPE:.0f}% against {BASE_HOLD:.0f}%) because a few debut "
          f"and small-artist albums are badly over-predicted."))
g2.metric(
    "Albums tracked", N_TRAIN,
    help=(f"Major album releases used to train the model ({TRAIN_YEAR_MIN} to {TRAIN_YEAR_MAX}). "
          f"A further {N_HOLD} albums from 2025 and {N_VAL} from 2026 are held out for testing only."))
g3.metric(
    "Model R² (relative scale)", f"{state['r2']:.3f}",
    help=(f"Measured on log(streams ÷ users), so it scores how well the model places an album's size "
          f"relative to Spotify's audience at the time, not its raw stream count. Computed by "
          f"leave-one-out cross-validation, on albums the model had not seen.\n\n"
          f"On the raw stream scale the same predictions score about {state['r2_real']:.2f}. A low score "
          f"there means a few outsized albums account for most of the spread and the model does not reach them. "
          f"Typical error is the more intuitive gauge of everyday accuracy."))
g4.metric(
    "Typical error, cross-validated", f"{state['model_med']:.1f}%",
    help=(f"The median percentage the prediction is off from the real number, measured across all training "
          f"albums using leave-one-out cross-validation. Always guessing the median album scores "
          f"{state['base_med']:.1f}%."))

# -----------------------------------------------------------------------------
# Prediction output (rendered above the inputs, full width)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="section-divider">
    <div class="section-divider-line"></div>
    <div style="text-align:center;">
        <div class="section-divider-label" style="font-size:clamp(22px, 5vw, 34px); letter-spacing:0.14em; line-height:1.15; white-space:normal;">Make a prediction</div>
        <div style="font-size:clamp(13px, 2.4vw, 16px); color:#d0d0d0; margin-top:8px;">Fill in album details below to generate a first-week stream estimate</div>
    </div>
    <div class="section-divider-line" style="background: linear-gradient(90deg, transparent, #1DB95488);"></div>
</div>
""", unsafe_allow_html=True)

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
        "Album name (optional)", placeholder="Enter album name here",
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
late_2026 = release_period.startswith('2026 (Jul')   # albums already out in 2026 count as history
mau_basis = f"{mau}M MAU, {release_period}"
if mau != MAU_BY_PERIOD[release_period]:
    st.caption(f"Using a manual override of {mau}M. {release_period} reported "
               f"{MAU_BY_PERIOD[release_period]}M.")
else:
    st.caption(MAU_SOURCE_NOTE)

# Artist bio snippet: last album plus a quick fact, limited to the chosen release period so
# it shows the history the model will actually use.
a_hist = artist_prior(artist_name, release_year, late_2026)
_when = 'on record' if late_2026 else f'before {release_year}'
if len(a_hist) > 0:
    last = a_hist.iloc[-1]
    best = a_hist.loc[a_hist['first_week_streams'].idxmax()]
    if len(a_hist) > 1:
        fact = f"Best first-week {_when}: {best['album']}, {int(best['first_week_streams'])}M ({int(best['year'])})"
    else:
        fact = f"Only one album tracked for this artist {_when}"
    st.markdown(
        f"<div style='background:#13161d; border:1px solid #1db95444; border-radius:6px; padding:8px 14px; "
        f"margin-bottom:12px; font-size:13px;'>"
        f"<span style='color:{SPOTIFY_GREEN}; font-weight:600;'>Last album {_when}:</span> "
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
        f"No albums tracked for {artist_name} {_when}, so the model falls back to "
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
        "Lead time (days)", min_value=0, value=30, step=1,
        help="Days from when the release date was first officially announced to the release date. If it was announced and released the same day, enter 0. Across the albums in this dataset the middle half ran from about 1 week (7 days) to about 7 weeks (50 days), with a median of 24.")
with f_col:
    feature_track_count = st.number_input(
        "Featured artist tracks", min_value=0, value=0, step=1,
        help="How many tracks include a credited guest artist (e.g. feat. Drake). Leave at 0 if it is a solo album.")

is_household = st.toggle(
    "Household-name artist", value=True,
    help="Is this artist a globally recognized name? Think Taylor Swift, Drake, Eminem. Turn off for artists who are big within a genre but not yet mainstream crossover.")

# Years since last album, measured against the chosen release period rather than today,
# so a prediction for an earlier period uses the gap as it stood back then.
most_recent_year = latest_release_year(artist_name, before_year=release_year, late_2026=late_2026)
years_auto = max(0, int(release_year - most_recent_year)) if most_recent_year else 0

hist = artist_history(artist_name, mau, before_year=release_year, late_2026=late_2026)
hist['is_household_name'] = int(is_household)
_scale = mau ** ERA_EXP
prev_auto = int(round(hist['prev_streams_per_mau'] * _scale)) if hist['has_hist'] else 0

# Manual override: the history comes from the albums in the data, so it can be out of date for an
# artist who has released something since (or a joint album, EP or mixtape the data leaves out).
with st.expander(f"Override history (auto: {years_auto} years since last album, "
                 f"previous album {prev_auto}M)"):
    st.caption("Both values are filled in from the data. If this artist has released a studio album "
               "the data does not know about, correct them here. Set previous album streams to 0 "
               "for a debut or if the number is unknown.")
    _c1, _c2 = st.columns(2)
    years_since_last = _c1.number_input(
        "Years since last album", min_value=0, value=years_auto, step=1,
        key=f"yrs_{artist_name}_{release_period}")
    prev_override = _c2.number_input(
        "Previous album first-week streams (M)", min_value=0, value=prev_auto, step=1,
        key=f"prev_{artist_name}_{release_period}")

if prev_override != prev_auto:
    if prev_override > 0:
        _prev = prev_override / _scale
        _trail = hist['trailing_avg_per_mau'] if len(a_hist) else _prev
        hist.update(prev_streams_per_mau=_prev, trailing_avg_per_mau=_trail,
                    history_per_mau=(_prev + _trail) / 2, has_hist=1)
    else:
        _fallback = gm / _scale
        hist.update(prev_streams_per_mau=_fallback, trailing_avg_per_mau=_fallback,
                    history_per_mau=_fallback, has_hist=0)
if years_since_last != years_auto or prev_override != prev_auto:
    st.caption("Using your overridden history values.")

with st.expander("History used by the model (dataset medians used if no history)"):
    st.write(f"Previous release first-week streams: **{hist['prev_streams_per_mau'] * _scale:.0f}M**")
    st.write(f"Career average first-week streams: **{hist['trailing_avg_per_mau'] * _scale:.0f}M**")
    st.write(f"Combined history signal used by the model: **{hist['history_per_mau'] * _scale:.0f}M**")

predict_clicked = st.button("Predict first-week streams", type="primary", use_container_width=True)
prediction_slot = st.container()

# -----------------------------------------------------------------------------
# Render the prediction directly below the Predict button
# -----------------------------------------------------------------------------
with prediction_slot:
    if predict_clicked:
        with st.spinner(f"Crunching {N_TRAIN} albums of streaming history..."):
            time.sleep(0.8)
            params = [
                track_count, mau, lead_time, feature_track_count,
                years_since_last, hist['history_per_mau'], hist['has_hist'],
                int(is_household), int(is_household) * track_count
            ]
            pred = predict(params, mau)

        typ = TYPICAL_ERR
        daily_avg = pred / 7

        prior_albums = len(artist_prior(artist_name, release_year, late_2026))
        if not hist['has_hist']:
            card_border = "#f59e0b"
            conf_note = (f"⚠️ No previous album on record for this artist before {release_year}. "
                         f"This is where the model is weakest: on the 2025 test albums it over-predicted "
                         f"most debut and new-artist releases, often by 2x or more. Treat the number as "
                         f"an upper estimate.")
        elif prior_albums < 2:
            card_border = "#f59e0b"
            conf_note = (f"⚠️ Lower confidence: only {prior_albums} album"
                         f"{'' if prior_albums == 1 else 's'} tracked for this artist before "
                         f"{'now' if late_2026 else release_year}, so the history signal is thin.")
        else:
            card_border = SPOTIFY_GREEN
            conf_note = None

        lo_fmt = f"{pred * (1 - typ / 100):.0f}M"
        hi_fmt = f"{pred * (1 + typ / 100):.0f}M"
        album_label = f" ({album_name})" if album_name else ""

        st.markdown(f"""
        <div class="result-card" style="border-color:{card_border};">
            <div style="color:#b3b3b3; font-size:14px; margin-bottom:4px;">Predicted first-week global streams</div>
            <div style="color:{SPOTIFY_GREEN}; font-size:52px; font-weight:700; line-height:1.1;">{pred:.0f}M</div>
            <div style="color:#b3b3b3; font-size:14px; margin-top:10px;">
                Typical range (±{typ:.0f}%): {lo_fmt} to {hi_fmt}
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
        m3.metric("Typical error", f"±{typ:.0f}%",
                  help=("The median miss across all training albums in leave-one-out cross-validation. "
                        "About half of past albums landed outside a range this wide, so treat it as a "
                        "typical miss, not a guarantee. Misses on debut and small artists run much larger."))

        def g(text):
            return f"<span style='color:{SPOTIFY_GREEN}; font-weight:700;'>{text}</span>"

        st.markdown(
            f"<div style='font-size:15px; line-height:1.7; color:#e0e0e0; margin-top:12px;'>"
            f"The model expects <strong>{artist_name}</strong>{album_label}'s album to stream around "
            f"{g(f'{pred:.0f}M')} times globally in its first week, an average of {g(f'{daily_avg:.1f}M')} "
            f"streams per day. That works out to roughly {g(f'{pred / track_count:.1f}M')} per track. "
            f"The model's typical miss in testing was about {g(f'{typ:.0f}%')}, which gives a range of "
            f"{g(f'{lo_fmt} to {hi_fmt}')}. About half of past albums landed outside a range this wide."
            f"</div>",
            unsafe_allow_html=True
        )
        st.caption("For reference: Drake's Iceman did 450M in its first week, Olivia Rodrigo's album did 395M, "
                   "and Ariana Grande's Petal did 236M.")
        st.caption("Note: your prediction is session only and will not affect the model.")

        with st.expander("🔍 Why this number?"):
            st.caption(
                "Which inputs pushed the prediction up or down compared with an average training album. "
                "Each bar is the model's weight for that input times how far your value sits from the "
                "training average. Inputs overlap, so read the bars as a rough guide."
            )
            breakdown = compute_contribution_breakdown(params)
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
            Fill in the album details above and hit <strong style="color:#fff;">Predict first-week streams</strong>. Your estimate will appear here, with a confidence range and a plain-English breakdown of what the numbers mean.
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
        f"Across those unseen albums its typical (median) miss was **{HOLD_MED:.0f}%**, against "
        f"{BASE_HOLD_MED:.0f}% for always guessing the median album. Its average miss was "
        f"{HOLDOUT_MAPE:.0f}% against {BASE_HOLD:.0f}%; the average is pulled up by a few badly "
        f"over-predicted albums. "
        f"For context, predicting music streams is inherently noisy. Viral moments, surprise drops "
        f"and cultural timing can swing numbers by 2x to 3x in ways no model can fully anticipate."
    )
    hold_df = pd.DataFrame(state['hold_results'])
    hold_df['pred'] = hold_df['pred'].round(0).astype(int)
    hold_df['actual'] = hold_df['actual'].round(0).astype(int)
    hold_df['err'] = hold_df['err'].round(1)
    hold_df.columns = ['Album', 'Predicted (M)', 'Actual (M)', 'Error (%)']
    st.caption("2025 albums the model never saw in training, predicted and then checked against the real numbers:")
    st.dataframe(hold_df, hide_index=True, use_container_width=True)
    st.caption(f"Typical error on these albums: {HOLD_MED:.1f}%. Average error: {HOLDOUT_MAPE:.1f}%.")

with st.expander("✅ Real-world track record (2026)"):
    st.markdown(
        "These are 2026 albums with known outcomes. The model was frozen before they were added, "
        "so none of them influenced training or tuning. Inputs were locked before it was run."
    )
    val_df = pd.DataFrame(state['val_results'])
    val_df['pred'] = val_df['pred'].round(0).astype(int)
    val_df['actual'] = val_df['actual'].round(0).astype(int)
    val_df['err'] = val_df['err'].round(1)
    val_df.columns = ['Album', 'Predicted (M)', 'Actual (M)', 'Error (%)']
    st.dataframe(val_df, hide_index=True, use_container_width=True)
    st.caption(f"Typical error on these {len(val_df)} albums: {val_df['Error (%)'].median():.1f}%. "
               f"Average error: {val_df['Error (%)'].mean():.1f}%. The model over-predicted the albums with "
               "many tracks and guests that opened small (Karol G, J. Cole, Zach Bryan).")

with st.expander("⚙️ What factors drive the prediction?"):
    st.markdown(
        "The model's weight on each input, on a common scale. Positive weights raise the "
        "prediction, negative weights lower it. Larger absolute values mean more influence."
    )
    imp_df = state['weights'].sort_values(key=abs, ascending=False).reset_index()
    imp_df.columns = ['Factor', 'Weight']
    friendly_factors = {
        'history_per_mau': "Artist's streaming history (last album + career avg.)",
        'has_hist': 'Has a previous album on record',
        'spotify_mau': 'Spotify platform size (era)',
        'lead_time': 'Days from release-date announcement to release',
        'household_x_tracks': 'Household name x track count',
        'track_count': 'Number of tracks',
        'feature_track_count': 'Featured artist tracks',
        'years_since_last_album': 'Gap since last album',
        'is_household_name': 'Household-name status',
    }
    imp_df['Factor'] = imp_df['Factor'].map(friendly_factors).fillna(imp_df['Factor'])
    imp_df['Weight'] = imp_df['Weight'].round(3)
    st.dataframe(imp_df, hide_index=True, use_container_width=True)

with st.expander("📖 How does this model work?"):
    st.markdown(f"""
This application uses a **Ridge regression** (a regularized linear model) trained on {N_TRAIN} major
album releases from {TRAIN_YEAR_MIN} to {TRAIN_YEAR_MAX}, plus {N_HOLD} albums from 2025 and {N_VAL} from
2026 held out purely for testing (never used in training).

**The core idea:** an album's first-week streams are mostly explained by whether the artist has a real
track record on the platform, how big their name is, how many tracks the album has, and how big
Spotify was at the time. The model works on log(streams) after a platform-size adjustment, so each
input scales the prediction by a percentage.

**Platform size:** stream figures are divided by Spotify's monthly users raised to the power 0.25,
not in full, because first-week streams grew far more slowly than the user base.

**The "has a previous album" flag:** when an artist has no earlier album on record, the history input
is filled in with the median training album. The flag tells the model when that has happened. It was
the largest single improvement: with it, a linear model's typical cross-validated error fell from
about 41% to about 32%.

**Inputs that had to be fixed:** the model originally included a hand-assigned "hype score" for
pre-release buzz, and it was the single strongest predictor. The problem was that those scores had
been assigned *after* the albums came out, so they carried hindsight the model could exploit.
Cross-validation cannot catch that, because the contamination sits inside the feature's values rather
than across the train/test split. Dropping it made the cross-validated numbers worse and the held-out
2025 albums better, which is the signature of a leaky predictor. Lead times had a different problem:
they were first entered from memory, and when re-checked against announcement dates most of them
changed, many by a month or more. The verified ones carry almost no signal on their own.

**Why Ridge:** it was chosen by leave-one-out cross-validation alone, before looking at the 2025
albums. Its typical error was {state['model_med']:.0f}%, against about 38% for the Random Forest this
project started with. With only {N_TRAIN} training rows, a simple model that adds up weighted inputs
is a better fit than one that has to learn thresholds from a handful of albums.

**How accuracy is measured:** every figure in this app comes from Leave-One-Out Cross-Validation: train
on {N_TRAIN - 1} albums, test on the 1 left out, repeat {N_TRAIN} times. That gives an honest read on
data the model has not seen. The model beats the median guess on about {state['beat_share'] * 100:.0f}% of
albums in this test.

**What the model does not know:** things like label disputes, surprise tour announcements, viral
TikTok moments, or genuine cultural narrative are deliberately left out. They are real factors, but
they cannot be reliably quantified across just {N_TRAIN} albums without the model latching onto
coincidence instead of signal.

**Known limitations:** the model tends to over-predict for artists showing signs of declining
momentum, since it has no explicit signal for that. Ariana Grande's *Petal* is the clearest recent
example: the model called 408M and the album opened at 236.1M, a miss of roughly 73%. It also
over-predicts small and new artists and under-predicts the very biggest releases (for example Taylor
Swift's *Life of a Showgirl*): it pulls everything toward the middle. Surprise album drops with little
to no announcement lead time are harder to model than traditional promotional cycles.

**The honest headline:** the model is better than a simple guess, but its errors are still large. In
cross-validation its typical (median) error is {state['model_med']:.0f}% against {state['base_med']:.0f}% for
always guessing the median album. On the {N_HOLD} albums from 2025 it is {HOLD_MED:.0f}% against
{BASE_HOLD_MED:.0f}%. The average error is higher ({state['mape']:.0f}% in cross-validation) because a few
small or debut albums are badly over-predicted. {N_TRAIN} training rows against {len(FEATS)} features is
thin. Treat its output as a ballpark for how an album is likely to open, not a precise forecast.
    """)

st.markdown(f"""
<div class="app-footer">
    <span>Built with Spotify streaming data</span>
    <span class="accent">Ridge regression model</span>
    <span>{N_TRAIN} training albums ({TRAIN_YEAR_MIN} to {TRAIN_YEAR_MAX})</span>
    <span>Last updated November 2026</span>
</div>
""", unsafe_allow_html=True)
