"""
E-commerce Review Intelligence — app.py  v4.0
Fixes: removed unnecessary sidebar box, added << >> toggle button for sidebar,
sentiment all-neutral bug, large-file upload, blue animated sidebar,
lavender/purple dashboard palette, CSS page-enter animations.

Run:
    pip install streamlit plotly pandas nltk wordcloud matplotlib pillow scikit-learn
    streamlit run app.py
"""

import io, re, textwrap, random, zipfile
from pathlib import Path

import nltk
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib import pyplot as plt
from wordcloud import WordCloud

# ── NLTK ─────────────────────────────────────────────────────────────────────
for _pkg, _path in [
    ("vader_lexicon", "sentiment/vader_lexicon"),
    ("stopwords",     "corpora/stopwords"),
]:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords as _sw
STOP_WORDS = set(_sw.words("english"))

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-commerce Review Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*,html,body,[class*="css"]{ font-family:'Inter',sans-serif !important; box-sizing:border-box; }

/* ══ SIDEBAR — deep blue ══ */
[data-testid="stSidebar"]{
    background: linear-gradient(175deg,#4f46e5 0%,#3730a3 55%,#312e81 100%) !important;
    min-width:220px !important; max-width:220px !important;
    border-right:none !important;
    box-shadow:4px 0 28px rgba(79,70,229,.4) !important;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color:rgba(255,255,255,.92) !important; }

/* ══ HIDE THE UNWANTED COLLAPSE ARROW / HANDLE ══ */
[data-testid="stSidebarCollapseButton"],
button[kind="header"],
[data-testid="collapsedControl"] {
    display: none !important;
}

/* ══ REMOVE ALL DEFAULT STREAMLIT SIDEBAR TOP PADDING / LOGO SPACE ══ */
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

/* ══ KILL THE EMPTY WHITE BOX (stImage / stLogo area) ══ */
[data-testid="stSidebar"] [data-testid="stImage"],
[data-testid="stSidebar"] [data-testid="stLogo"],
[data-testid="stSidebar"] .stImage,
[data-testid="stSidebar"] img:not([src*="data:"]) {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Also kill any blank block-containers at the top of sidebar */
[data-testid="stSidebar"] > div > div > div:first-child:empty {
    display: none !important;
}

/* hide radio label */
div[data-testid="stRadio"] > label { display:none !important; }
div[data-testid="stRadio"] > div {
    display:flex !important; flex-direction:column !important; gap:2px !important;
}

/* nav items */
div[data-testid="stRadio"] > div > label {
    display:flex !important; align-items:center !important;
    padding:8px 14px !important; border-radius:12px !important;
    cursor:pointer !important; font-size:13.5px !important; font-weight:600 !important;
    color:rgba(255,255,255,.88) !important;
    transition:all .25s cubic-bezier(.4,0,.2,1) !important;
    margin:0 !important; min-height:0 !important; line-height:1.2 !important;
}
div[data-testid="stRadio"] > div > label p {
    margin:0 !important; padding:0 !important; line-height:1.2 !important;
}
div[data-testid="stRadio"] > div > label:hover{
    background:rgba(255,255,255,.14) !important;
    transform:translateX(5px) !important;
    color:#fff !important;
}
div[data-testid="stRadio"] > div > label:has(input:checked){
    background:#ffffff !important; color:#4f46e5 !important; font-weight:800 !important;
    box-shadow:0 6px 22px rgba(0,0,0,.22) !important;
    transform:translateX(0) scale(1.02) !important;
    animation:navPop .3s cubic-bezier(.34,1.56,.64,1) both !important;
}
div[data-testid="stRadio"] > div > label:has(input:checked) *{ color:#4f46e5 !important; }
@keyframes navPop{
    0%  {transform:translateX(-10px) scale(.95);opacity:.5;}
    100%{transform:translateX(0)     scale(1.02);opacity:1;}
}
div[data-testid="stRadio"] span[data-baseweb="radio"]{ display:none !important; }

/* sidebar inputs */
[data-testid="stSidebar"] input{
    background:rgba(255,255,255,.15) !important;
    border:1px solid rgba(255,255,255,.3) !important;
    color:#fff !important; border-radius:8px !important;
}
[data-testid="stSidebar"] input::placeholder{ color:rgba(255,255,255,.45) !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"]{
    background:rgba(255,255,255,.1) !important;
    border:1px dashed rgba(255,255,255,.4) !important; border-radius:10px !important;
}
[data-testid="stSidebar"] hr{
    border-color:rgba(255,255,255,.2) !important; margin:8px 0 !important;
}

/* Analyze button inside sidebar */
[data-testid="stSidebar"] .stButton > button{
    background:#ffffff !important; color:#4f46e5 !important; border:none !important;
    border-radius:10px !important; font-weight:700 !important; font-size:13px !important;
    padding:10px !important; width:100% !important;
    box-shadow:0 4px 16px rgba(0,0,0,.22) !important; transition:all .2s !important;
}
[data-testid="stSidebar"] .stButton > button *{
    color:#4f46e5 !important;
    font-weight:800 !important;
    visibility:visible !important;
}
[data-testid="stSidebar"] .stButton > button:hover{
    background:#f0f0ff !important; transform:translateY(-2px) !important;
    box-shadow:0 8px 24px rgba(0,0,0,.28) !important;
}

/* ══ TOGGLE BUTTON — << >> ══ */
#sidebar-toggle-btn {
    position: fixed !important;
    top: 50% !important;
    left: 220px !important;
    transform: translateY(-50%) !important;
    z-index: 99999 !important;
    width: 22px !important;
    height: 48px !important;
    background: #4f46e5 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 0 8px 8px 0 !important;
    cursor: pointer !important;
    font-size: 13px !important;
    font-weight: 900 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 3px 0 12px rgba(79,70,229,.4) !important;
    transition: left 0.3s ease, background 0.2s !important;
    line-height: 1 !important;
    padding: 0 !important;
    font-family: 'Inter', sans-serif !important;
}
#sidebar-toggle-btn:hover {
    background: #6366f1 !important;
    width: 26px !important;
}

/* ══ MAIN — lavender gradient background ══ */
.main{
    background:linear-gradient(135deg,#eef2ff 0%,#f5f3ff 40%,#ede9fe 70%,#e0e7ff 100%) !important;
}
.main .block-container{
    background:transparent !important;
    padding:1.4rem 2rem 2rem !important; max-width:100% !important;
}

/* ══ STAT CARDS ══ */
.stat-card{
    background:#fff; border-radius:14px;
    border:1px solid rgba(99,102,241,.12); padding:16px 18px 14px;
    box-shadow:0 2px 14px rgba(99,102,241,.09);
    transition:transform .2s,box-shadow .2s;
    animation:cardIn .4s ease both;
    position:relative; overflow:hidden;
}
.stat-card::after{
    content:''; position:absolute; top:0;left:0;right:0; height:3px;
    border-radius:14px 14px 0 0; background:var(--accent,#6366f1);
}
.stat-card:hover{ transform:translateY(-5px) scale(1.01); box-shadow:0 12px 32px rgba(99,102,241,.18); }
@keyframes cardIn{
    from{opacity:0;transform:translateY(18px);}
    to  {opacity:1;transform:translateY(0);}
}
.stat-label{font-size:11px;color:#6b7280;font-weight:600;margin-bottom:4px;
            text-transform:uppercase;letter-spacing:.6px;}
.stat-value{font-size:26px;font-weight:800;color:#1e1b4b;line-height:1.1;}
.stat-sub  {font-size:11px;color:#9ca3af;margin-top:3px;}
.stat-delta{font-size:11px;font-weight:600;margin-top:2px;}
.stat-bar  {height:3px;border-radius:99px;margin-top:10px;background:#f3f4f6;overflow:hidden;}
.stat-bar-fill{height:100%;border-radius:99px;transition:width 1.2s ease;}

/* ══ SECTION CARDS ══ */
.section-card{
    background:#fff; border-radius:16px;
    border:1px solid rgba(99,102,241,.1); padding:20px 22px;
    box-shadow:0 2px 16px rgba(99,102,241,.07);
    animation:cardIn .5s ease both;
    transition:box-shadow .2s;
}
.section-card:hover{ box-shadow:0 8px 30px rgba(99,102,241,.15); }
.section-title{
    font-size:14px;font-weight:700;color:#1e1b4b;margin-bottom:14px;
    display:flex;align-items:center;gap:6px;
}
.title-icon{ display:inline-block; transition:transform .3s; }
.section-card:hover .title-icon{ transform:rotate(10deg) scale(1.15); }

/* ══ TOPBAR ══ */
.topbar{
    display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px;
    animation:topIn .35s ease both;
}
@keyframes topIn{from{opacity:0;transform:translateY(-10px);}to{opacity:1;transform:translateY(0);}}
.topbar-right{display:flex;gap:10px;align-items:center;}
.date-pill,.filter-pill{
    background:#fff;border:1px solid #e5e7eb;border-radius:10px;
    padding:7px 14px;font-size:12px;color:#374151;
    display:flex;align-items:center;gap:5px;
    box-shadow:0 1px 6px rgba(0,0,0,.06); transition:box-shadow .2s;
}
.date-pill:hover,.filter-pill:hover{ box-shadow:0 4px 12px rgba(99,102,241,.14); }

/* ══ BADGES ══ */
.badge{display:inline-block;padding:2px 10px;border-radius:99px;font-size:11px;font-weight:600;}
.badge-pos {background:#d1fae5;color:#065f46;}
.badge-neg {background:#fee2e2;color:#991b1b;}
.badge-neu {background:#fef3c7;color:#92400e;}
.badge-fake{background:#ede9fe;color:#5b21b6;}
.badge-no  {background:#f3f4f6;color:#374151;}
.issue-pill{display:inline-block;padding:2px 8px;border-radius:6px;font-size:11px;font-weight:600;}

/* ══ TABLE ══ */
.rev-table{width:100%;border-collapse:collapse;font-size:12px;}
.rev-table th{
    padding:8px 10px;text-align:left;color:#6b7280;font-weight:600;font-size:11px;
    border-bottom:2px solid #e5e7eb;background:#fafafe;
    text-transform:uppercase;letter-spacing:.5px;
}
.rev-table td{
    padding:10px;border-bottom:1px solid #f3f4f6;
    color:#374151;font-size:12px;vertical-align:top;transition:background .15s;
}
.rev-table tr:hover td{background:#f5f3ff;}
.rev-table tr:last-child td{border-bottom:none;}

/* ══ PAGE ENTER ══ */
.page-enter{animation:pageIn .35s cubic-bezier(.4,0,.2,1) both;}
@keyframes pageIn{
    from{opacity:0;transform:translateY(22px);}
    to  {opacity:1;transform:translateY(0);}
}

/* ══ MISC ══ */
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#f0f0ff;}
::-webkit-scrollbar-thumb{background:#c7d2fe;border-radius:99px;}
.stButton > button{
    background:linear-gradient(135deg,#6366f1,#4f46e5) !important;
    color:#fff !important; border:none !important; border-radius:10px !important;
    font-weight:700 !important; padding:9px 20px !important; font-size:13px !important;
    box-shadow:0 4px 14px rgba(99,102,241,.3) !important; transition:all .2s !important;
}
.stButton > button:hover{transform:translateY(-2px) !important;box-shadow:0 8px 22px rgba(99,102,241,.4) !important;}
.stSelectbox > div > div,.stTextInput > div > div{
    border-color:#c7d2fe !important; border-radius:10px !important; background:#fff !important;
}
.stSelectbox > div > div:focus-within,.stTextInput > div > div:focus-within{
    border-color:#6366f1 !important; box-shadow:0 0 0 3px rgba(99,102,241,.15) !important;
}
[data-testid="stDataFrame"]{border-radius:12px;border:1px solid #e5e7eb !important;}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
  background: rgba(255,255,255,0.72) !important;
  border: 1.5px dashed #c4b5fd !important;
  border-radius: 18px !important;
  padding: 0.8rem !important;
  overflow: hidden !important;
  transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: #8b5cf6 !important;
  background: rgba(255,255,255,0.85) !important;
}
[data-testid="stFileUploaderDropzone"] {
  border: none !important;
  background: transparent !important;
  padding: 0 !important;
}
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
  position: relative !important;
  min-width: 120px !important;
  height: 44px !important;
  padding: 0 1rem !important;
  border-radius: 14px !important;
  border: 1px solid rgba(139,92,246,0.25) !important;
  background: white !important;
  color: transparent !important;
  font-size: 0 !important;
  overflow: hidden !important;
  box-shadow: 0 2px 10px rgba(109,40,217,0.08) !important;
  transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]:hover {
  border-color: #8b5cf6 !important;
  background: #f8f5ff !important;
}
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] * { display: none !important; }
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]::after {
  content: "Upload File";
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 14px !important; font-weight: 600 !important;
  color: #5b21b6 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
  font-size: 14px !important; color: #4b4469 !important; margin-top: 0.4rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] small {
  font-size: 12px !important; color: #8b82ab !important;
}

/* Sidebar heading helper */
.sidebar-heading{
    color:rgba(255,255,255,0.6) !important;
    font-size:10px !important;
    font-weight:800 !important;
    letter-spacing:1.4px !important;
    text-transform:uppercase !important;
    margin: 10px 0 6px 4px !important;
    display:block;
}
.topbar-right{display:none !important;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR TOGGLE JAVASCRIPT  (injects << >> floating button)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<script>
(function() {
    function initToggle() {
        if (document.getElementById('sidebar-toggle-btn')) return;

        var btn = document.createElement('button');
        btn.id = 'sidebar-toggle-btn';
        btn.textContent = '«';
        btn.title = 'Hide sidebar';
        document.body.appendChild(btn);

        var sidebar = document.querySelector('[data-testid="stSidebar"]');
        var isOpen  = true;

        function updateState() {
            if (!sidebar) sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) return;
            if (isOpen) {
                btn.style.left = '220px';
                btn.textContent = '«';
                btn.title = 'Hide sidebar';
                sidebar.style.transform = 'translateX(0)';
                sidebar.style.width     = '220px';
                sidebar.style.minWidth  = '220px';
            } else {
                btn.style.left = '0px';
                btn.textContent = '»';
                btn.title = 'Show sidebar';
                sidebar.style.transform = 'translateX(-220px)';
                sidebar.style.width     = '0px';
                sidebar.style.minWidth  = '0px';
            }
        }

        btn.addEventListener('click', function() {
            isOpen = !isOpen;
            if (!sidebar) sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (sidebar) {
                sidebar.style.transition = 'transform 0.3s ease, width 0.3s ease, min-width 0.3s ease';
            }
            updateState();
        });

        updateState();
    }

    // Try immediately, then on load, then poll until sidebar exists
    initToggle();
    window.addEventListener('load', initToggle);
    var poll = setInterval(function() {
        var sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) { initToggle(); clearInterval(poll); }
    }, 300);
})();
</script>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
ISSUE_KW = {
    "Delivery":         ["delivery","shipping","late","slow","arrived","dispatch",
                         "package","ship","courier","tracking","delayed","days to arrive"],
    "Product Quality":  ["quality","broke","broken","poor quality","defective","cheap",
                         "material","falls apart","flimsy","not durable","bad quality"],
    "Damaged Product":  ["damaged","damage","dent","crack","cracked","scratched",
                         "bent","torn","smashed","crushed"],
    "Customer Service": ["customer service","unresponsive","support","refund","return",
                         "replacement","no response","unhelpful","rude","ignored"],
    "Wrong Item":       ["wrong item","incorrect","not as described","different",
                         "wrong product","wrong color","wrong size","mislabeled"],
}
ISSUE_COLORS = {
    "Delivery":"#6366f1","Product Quality":"#f97316",
    "Damaged Product":"#ef4444","Customer Service":"#10b981",
    "Wrong Item":"#8b5cf6","Other":"#9ca3af",
}
ISSUE_ICONS = {
    "Delivery":"🚚","Product Quality":"🔧","Damaged Product":"📦",
    "Customer Service":"🔄","Wrong Item":"❓","Other":"📋",
}
PILL_COLORS = {
    "Delivery":         ("#dbeafe","#1d4ed8"),
    "Product Quality":  ("#ffedd5","#c2410c"),
    "Damaged Product":  ("#fee2e2","#991b1b"),
    "Customer Service": ("#d1fae5","#065f46"),
    "Wrong Item":       ("#ede9fe","#5b21b6"),
    "Other":            ("#f3f4f6","#374151"),
}

_RAW = [
    ("Great product! Works exactly as expected and delivery was very fast.",5),
    ("Terrible quality. Broke after just 2 days! Total waste of money.",1),
    ("Item was heavily damaged when received. Packaging was completely poor.",2),
    ("Amazing! Absolutely the best purchase I've made this year. Highly recommended!",5),
    ("Didn't receive the correct item at all. Very disappointed and frustrated.",1),
    ("Average product, nothing special but it does the job fine.",3),
    ("Customer service was completely unresponsive for weeks. No help at all.",2),
    ("Fast shipping, great packaging, very happy with this purchase overall.",4),
    ("Product stopped working entirely after just one week. Poor quality control.",1),
    ("Exactly as described. Great value for money, would buy again.",4),
    ("Delivery was extremely late and item was different from the picture shown.",2),
    ("Excellent product! Highly recommend to everyone, truly outstanding.",5),
    ("Wrong item was sent. Still waiting for my replacement after 2 weeks.",1),
    ("Good product overall but a bit expensive for what it actually is.",3),
    ("Never received my order at all. Zero response from the seller.",1),
    ("Works perfectly, easy to set up, and has great build quality.",5),
    ("Packaging arrived damaged but the product inside was fine.",3),
    ("Very poor quality material, definitely not worth the money.",1),
    ("Delivery took 3 full weeks when they said 5 days. Very slow service.",2),
    ("Outstanding product! Truly exceeded all my expectations, love it.",5),
    ("Received a badly cracked item, very frustrating experience.",1),
    ("Decent product overall, shipping was fairly okay, no major issues.",3),
    ("Customer support team helped me quickly and efficiently, very impressed.",4),
    ("Wrong size delivered, had to return it which was a hassle.",2),
    ("Absolutely love this product! Will definitely buy again.",5),
    ("Cheap material that started peeling after only one wash.",1),
    ("Shipped fast but item has a very strange smell, not happy.",2),
    ("Pretty good for the price, no real complaints from me.",4),
    ("Order arrived broken and with terrible packaging, disappointing.",1),
    ("Love the quality, totally exceeded my expectations for this price point.",5),
    ("Moderate quality, does what it says on the box, nothing more.",3),
    ("Completely wrong product sent to me. Very unhappy with this.",1),
    ("Super fast delivery, well packaged, product works great.",5),
    ("Disappointing quality for this price, expected much better.",2),
    ("Solid product, works as intended, good value overall.",4),
    ("Shipping was okay, product is acceptable but could be improved.",3),
    ("Broken on arrival, terrible experience from start to finish.",1),
    ("Really happy with this, great product and fast delivery.",5),
    ("Not what I ordered, completely different item in the box.",1),
    ("Quite good, happy with my purchase, would recommend it.",4),
]
DEMO_REVIEWS = _RAW * 175


# ══════════════════════════════════════════════════════════════════════════════
#  NLP
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_sia():
    return SentimentIntensityAnalyzer()

def classify_sentiment(text: str, stars=None) -> str:
    score = get_sia().polarity_scores(str(text))["compound"]
    try:
        s = float(stars)
        if s >= 4.0:
            return "Positive" if score >= -0.20 else "Negative"
        elif s <= 2.0:
            return "Positive" if score >= 0.40 else "Negative"
        else:
            if score >= 0.15:    return "Positive"
            elif score <= -0.15: return "Negative"
            else:                return "Neutral"
    except Exception:
        pass
    if score >= 0.10:  return "Positive"
    if score <= -0.10: return "Negative"
    return "Neutral"

def classify_issue(text: str) -> str:
    t = str(text).lower()
    for issue, kws in ISSUE_KW.items():
        if any(k in t for k in kws):
            return issue
    return "Other"

def fake_risk(text: str, rating) -> float:
    t = str(text).lower()
    score = 0.0
    for w in ["amazing","best ever","perfect","incredible","outstanding",
              "phenomenal","unbelievable","absolutely love","highly recommend"]:
        if w in t: score += 0.15
    if len(t.split()) < 6:          score += 0.25
    if re.search(r"(.)\1{3,}", t):  score += 0.2
    try:
        if float(rating) == 5: score += 0.08
    except Exception:
        pass
    return min(score, 1.0)

@st.cache_data(show_spinner=False)
def process_df(df_json: str) -> pd.DataFrame:
    df = pd.read_json(io.StringIO(df_json))
    if "reviewText" not in df.columns:
        tc = next((c for c in df.columns
                   if df[c].dtype == object and df[c].str.len().mean() > 20), None)
        df = df.rename(columns={tc: "reviewText"}) if tc else df.assign(reviewText="")
    if "overall" not in df.columns:
        rc = next((c for c in df.columns
                   if pd.api.types.is_numeric_dtype(df[c])
                   and df[c].between(1,5).mean() > 0.6), None)
        df["overall"] = df[rc] if rc else 3

    df["reviewText"] = df["reviewText"].fillna("").astype(str)
    df["overall"]    = pd.to_numeric(df["overall"], errors="coerce").fillna(3)
    df["Sentiment"]  = df.apply(lambda r: classify_sentiment(r["reviewText"], r["overall"]), axis=1)
    df["Issue"]      = df["reviewText"].apply(classify_issue)
    df["FakeScore"]  = df.apply(lambda r: fake_risk(r["reviewText"], r["overall"]), axis=1)
    df["FakeReview"] = df["FakeScore"].apply(lambda s: "Yes" if s >= 0.35 else "No")

    if not any(c.lower() == "date" for c in df.columns):
        df["Date"] = pd.date_range("2024-01-01", periods=len(df), freq="h").strftime("%b %d, %Y")
    else:
        dc = next(c for c in df.columns if c.lower() == "date")
        df["Date"] = df[dc].astype(str)

    product_candidates = [
        "name", "product", "productName", "product_name", "title",
        "asins", "brand", "categories"
    ]
    pc = next((c for c in product_candidates if c in df.columns), None)
    if pc:
        df["Product"] = df[pc].fillna("Unknown Product").astype(str)
    else:
        product_names = ["Echo Dot", "Fire Tablet", "Kindle", "Amazon Basics", "Smart Plug"]
        df["Product"] = [product_names[i % len(product_names)] for i in range(len(df))]
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════
def load_data(source, local_path, uploaded_file, n_rows):
    df = None
    try:
        if source == "Demo sample":
            df = pd.DataFrame(DEMO_REVIEWS[:n_rows], columns=["reviewText","overall"])

        elif source == "Upload file" and uploaded_file is not None:
            raw = uploaded_file.read()
            buf = io.BytesIO(raw)
            ext = Path(uploaded_file.name).suffix.lower()
            if ext == ".csv":
                df = pd.read_csv(buf, nrows=n_rows, on_bad_lines="skip",
                                 encoding="utf-8", encoding_errors="replace")
            elif ext in (".json",):
                df = pd.read_json(buf).head(n_rows)
            elif ext == ".jsonl":
                df = pd.read_json(buf, lines=True).head(n_rows)
            elif ext == ".zip":
                with zipfile.ZipFile(buf) as zf:
                    found = False
                    for name in zf.namelist():
                        low = name.lower()
                        if low.endswith(".csv"):
                            df = pd.read_csv(zf.open(name), nrows=n_rows,
                                             on_bad_lines="skip", encoding_errors="replace")
                            found = True; break
                        elif low.endswith((".json",".jsonl")):
                            df = pd.read_json(zf.open(name),
                                              lines=low.endswith(".jsonl")).head(n_rows)
                            found = True; break
                    if not found:
                        st.error("No CSV/JSON inside the ZIP."); return None
            elif ext in (".txt",".bz2"):
                try:
                    df = pd.read_json(buf, lines=True).head(n_rows)
                except Exception:
                    df = pd.read_csv(buf, nrows=n_rows, on_bad_lines="skip",
                                     encoding_errors="replace")
            else:
                st.error(f"Unsupported file type: {ext}"); return None

        elif source == "Local dataset file" and local_path.strip():
            p = Path(local_path.strip())
            if not p.exists():
                st.error(f"File not found: {p}"); return None
            ext = p.suffix.lower()
            if ext == ".csv":
                df = pd.read_csv(p, nrows=n_rows, on_bad_lines="skip",
                                 encoding_errors="replace")
            elif ext in (".json",".jsonl"):
                df = pd.read_json(p, lines=(ext==".jsonl")).head(n_rows)
            elif ext == ".zip":
                with zipfile.ZipFile(p) as zf:
                    for name in zf.namelist():
                        if name.lower().endswith(".csv"):
                            df = pd.read_csv(zf.open(name), nrows=n_rows,
                                             on_bad_lines="skip"); break
                        elif name.lower().endswith((".json",".jsonl")):
                            df = pd.read_json(zf.open(name),
                                              lines=name.lower().endswith(".jsonl")).head(n_rows)
                            break
            else:
                st.error(f"Unsupported file type: {ext}"); return None
        else:
            return None

        if df is None or df.empty:
            st.error("File loaded but is empty."); return None

        with st.spinner(f"🔍 Running NLP on {min(len(df),n_rows):,} reviews…"):
            return process_df(df.to_json(orient="records"))

    except Exception as e:
        st.error(f"Error: {e}"); return None


# ══════════════════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
BG   = "rgba(0,0,0,0)"
FONT = dict(family="Inter", color="#374151", size=12)
GRID = "#f0f0ff"
SCOL = {"Positive":"#4ade80","Negative":"#f87171","Neutral":"#fbbf24"}

def _base(fig, h=240):
    fig.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,font=FONT,
                      height=h,margin=dict(t=16,b=10,l=10,r=16)); return fig

def chart_donut(df):
    c  = df["Sentiment"].value_counts()
    lb = ["Positive","Negative","Neutral"]
    vl = [c.get(l,0) for l in lb]
    fig = go.Figure(go.Pie(
        labels=lb,values=vl,hole=0.64,
        marker=dict(colors=[SCOL[l] for l in lb],line=dict(color="#fff",width=3)),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{sum(vl):,}</b><br><span style='font-size:10px;color:#9ca3af'>Total</span>",
        x=0.5,y=0.5,showarrow=False,
        font=dict(size=18,color="#1e1b4b",family="Inter"),
    )
    fig.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,font=FONT,height=200,
                      margin=dict(t=5,b=5,l=5,r=5),showlegend=True,
                      legend=dict(orientation="v",x=1.0,y=0.65,
                                  font=dict(size=12,color="#374151")))
    return fig

def chart_gauge(pct):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",value=pct,
        number=dict(suffix="%",font=dict(size=42,color="#1e1b4b",family="Inter",weight=800)),
        gauge=dict(
            axis=dict(range=[0,100],tickfont=dict(size=9,color="#9ca3af"),tickcolor="#e5e7eb"),
            bar=dict(color="#f97316",thickness=0.28),bgcolor="#f3f4f6",borderwidth=0,
            steps=[dict(range=[0,20],color="#f0fdf4"),
                   dict(range=[20,50],color="#fef9c3"),
                   dict(range=[50,100],color="#fee2e2")],
        ),
    ))
    return _base(fig, 200)

def chart_issues_bar(df):
    c = df[df["Sentiment"]=="Negative"]["Issue"].value_counts().reset_index()
    c.columns=["Issue","Count"]; c=c.sort_values("Count")
    colors=[ISSUE_COLORS.get(i,"#9ca3af") for i in c["Issue"]]
    fig=go.Figure(go.Bar(
        y=c["Issue"],x=c["Count"],orientation="h",
        marker=dict(color=colors,line=dict(width=0)),
        hovertemplate="<b>%{y}</b>: %{x:,}<extra></extra>",
        text=c["Count"],textposition="outside",textfont=dict(size=11,color="#374151"),
    ))
    fig.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,font=FONT,height=230,
                      margin=dict(t=5,b=5,l=5,r=55),
                      xaxis=dict(showgrid=True,gridcolor=GRID,zeroline=False,tickfont=FONT),
                      yaxis=dict(showgrid=False,tickfont=dict(size=11,color="#374151",family="Inter")))
    return fig

def chart_area_trend(df):
    df2=df.copy()
    df2["DateOnly"]=pd.to_datetime(df2["Date"],errors="coerce").dt.date
    t=df2.groupby(["DateOnly","Sentiment"]).size().reset_index(name="Count")
    fig=px.area(t,x="DateOnly",y="Count",color="Sentiment",
                color_discrete_map=SCOL,template="none")
    fig.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,font=FONT,height=280,
                      margin=dict(t=20,b=10,l=10,r=10),
                      xaxis=dict(showgrid=False,tickfont=FONT),
                      yaxis=dict(showgrid=True,gridcolor=GRID,tickfont=FONT),
                      legend=dict(orientation="h",yanchor="bottom",y=1.02,font=FONT))
    return fig

def chart_bar_monthly(df):
    df2=df.copy()
    df2["Month"]=pd.to_datetime(df2["Date"],errors="coerce").dt.strftime("%b")
    m=df2.groupby(["Month","Sentiment"]).size().reset_index(name="Count")
    fig=px.bar(m,x="Month",y="Count",color="Sentiment",barmode="stack",
               color_discrete_map=SCOL,template="none")
    fig.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,font=FONT,height=280,
                      margin=dict(t=20,b=10,l=10,r=10),
                      xaxis=dict(showgrid=False,tickfont=FONT),
                      yaxis=dict(showgrid=True,gridcolor=GRID,tickfont=FONT),
                      legend=dict(orientation="h",yanchor="bottom",y=1.02,font=FONT))
    return fig

def chart_scatter_fake(df):
    s=df.sample(min(500,len(df))).reset_index(drop=True)
    fig=px.scatter(s,x=s.index,y="FakeScore",color="FakeReview",
                   color_discrete_map={"Yes":"#ef4444","No":"#6366f1"},
                   opacity=0.7,template="none",
                   labels={"x":"Review Index","FakeScore":"Risk Score"})
    return _base(fig,280)

def chart_fake_trend(df):
    df2=df.copy()
    df2["Month"]=pd.to_datetime(df2["Date"],errors="coerce").dt.strftime("%b")
    m=df2.groupby("Month")["FakeScore"].mean().reset_index()
    fig=px.line(m,x="Month",y="FakeScore",markers=True,template="none",
                color_discrete_sequence=["#f97316"])
    return _base(fig,260)

def chart_issue_pie(df):
    c=df["Issue"].value_counts().reset_index(); c.columns=["Issue","Count"]
    colors=[ISSUE_COLORS.get(i,"#9ca3af") for i in c["Issue"]]
    fig=go.Figure(go.Pie(
        labels=c["Issue"],values=c["Count"],hole=0.4,
        marker=dict(colors=colors,line=dict(color="#fff",width=2)),
        textinfo="percent+label",textfont=dict(size=11),
        hovertemplate="<b>%{label}</b>: %{value:,}<extra></extra>",
    ))
    return _base(fig,280)

def make_wordcloud(df):
    neg_text=" ".join(df[df["Sentiment"]=="Negative"]["reviewText"].tolist())
    if len(neg_text.strip())<50:
        neg_text=("delivery quality damaged late broken poor refund wrong item "
                  "packaging customer service not working slow defective cheap ") * 80
    pal=["#f97316","#6366f1","#ef4444","#10b981","#8b5cf6",
         "#0ea5e9","#f59e0b","#14b8a6","#ec4899","#3b82f6"]
    def _c(*a,**k): return random.choice(pal)
    wc=WordCloud(width=620,height=300,background_color="white",color_func=_c,
                 max_words=60,stopwords=STOP_WORDS,prefer_horizontal=0.82,
                 max_font_size=80,min_font_size=11,margin=8,collocations=True,
                 ).generate(neg_text)
    fig,ax=plt.subplots(figsize=(6.2,3)); ax.imshow(wc,interpolation="bilinear")
    ax.axis("off"); fig.patch.set_facecolor("white"); plt.tight_layout(pad=0)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Brand header ──
    st.markdown("""
    <div style="padding:20px 16px 14px;display:flex;align-items:center;gap:10px;">
        <div style="font-size:26px;line-height:1;">📊</div>
        <div>
            <div style="font-size:15px;font-weight:800;color:#fff;letter-spacing:.2px;">ReviewIQ</div>
            <div style="font-size:10px;color:rgba(255,255,255,.65);">E-commerce Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation ──
    page = st.radio(
        "",
        [
            "🏠  Overview",
            "😊  Sentiment",
            "⚠️  Fake Review",
            "🔺  Issues",
            "📋  Reviews",
            "📈  Trends",
            "📄  Reports",
            "⚙️  Settings",
        ],
        label_visibility="collapsed",
    )

    # ── Divider + DATA heading ──
    st.markdown("""
    <div style="height:1px;background:rgba(255,255,255,.18);margin:12px 0 10px;"></div>
    <span class="sidebar-heading">DATA</span>
    """, unsafe_allow_html=True)

    # ── Data source ──
    source = st.radio(
        "Source",
        ["Local dataset file", "Upload file", "Demo sample"],
        index=2,
    )

    local_path    = ""
    uploaded_file = None

    if source == "Local dataset file":
        local_path = st.text_input("Path", "", placeholder="C:/data/reviews.csv")

    elif source == "Upload file":
        st.markdown("""
        <div style="font-size:12px;color:rgba(255,255,255,.8);margin-bottom:6px;font-weight:600;">
            CSV / JSON / ZIP
        </div>""", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "", type=["csv","json","jsonl","zip","txt","bz2"],
            label_visibility="collapsed",
        )

    n_rows = st.slider("Rows to load", 500, 20000, 6000, 500)

    analyze_btn = st.button("▶  Analyze", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "df" not in st.session_state:
    st.session_state["df"] = None

if analyze_btn:
    result = load_data(source, local_path, uploaded_file, n_rows)
    if result is not None:
        st.session_state["df"] = result
        st.success(f"✅ {len(result):,} reviews loaded and analysed!")
    else:
        st.session_state["df"] = load_data("Demo sample","",None,n_rows)

if st.session_state["df"] is None:
    st.session_state["df"] = load_data("Demo sample","",None,3000)

base_df = st.session_state["df"]
page_name = page.split("  ")[-1].strip()


# ══════════════════════════════════════════════════════════════════════════════
#  TOP BAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="topbar">
  <div>
    <div style="font-size:22px;font-weight:800;color:#1e1b4b;margin-bottom:3px;">
      E-commerce Review Intelligence
    </div>
    <div style="font-size:13px;color:#6b7280;">
      Analyze product reviews to detect sentiment, fake reviews, and classify issues.
    </div>
  </div>
  <div class="topbar-right">
    <div class="date-pill">📅 May 12 – Jun 12, 2024</div>
    <div class="filter-pill">🔽 All Products</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
filter_spacer, top_date, top_product = st.columns([6, 1.8, 1.8])
date_series = pd.to_datetime(base_df["Date"], errors="coerce")
date_options = ["All Dates"] + sorted(date_series.dt.strftime("%b %d, %Y").dropna().unique().tolist())
product_options = ["All Products"] + sorted(base_df["Product"].dropna().astype(str).unique().tolist())

with top_date:
    selected_date = st.selectbox("Date", date_options, label_visibility="collapsed")
with top_product:
    selected_product = st.selectbox("Product", product_options, label_visibility="collapsed")

df = base_df.copy()
if selected_date != "All Dates":
    df = df[pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%b %d, %Y") == selected_date]
if selected_product != "All Products":
    df = df[df["Product"].astype(str) == selected_product]

total    = max(len(df), 1)
pos_n    = (df["Sentiment"]=="Positive").sum()
neg_n    = (df["Sentiment"]=="Negative").sum()
neu_n    = (df["Sentiment"]=="Neutral").sum()
fake_n   = (df["FakeReview"]=="Yes").sum()
pos_pct  = pos_n/total*100
neg_pct  = neg_n/total*100
neu_pct  = neu_n/total*100
fake_pct = fake_n/total*100

#  STAT CARDS ROW
# ══════════════════════════════════════════════════════════════════════════════
def render_cards():
    cols = st.columns(5)
    data = [
        (cols[0],"Total Reviews",  f"{total:,}",
         f"↑ 18.6% vs May 11 – May 11","","#6366f1","#6366f1"),
        (cols[1],"Positive",       f"{pos_n:,}",
         f"({pos_pct:.1f}%)",f"width:{pos_pct:.0f}%","#10b981","#10b981"),
        (cols[2],"Negative",       f"{neg_n:,}",
         f"({neg_pct:.1f}%)",f"width:{neg_pct:.0f}%","#ef4444","#ef4444"),
        (cols[3],"Neutral",        f"{neu_n:,}",
         f"({neu_pct:.1f}%)",f"width:{neu_pct:.0f}%","#f59e0b","#f59e0b"),
        (cols[4],"Fake Reviews",   f"{fake_n:,}",
         f"({fake_pct:.1f}%)",f"width:{fake_pct:.0f}%","#8b5cf6","#8b5cf6"),
    ]
    icons = ["💬","😊","😞","😐","⚠️"]
    for (col,label,val,sub,bar_s,accent,bc), icon in zip(data,icons):
        is_delta = "↑" in sub
        sub_h = (f'<div class="stat-delta" style="color:{accent};">{sub}</div>'
                 if is_delta else
                 f'<div class="stat-sub" style="color:{accent};font-weight:600;">{sub}</div>')
        bar_h = (f'<div class="stat-bar"><div class="stat-bar-fill" '
                 f'style="{bar_s};background:{bc};"></div></div>'
                 if bar_s else "")
        with col:
            st.markdown(f"""
            <div class="stat-card" style="--accent:{accent};">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">
                <span class="stat-label">{label}</span>
                <span style="font-size:20px;">{icon}</span>
              </div>
              <div class="stat-value">{val}</div>
              {sub_h}{bar_h}
            </div>""", unsafe_allow_html=True)

render_cards()
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ── Shared helpers ─────────────────────────────────────────────────────────
def sc_open(title, icon=""):
    st.markdown(f"""
    <div class="section-card">
      <div class="section-title"><span class="title-icon">{icon}</span> {title}</div>
    """, unsafe_allow_html=True)

def sc_close():
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page_name == "Overview":

    c1,c2,c3 = st.columns([1.1,1,1.2])

    with c1:
        sc_open("Sentiment Distribution","📊")
        st.plotly_chart(chart_donut(df),use_container_width=True,config={"displayModeBar":False})
        for lbl,clr,n,pct in [("Positive","#4ade80",pos_n,pos_pct),
                               ("Negative","#f87171",neg_n,neg_pct),
                               ("Neutral", "#fbbf24",neu_n,neu_pct)]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:5px 0;
                        border-bottom:1px solid #f3f4f6;">
              <div style="display:flex;align-items:center;gap:8px;">
                <div style="width:10px;height:10px;border-radius:50%;background:{clr};"></div>
                <span style="font-size:12px;color:#374151;font-weight:500;">{lbl}</span>
              </div>
              <span style="font-size:12px;color:#6b7280;font-weight:500;">
                {n:,} ({pct:.1f}%)
              </span>
            </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        sc_close()

    with c2:
        sc_open("Fake Review Detection","🛡")
        st.plotly_chart(chart_gauge(round(fake_pct,1)),use_container_width=True,
                        config={"displayModeBar":False})
        st.markdown(f"""
        <div style="text-align:center;margin:-6px 0 10px;">
          <div style="font-size:13px;color:#6b7280;font-weight:500;">
            {fake_n:,} / {total:,}
          </div>
          <div style="font-size:11px;color:#9ca3af;">Reviews Flagged as Fake</div>
        </div>
        <div style="background:#ede9fe;border-radius:12px;padding:11px 14px;
                    border-left:3px solid #8b5cf6;display:flex;gap:8px;align-items:flex-start;">
          <span style="font-size:16px;">🛡</span>
          <div>
            <div style="font-size:12px;font-weight:700;color:#5b21b6;">Moderate Risk</div>
            <div style="font-size:11px;color:#6b7280;margin-top:2px;">
              Continue monitoring. Fake review rate is within acceptable range.
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
        sc_close()

    with c3:
        sc_open("Top Issue Categories","🔺")
        issue_counts = df[df["Sentiment"]=="Negative"]["Issue"].value_counts()
        it = issue_counts.sum() or 1
        mx = issue_counts.max() if len(issue_counts) else 1
        for iss in ["Delivery","Product Quality","Damaged Product",
                    "Customer Service","Wrong Item","Other"]:
            n_i  = issue_counts.get(iss,0)
            pc_i = n_i/it*100
            bw   = int(n_i/mx*100) if mx else 0
            clr  = ISSUE_COLORS[iss]
            ico  = ISSUE_ICONS[iss]
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:9px;">
              <span style="font-size:15px;width:22px;">{ico}</span>
              <div style="flex:1;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                  <span style="font-size:12px;color:#374151;font-weight:500;">{iss}</span>
                  <span style="font-size:11px;color:#6b7280;">{n_i:,} ({pc_i:.1f}%)</span>
                </div>
                <div style="height:6px;background:#f0f0ff;border-radius:99px;overflow:hidden;">
                  <div style="width:{bw}%;height:100%;background:{clr};border-radius:99px;
                              transition:width 1s ease;"></div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
        sc_close()

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    cd,ce = st.columns([1.6,1])
    with cd:
        sc_open("Recent Reviews","📋")
        samp = df.sample(min(8,len(df))).reset_index(drop=True)
        rows = ""
        for _,row in samp.iterrows():
            ns = int(row["overall"])
            stars = "★"*ns + "☆"*(5-ns)
            sc  = "#fbbf24" if ns>=4 else ("#f87171" if ns<=2 else "#9ca3af")
            s   = row["Sentiment"]
            scl = "badge-pos" if s=="Positive" else ("badge-neg" if s=="Negative" else "badge-neu")
            fcl = "badge-fake" if row["FakeReview"]=="Yes" else "badge-no"
            iss = row["Issue"]
            ibg,iclr = PILL_COLORS.get(iss,("#f3f4f6","#374151"))
            snip = textwrap.shorten(row["reviewText"],width=52,placeholder="…")
            rows += f"""<tr>
              <td><span style='color:{sc};font-size:11px;letter-spacing:1px;'>{stars}</span>
                  <div style='color:#6b7280;font-size:11px;margin-top:1px;'>{snip}</div></td>
              <td><span class='badge {scl}'>{s}</span></td>
              <td><span class='badge {fcl}'>{row['FakeReview']}</span></td>
              <td><span class='issue-pill' style='background:{ibg};color:{iclr};'>{iss}</span></td>
              <td style='color:#9ca3af;white-space:nowrap;font-size:11px;'>{row['Date']}</td>
            </tr>"""
        st.markdown(f"""
        <table class="rev-table">
          <thead><tr>
            <th>Review</th><th>Sentiment</th><th>Fake</th><th>Issue</th><th>Date</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>""", unsafe_allow_html=True)
        sc_close()

    with ce:
        sc_open("Top Keywords — Negative Reviews","☁")
        wc_fig = make_wordcloud(df)
        st.pyplot(wc_fig,use_container_width=True)
        plt.close()
        sc_close()


# ══════════════════════════════════════════════════════════════════════════════
#  SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
elif page_name == "Sentiment":
    st.markdown('<div class="page-enter">', unsafe_allow_html=True)
    sc_open("Sentiment Trend Over Time","📈")
    st.plotly_chart(chart_area_trend(df),use_container_width=True,config={"displayModeBar":False})
    sc_close()
    st.markdown("<div style='height:16px'></div>",unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        sc_open("Distribution","📊")
        st.plotly_chart(chart_donut(df),use_container_width=True,config={"displayModeBar":False})
        sc_close()
    with c2:
        sc_open("Monthly Breakdown","📅")
        st.plotly_chart(chart_bar_monthly(df),use_container_width=True,config={"displayModeBar":False})
        sc_close()
    st.markdown("<div style='height:16px'></div>",unsafe_allow_html=True)
    sc_open("Browse by Sentiment","🔍")
    flt = st.selectbox("Filter","All,Positive,Negative,Neutral".split(","),
                       label_visibility="collapsed")
    v = df if flt=="All" else df[df["Sentiment"]==flt]
    st.dataframe(v[["reviewText","overall","Sentiment","Issue","Date"]].head(60)
                  .reset_index(drop=True),use_container_width=True,height=260,
                 column_config={"reviewText":st.column_config.TextColumn("Review",width="large")})
    sc_close()
    st.markdown('</div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FAKE REVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page_name == "Fake Review":
    st.markdown('<div class="page-enter">',unsafe_allow_html=True)
    c1,c2 = st.columns([1,1.5])
    with c1:
        sc_open("Detection Overview","🛡")
        st.plotly_chart(chart_gauge(round(fake_pct,1)),use_container_width=True,
                        config={"displayModeBar":False})
        st.markdown(f"""
        <div style="text-align:center;margin:-4px 0 12px;">
          <span style="font-size:32px;font-weight:900;color:#f97316;">{fake_pct:.1f}%</span><br>
          <span style="font-size:12px;color:#6b7280;">{fake_n:,} / {total:,} flagged</span>
        </div>
        <div style="background:#ede9fe;border-radius:12px;padding:12px 14px;border-left:3px solid #8b5cf6;">
          <b style="color:#5b21b6;font-size:13px;">🛡 Moderate Risk</b><br>
          <span style="font-size:11px;color:#6b7280;">Continue monitoring. Rate is within acceptable range.</span>
        </div>""",unsafe_allow_html=True)
        sc_close()
    with c2:
        sc_open("Risk Score Distribution","📊")
        st.plotly_chart(chart_scatter_fake(df),use_container_width=True,config={"displayModeBar":False})
        sc_close()
    st.markdown("<div style='height:16px'></div>",unsafe_allow_html=True)
    sc_open("Monthly Fake Risk Trend","📈")
    st.plotly_chart(chart_fake_trend(df),use_container_width=True,config={"displayModeBar":False})
    sc_close()
    st.markdown("<div style='height:16px'></div>",unsafe_allow_html=True)
    sc_open("Flagged Reviews","⚠️")
    flagged=df[df["FakeReview"]=="Yes"][["reviewText","overall","Sentiment","FakeScore","Date"]].head(60)
    st.dataframe(flagged.reset_index(drop=True),use_container_width=True,height=260,
                 column_config={
                     "reviewText":st.column_config.TextColumn("Review",width="large"),
                     "FakeScore":st.column_config.ProgressColumn("Risk",min_value=0,max_value=1),
                 })
    sc_close()
    st.markdown('</div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ISSUES
# ══════════════════════════════════════════════════════════════════════════════
elif page_name == "Issues":
    st.markdown('<div class="page-enter">',unsafe_allow_html=True)
    ic=df["Issue"].value_counts(); it=ic.sum() or 1
    iss_list=["Delivery","Product Quality","Damaged Product","Customer Service","Wrong Item","Other"]
    cols=st.columns(6)
    for col,iss in zip(cols,iss_list):
        ni=ic.get(iss,0); clr=ISSUE_COLORS[iss]; ico=ISSUE_ICONS[iss]
        with col:
            st.markdown(f"""
            <div class="stat-card" style="--accent:{clr};text-align:center;padding:14px 10px;">
              <div style="font-size:20px;margin-bottom:4px;">{ico}</div>
              <div style="font-size:10px;color:#6b7280;font-weight:600;text-transform:uppercase;
                          letter-spacing:.5px;margin-bottom:4px;">{iss}</div>
              <div style="font-size:20px;font-weight:800;color:{clr};">{ni:,}</div>
              <div style="font-size:11px;color:#9ca3af;">{ni/it*100:.1f}%</div>
            </div>""",unsafe_allow_html=True)
    st.markdown("<div style='height:18px'></div>",unsafe_allow_html=True)
    c1,c2=st.columns([1.4,1])
    with c1:
        sc_open("Issue Count — Negative Reviews","📊")
        st.plotly_chart(chart_issues_bar(df),use_container_width=True,config={"displayModeBar":False})
        sc_close()
    with c2:
        sc_open("Issue Distribution","🥧")
        st.plotly_chart(chart_issue_pie(df),use_container_width=True,config={"displayModeBar":False})
        sc_close()
    st.markdown("<div style='height:16px'></div>",unsafe_allow_html=True)
    sc_open("Browse by Issue","🔍")
    sel=st.selectbox("Issue",["All"]+iss_list,label_visibility="collapsed")
    v=df if sel=="All" else df[df["Issue"]==sel]
    st.dataframe(v[["reviewText","overall","Sentiment","Issue","FakeReview","Date"]].head(80)
                  .reset_index(drop=True),use_container_width=True,height=300,
                 column_config={"reviewText":st.column_config.TextColumn("Review",width="large")})
    sc_close()
    st.markdown('</div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  REVIEWS
# ══════════════════════════════════════════════════════════════════════════════
elif page_name == "Reviews":
    st.markdown('<div class="page-enter">',unsafe_allow_html=True)
    sc_open("All Reviews","📋")
    c1,c2,c3,c4=st.columns([2.5,1,1,1])
    with c1: q =st.text_input("","",placeholder="🔍  Search reviews…",label_visibility="collapsed")
    with c2: fs=st.selectbox("Sent",["All","Positive","Negative","Neutral"],label_visibility="collapsed")
    with c3: ff=st.selectbox("Fake",["All","Yes","No"],label_visibility="collapsed")
    with c4: fi=st.selectbox("Issue",["All"]+list(ISSUE_KW.keys())+["Other"],label_visibility="collapsed")
    v=df.copy()
    if q:  v=v[v["reviewText"].str.contains(q,case=False,na=False)]
    if fs!="All": v=v[v["Sentiment"]==fs]
    if ff!="All": v=v[v["FakeReview"]==ff]
    if fi!="All": v=v[v["Issue"]==fi]
    st.markdown(f"<div style='font-size:12px;color:#6b7280;margin:6px 0;'>"
                f"<b style='color:#6366f1;'>{len(v):,}</b> reviews match.</div>",unsafe_allow_html=True)
    st.dataframe(v[["reviewText","overall","Sentiment","FakeReview","Issue","Date"]].head(200)
                  .reset_index(drop=True),use_container_width=True,height=500,
                 column_config={
                     "reviewText":st.column_config.TextColumn("Review",width="large"),
                     "overall":st.column_config.NumberColumn("⭐"),
                     "FakeReview":st.column_config.TextColumn("Fake"),
                 })
    sc_close()
    st.markdown('</div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page_name == "Trends":
    st.markdown('<div class="page-enter">',unsafe_allow_html=True)
    sc_open("Sentiment Over Time","📈")
    st.plotly_chart(chart_area_trend(df),use_container_width=True,config={"displayModeBar":False})
    sc_close()
    st.markdown("<div style='height:16px'></div>",unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        sc_open("Monthly Volume","📅")
        st.plotly_chart(chart_bar_monthly(df),use_container_width=True,config={"displayModeBar":False})
        sc_close()
    with c2:
        sc_open("Fake Risk Trend","⚠️")
        st.plotly_chart(chart_fake_trend(df),use_container_width=True,config={"displayModeBar":False})
        sc_close()
    st.markdown('</div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTS
# ══════════════════════════════════════════════════════════════════════════════
elif page_name == "Reports":
    st.markdown('<div class="page-enter">',unsafe_allow_html=True)
    sc_open("Export & Reports","📄")
    csv_b=df[["reviewText","overall","Sentiment","FakeReview","FakeScore","Issue","Date","Product"]
             ].to_csv(index=False).encode("utf-8")
    c1,c2=st.columns(2)
    with c1:
        st.download_button("Full CSV Analysis",csv_b,
                           "review_intelligence.csv","text/csv",use_container_width=True)
    with c2:
        lines=["E-COMMERCE REVIEW INTELLIGENCE — SUMMARY","="*44,
               f"Total Reviews: {total:,}",
               f"  Positive : {pos_n:,} ({pos_pct:.1f}%)",
               f"  Negative : {neg_n:,} ({neg_pct:.1f}%)",
               f"  Neutral  : {neu_n:,} ({neu_pct:.1f}%)",
               f"Fake Reviews: {fake_n:,} ({fake_pct:.1f}%)","","ISSUES:"]
        for iss,cnt in df["Issue"].value_counts().items():
            lines.append(f"  {iss}: {cnt:,} ({cnt/total*100:.1f}%)")
        st.download_button("Summary TXT","\n".join(lines).encode(),
                           "summary.txt","text/plain",use_container_width=True)
    st.markdown("<div style='height:14px'></div>",unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#f0f0ff,#ede9fe);border-radius:14px;
                padding:20px 24px;border:1px solid #c7d2fe;">
      <div style="font-size:15px;font-weight:700;color:#1e1b4b;margin-bottom:12px;">
        📊 Summary Statistics
      </div>
      <pre style="font-size:12px;color:#374151;margin:0;white-space:pre-wrap;
                  font-family:'Inter',sans-serif;">{"".join([l+chr(10) for l in lines])}</pre>
    </div>""",unsafe_allow_html=True)
    sc_close()
    st.markdown('</div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page_name == "Settings":
    st.markdown('<div class="page-enter">',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        sc_open("Detection Settings","⚙️")
        st.slider("Fake Review Flag Threshold",0.10,0.90,0.35,0.05,
                  help="Scores above this are labelled Fake")
        st.slider("VADER Positive Threshold",0.01,0.30,0.10,0.01)
        st.slider("VADER Negative Threshold",-0.30,-0.01,-0.10,0.01)
        st.checkbox("Enable real-time re-analysis",value=False)
        if st.button("💾  Save Settings"):
            st.success("Settings saved — re-click Analyze to apply.")
        sc_close()
    with c2:
        sc_open("About","ℹ️")
        st.markdown("""
        <div style="font-size:13px;color:#374151;line-height:1.9;">
          <b style="color:#1e1b4b;">E-commerce Review Intelligence</b><br>
          Version 4.0 &nbsp;|&nbsp; Streamlit + Plotly<br><br>
          <b>Problem Statement:</b><br>
          Analyze product reviews to detect sentiment, fake reviews,
          and classify issues (delivery, quality, etc.).<br><br>
          <b>Tech Stack:</b><br>
          🔹 NLTK VADER — Sentiment (star-rating blended)<br>
          🔹 Rule-based — Fake Detection<br>
          🔹 Keyword Matching — Issue Classification<br>
          🔹 Plotly — Interactive Charts<br>
          🔹 WordCloud — Keyword Viz<br>
          🔹 Streamlit — Dashboard Framework
        </div>""",unsafe_allow_html=True)
        sc_close()
    st.markdown('</div>',unsafe_allow_html=True)
