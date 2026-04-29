"""
Phishing Detection - Streamlit UI
Run: streamlit run src/app.py
Requires: data/raw/PhiUSIIL_Phishing_URL_Dataset.csv (first run trains model).
"""

import streamlit as st
import pandas as pd
import numpy as np
import re, os, warnings, urllib.parse
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw"
MODELS_DIR = ROOT_DIR / "artifacts" / "models"

st.set_page_config(page_title="Phishing Detector", page_icon="🛡️", layout="centered")

st.markdown("""
<style>
    .stApp { background: #0f172a; }
    .main-title {
        text-align: center; font-size: 2.4rem; font-weight: 800;
        color: #f8fafc; margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center; color: #94a3b8;
        font-size: 1rem; margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem 2rem; border-radius: 14px;
        margin: 1.2rem 0; font-size: 1.1rem;
    }
    .legit-box { background: #052e16; border: 2px solid #16a34a; }
    .phish-box { background: #2d0a0a; border: 2px solid #dc2626; }
    .verdict   { font-size: 2rem; font-weight: 800; margin-bottom: 0.5rem; }
    .legit-txt { color: #4ade80; }
    .phish-txt { color: #f87171; }
    .conf-label { color: #94a3b8; font-size: 0.9rem; margin-top: 0.8rem; }
    .feature-grid {
        display: grid; grid-template-columns: 1fr 1fr;
        gap: 0.6rem; margin-top: 1rem;
    }
    .feat-card {
        background: #1e293b; border-radius: 10px;
        padding: 0.6rem 1rem; border-left: 4px solid #334155;
    }
    .feat-card.warn { border-left-color: #ef4444; }
    .feat-card.ok   { border-left-color: #22c55e; }
    .feat-name { color: #94a3b8; font-size: 0.75rem; }
    .feat-val  { color: #f1f5f9; font-size: 1rem; font-weight: 600; }
    .warning-tag { color: #f87171; font-size: 0.72rem; }
    .ok-tag      { color: #4ade80; font-size: 0.72rem; }
    .reason-box {
        background: #1e293b; border-radius: 10px;
        padding: 1rem 1.2rem; margin-top: 0.8rem;
    }
    .reason-item { color: #fbbf24; font-size: 0.88rem; margin: 0.3rem 0; }
    .reason-ok   { color: #4ade80; font-size: 0.88rem; margin: 0.3rem 0; }
    div[data-testid="stTextInput"] input {
        background: #1e293b !important; border: 1.5px solid #334155 !important;
        color: #f1f5f9 !important; font-size: 1rem !important;
        border-radius: 10px !important;
    }
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white; border: none; border-radius: 10px;
        font-weight: 700; font-size: 1rem;
        padding: 0.6rem 2rem; width: 100%;
    }
    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #2563eb, #4f46e5);
    }
</style>
""", unsafe_allow_html=True)


# ── Known legitimate base domains ─────────────────────────────
KNOWN_LEGIT = {
    'google','youtube','facebook','twitter','instagram','linkedin',
    'microsoft','apple','amazon','github','wikipedia','netflix',
    'paypal','ebay','reddit','stackoverflow','whatsapp','telegram',
    'zoom','dropbox','adobe','salesforce','shopify','wordpress',
    'pinterest','tumblr','snapchat','tiktok','discord','slack',
    'bing','yahoo','duckduckgo','cloudflare','stripe','twitch',
    'spotify','notion','figma','atlassian','gitlab','bitbucket',
}

TLD_PROB = {
    'com':0.5229,'org':0.0799,'net':0.0594,'edu':0.0274,'gov':0.0162,
    'uk':0.0388,'de':0.0295,'jp':0.0193,'fr':0.0182,'au':0.0172,
    'in':0.0051,'ca':0.0149,'ru':0.0130,'br':0.0096,'it':0.0094,
    'nl':0.0092,'es':0.0083,'pl':0.0070,'mx':0.0052,'info':0.0034,
    'io':0.0028,'co':0.0020,'me':0.0014,'tv':0.0008,'biz':0.0006,
    'xyz':0.00005,'tk':0.00003,'ml':0.00002,'ga':0.00001,'cf':0.00001,
}

MODEL_VERSION = 2


# ── Feature extractor ──────────────────────────────────────────
def extract_features(url: str) -> dict:
    full = url.strip()
    if not re.match(r'^https?://', full, re.I):
        full = 'http://' + full

    parsed        = urllib.parse.urlparse(full)
    hostname_full = parsed.netloc.lower()
    hostname      = re.sub(r':\d+$', '', hostname_full)
    url_len       = len(full)
    letters       = re.findall(r'[a-zA-Z]', full)
    digits        = re.findall(r'\d', full)
    specials      = re.findall(r'[^a-zA-Z0-9]', full)
    is_https      = 1 if full.lower().startswith('https') else 0

    parts      = hostname.split('.')
    tld        = parts[-1] if parts else ''
    sld        = parts[-2] if len(parts) >= 2 else ''
    extra_subs = max(0, len(parts) - 2)
    n_subs     = len(parts) - 1
    base_domain = sld.lower()
    is_known    = base_domain in KNOWN_LEGIT

    # URLSimilarityIndex
    if is_known:
        usi = 100.0
    else:
        usi = 50.0
        if is_https:                                usi += 15.0
        if tld in ('com','org','net','edu','gov'):  usi += 10.0
        if extra_subs == 0:                         usi += 10.0
        if hostname.count('-') == 0:                usi += 5.0
        if url_len < 50:                            usi += 5.0
        if not re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname): usi += 5.0
        usi = round(min(99.0, usi), 2)

    consec  = sum(1 for i in range(len(full)-1)
                  if full[i].isalpha() and full[i+1].isalpha())
    ccr      = consec / max(url_len, 1)
    tld_prob = TLD_PROB.get(tld, 0.00001)
    ucp      = round(min(0.09, len(letters) / max(url_len**1.2, 1)), 6)

    utms = 90.0 if is_known else 50.0
    if is_https and not is_known: utms += 10.0
    if extra_subs == 0:           utms += 10.0
    if tld_prob < 0.001:          utms -= 30.0
    if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname): utms -= 50.0
    if url_len > 75:              utms -= 15.0
    utms -= extra_subs * 8.0
    utms -= hostname.count('-') * 5.0
    utms  = round(max(0.0, min(100.0, utms)), 6)

    n_other_special = len(re.findall(r'[^a-zA-Z0-9./:_\-?=&@#%~]', full))
    spatial_ratio   = len(specials) / max(url_len, 1)
    letter_ratio    = len(letters)  / max(url_len, 1)
    digit_ratio     = len(digits)   / max(url_len, 1)
    obf_chars       = len(re.findall(r'%[0-9A-Fa-f]{2}', full))
    obf_ratio       = obf_chars / max(url_len, 1)

    return {
        'URLSimilarityIndex'         : usi,
        'CharContinuationRate'       : round(ccr, 6),
        'TLDLegitimateProb'          : round(tld_prob, 6),
        'URLCharProb'                : round(ucp, 6),
        'IsHTTPS'                    : is_https,
        'NoOfSubDomain'              : n_subs,
        'URLTitleMatchScore'         : utms,
        'NoOfOtherSpecialCharsInURL' : n_other_special,
        'SpacialCharRatioInURL'      : round(spatial_ratio, 6),
        'LetterRatioInURL'           : round(letter_ratio, 6),
        'DegitRatioInURL'            : round(digit_ratio, 6),
        'URLLength'                  : url_len,
        'NoOfLettersInURL'           : len(letters),
        'DomainLength'               : len(hostname_full),
        'NoOfDegitsInURL'            : len(digits),
        'IsDomainIP'                 : 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname) else 0,
        'ObfuscationRatio'           : round(obf_ratio, 6),
        'NoOfQMarkInURL'             : full.count('?'),
        'NoOfAmpersandInURL'         : full.count('&'),
        'NoOfEqualsInURL'            : full.count('='),
    }


def rule_based_override(raw: dict):
    """
    Override ML when signals are crystal clear.
    Returns (pred, confidence) or (None, None) to fall through to ML.
    """
    r = raw
    # Hard PHISHING signals
    if r.get('IsDomainIP', 0) == 1:
        return 0, 95.0
    if r.get('ObfuscationRatio', 0) > 0.05:
        return 0, 92.0
    if r.get('TLDLegitimateProb', 1) < 0.0001 and r.get('IsHTTPS', 1) == 0:
        return 0, 90.0
    if r.get('URLSimilarityIndex', 100) < 60 and r.get('IsHTTPS', 1) == 0:
        return 0, 88.0

    # Hard LEGITIMATE signals — ALL must hold
    all_green = (
        r.get('URLSimilarityIndex', 0)  == 100.0  and
        r.get('IsHTTPS', 0)             == 1       and
        r.get('IsDomainIP', 1)          == 0       and
        r.get('TLDLegitimateProb', 0)   >= 0.001   and
        r.get('ObfuscationRatio', 1)    == 0.0     and
        r.get('URLTitleMatchScore', 0)  >= 70.0    and
        r.get('NoOfSubDomain', 99)      <= 2       and
        r.get('NoOfQMarkInURL', 99)     == 0
    )
    if all_green:
        return 1, 97.0

    return None, None  # defer to ML model


def likely_legit_profile(raw: dict) -> bool:
    """
    Conservative profile for legitimate URLs used only to reduce false positives
    on borderline ML outputs.
    """
    return (
        raw.get('IsHTTPS', 0) == 1 and
        raw.get('IsDomainIP', 1) == 0 and
        raw.get('ObfuscationRatio', 1.0) == 0.0 and
        raw.get('TLDLegitimateProb', 0.0) >= 0.0005 and
        raw.get('NoOfSubDomain', 99) <= 3 and
        raw.get('URLLength', 999) <= 120 and
        raw.get('NoOfQMarkInURL', 99) <= 1 and
        raw.get('NoOfAmpersandInURL', 99) <= 2 and
        raw.get('SpacialCharRatioInURL', 1.0) < 0.2 and
        raw.get('URLSimilarityIndex', 0.0) >= 75.0 and
        raw.get('URLTitleMatchScore', 0.0) >= 45.0
    )


def strong_phishing_signals(raw: dict) -> bool:
    """
    Keep clearly dangerous URLs classified as phishing even when
    applying false-positive reduction logic.
    """
    return (
        raw.get('IsDomainIP', 0) == 1 or
        raw.get('ObfuscationRatio', 0.0) > 0.02 or
        (raw.get('IsHTTPS', 1) == 0 and raw.get('TLDLegitimateProb', 1.0) < 0.0001) or
        raw.get('URLSimilarityIndex', 100.0) < 45.0
    )


def safe_legit_override_profile(raw: dict) -> bool:
    """
    Conservative allow-list style profile used to reduce false positives
    on normal HTTPS websites.
    """
    return (
        raw.get('IsHTTPS', 0) == 1 and
        raw.get('IsDomainIP', 1) == 0 and
        raw.get('ObfuscationRatio', 1.0) == 0.0 and
        raw.get('TLDLegitimateProb', 0.0) >= 0.0005 and
        raw.get('URLSimilarityIndex', 0.0) >= 85.0 and
        raw.get('URLTitleMatchScore', 0.0) >= 55.0 and
        raw.get('URLLength', 999) <= 160 and
        raw.get('NoOfSubDomain', 99) <= 4 and
        raw.get('NoOfQMarkInURL', 99) <= 2 and
        raw.get('NoOfAmpersandInURL', 99) <= 3 and
        raw.get('SpacialCharRatioInURL', 1.0) < 0.22
    )


@st.cache_resource(show_spinner="Loading model (first run trains from CSV)...")
def load_model():
    import joblib
    RF_MODEL_PATH = MODELS_DIR / "phishing_rf_model.pkl"
    FEAT_PATH = MODELS_DIR / "selected_features.pkl"
    MODEL_PATH = MODELS_DIR / "phishing_model.pkl"

    # Prefer pre-trained artifacts exported by phishing_detection.py
    if RF_MODEL_PATH.exists() and FEAT_PATH.exists():
        try:
            model = joblib.load(RF_MODEL_PATH)
            features = joblib.load(FEAT_PATH)
            if model is not None and isinstance(features, list) and len(features) > 0:
                return model, features
        except Exception:
            pass

    if MODEL_PATH.exists():
        try:
            data = joblib.load(MODEL_PATH)
            if (
                isinstance(data, dict) and
                data.get("model_version") == MODEL_VERSION and
                "model" in data and
                "features" in data
            ):
                return data["model"], data["features"]
        except Exception:
            pass

    CSV_PATH = DATA_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
    if not CSV_PATH.exists():
        return None, None

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    URL_FEATURES = [
        'URLSimilarityIndex','CharContinuationRate','TLDLegitimateProb',
        'URLCharProb','IsHTTPS','NoOfSubDomain','URLTitleMatchScore',
        'NoOfOtherSpecialCharsInURL','SpacialCharRatioInURL',
        'LetterRatioInURL','DegitRatioInURL','URLLength',
        'NoOfLettersInURL','DomainLength','NoOfDegitsInURL',
        'IsDomainIP','ObfuscationRatio','NoOfQMarkInURL',
        'NoOfAmpersandInURL','NoOfEqualsInURL'
    ]
    df = pd.read_csv(CSV_PATH)
    URL_FEATURES = [f for f in URL_FEATURES if f in df.columns]
    X = df[URL_FEATURES].copy()
    y = df['label'].copy()
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": URL_FEATURES,
        "model_version": MODEL_VERSION,
    }, MODEL_PATH)
    return model, URL_FEATURES


def predict_url(url, model, features):
    raw = extract_features(url)

    # Try rule-based override first
    pred_ov, conf_ov = rule_based_override(raw)
    if pred_ov is not None:
        p_legit = conf_ov if pred_ov == 1 else (100.0 - conf_ov)
        p_phish = 100.0 - p_legit
        return pred_ov, p_legit, p_phish, raw

    # Fall back to ML model
    row   = {f: raw.get(f, 0) for f in features}
    X_in  = pd.DataFrame([row])[features]
    proba = model.predict_proba(X_in)[0]
    pred  = model.predict(X_in)[0]

    # Reduce false positives on borderline outputs while preserving
    # clearly dangerous URLs.
    p_phish = proba[0] * 100
    p_legit = proba[1] * 100
    if pred == 0 and not strong_phishing_signals(raw):
        if safe_legit_override_profile(raw) and p_phish < 92.0:
            pred = 1
            p_legit = max(p_legit, 70.0)
            p_phish = 100.0 - p_legit
            return pred, p_legit, p_phish, raw

        if likely_legit_profile(raw) and p_phish < 85.0:
            pred = 1
            p_legit = max(p_legit, 65.0)
            p_phish = 100.0 - p_legit
            return pred, p_legit, p_phish, raw

    return pred, p_legit, p_phish, raw


# ══════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🛡️ Phishing Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter any website URL to check if it\'s safe or a phishing attempt</div>',
            unsafe_allow_html=True)

model, features = load_model()

url_input     = st.text_input("URL",
                               placeholder="e.g. https://www.google.com  or  http://paypal-secure.xyz/login",
                               label_visibility="collapsed")
check_clicked = st.button("🔍 Check URL")

# ── Result ─────────────────────────────────────────────────────
if check_clicked:
    if not url_input.strip():
        st.warning("Please enter a URL first.")
    elif model is None:
        st.error("❌ Model not loaded. Place PhiUSIIL_Phishing_URL_Dataset.csv in data/raw and restart.")
    else:
        with st.spinner("Analysing URL..."):
            pred, p_legit, p_phish, raw = predict_url(url_input.strip(), model, features)

        is_legit = pred == 1
        box_cls  = "legit-box" if is_legit else "phish-box"
        txt_cls  = "legit-txt" if is_legit else "phish-txt"
        icon     = "✅" if is_legit else "🚨"
        label    = "LEGITIMATE" if is_legit else "PHISHING"
        conf     = p_legit if is_legit else p_phish

        st.markdown(f"""
        <div class="result-box {box_cls}">
            <div class="verdict {txt_cls}">{icon} {label}</div>
            <div class="conf-label">Confidence: <b style="color:{'#4ade80' if is_legit else '#f87171'}">{conf:.1f}%</b></div>
            <div class="conf-label" style="margin-top:0.4rem">
                Legitimate: {p_legit:.1f}% &nbsp;|&nbsp; Phishing: {p_phish:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature cards
        st.markdown("#### 🔎 Feature Breakdown")
        FEAT_DISPLAY = [
            ("URLSimilarityIndex",  "URL Similarity",    lambda v: v < 80,   lambda v: f"{v:.1f} / 100"),
            ("URLTitleMatchScore",  "Title Match Score", lambda v: v < 40,   lambda v: f"{v:.1f} / 100"),
            ("IsHTTPS",             "HTTPS",             lambda v: v == 0,   lambda v: "Yes ✅" if v==1 else "No ❌"),
            ("IsDomainIP",          "IP as Domain",      lambda v: v == 1,   lambda v: "Yes ❌" if v==1 else "No ✅"),
            ("TLDLegitimateProb",   "TLD Reputation",    lambda v: v < 0.01, lambda v: f"{v:.4f}"),
            ("NoOfSubDomain",       "Subdomains",        lambda v: v > 3,    lambda v: str(int(v))),
            ("DomainLength",        "Domain Length",     lambda v: v > 30,   lambda v: str(int(v))),
            ("URLLength",           "URL Length",        lambda v: v > 75,   lambda v: str(int(v))),
            ("DegitRatioInURL",     "Digit Ratio",       lambda v: v > 0.1,  lambda v: f"{v:.3f}"),
            ("ObfuscationRatio",    "Obfuscation",       lambda v: v > 0,    lambda v: f"{v:.4f}"),
            ("NoOfQMarkInURL",      "? Count",           lambda v: v > 0,    lambda v: str(int(v))),
            ("NoOfAmpersandInURL",  "& Count",           lambda v: v > 0,    lambda v: str(int(v))),
        ]

        cards_html = '<div class="feature-grid">'
        for feat, lbl, is_warn, fmt in FEAT_DISPLAY:
            if feat in raw:
                val      = raw[feat]
                warn     = is_warn(val)
                card_cls = "feat-card warn" if warn else "feat-card ok"
                tag      = '<span class="warning-tag">⚠ suspicious</span>' if warn else '<span class="ok-tag">✔ ok</span>'
                cards_html += f"""
                <div class="{card_cls}">
                    <div class="feat-name">{lbl}</div>
                    <div class="feat-val">{fmt(val)}</div>
                    {tag}
                </div>"""
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        # Reasons
        st.markdown("#### 📋 Reasons")
        reasons = []
        r = raw
        if r.get('URLSimilarityIndex', 100) < 80:
            reasons.append(f"URL Similarity = {r['URLSimilarityIndex']:.1f} (legit = 100, phishing avg = 49)")
        if r.get('URLTitleMatchScore', 100) < 40:
            reasons.append(f"Title Match Score = {r['URLTitleMatchScore']:.1f} (legit avg = 75, phishing avg = 21)")
        if r.get('IsHTTPS', 1) == 0:
            reasons.append("No HTTPS — all legitimate sites use HTTPS")
        if r.get('IsDomainIP', 0) == 1:
            reasons.append("Domain is an IP address — major phishing red flag")
        if r.get('TLDLegitimateProb', 1) < 0.001:
            reasons.append(f"TLD has near-zero legitimacy probability ({r['TLDLegitimateProb']:.5f})")
        if r.get('NoOfQMarkInURL', 0) > 0:
            reasons.append(f"{int(r['NoOfQMarkInURL'])} '?' in URL — legitimate URLs typically have none")
        if r.get('DegitRatioInURL', 0) > 0.1:
            reasons.append(f"High digit ratio = {r['DegitRatioInURL']:.3f} (legit avg = 0.002)")
        if r.get('ObfuscationRatio', 0) > 0:
            reasons.append(f"Obfuscated characters detected (ratio = {r['ObfuscationRatio']:.4f})")

        reasons_html = '<div class="reason-box">'
        if is_legit and not reasons:
            reasons_html += '<div class="reason-ok">✔ URLSimilarityIndex = 100, HTTPS = Yes, TLD is legitimate → all signals match a legitimate site</div>'
        elif not is_legit and not reasons:
            reasons_html += '<div class="reason-item">⚠ Combination of URL features matches phishing patterns learned from 235,000+ URLs</div>'
        else:
            for rr in reasons:
                cls_r  = "reason-ok" if is_legit else "reason-item"
                icon_r = "✔" if is_legit else "⚠"
                reasons_html += f'<div class="{cls_r}">{icon_r} {rr}</div>'
        reasons_html += '</div>'
        st.markdown(reasons_html, unsafe_allow_html=True)