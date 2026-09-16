
# ============================================================
# IMPORTS
# ============================================================

import os
import json
import hashlib
import ipaddress
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import joblib

import streamlit as st
import pydeck as pdk

from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Automated Threat Prioritization",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SETTINGS
# ============================================================

TIMESTEPS = 8

CLOUDS = [
    "AWS",
    "Azure",
    "GCP"
]


SEVERITY_SCORE = {
    "Low": 25,
    "Medium": 50,
    "High": 75,
    "Critical": 100
}


SEVERITY_VALUE = {
    "Low": 0.25,
    "Medium": 0.50,
    "High": 0.75,
    "Critical": 1.00
}


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #07090d;
    }

    .main-title {
        font-size: 36px;
        font-weight: 900;
        color: #ff1744;
        letter-spacing: 1px;
    }

    .subtitle {
        color: #8d96a6;
        margin-bottom: 25px;
    }

    .login-title {
        font-size: 38px;
        font-weight: 900;
        color: #ff1744;
        text-align: center;
        margin-top: 70px;
    }

    .login-subtitle {
        color: #8d96a6;
        text-align: center;
        margin-bottom: 30px;
    }

    .focus-box {
        background-color: #12161e;
        border-left: 6px solid #ff1744;
        border-radius: 12px;
        padding: 22px;
        margin-top: 15px;
        margin-bottom: 25px;
    }

    .focus-title {
        color: #ff1744;
        font-size: 18px;
        font-weight: 900;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# USER SYSTEM
# ============================================================

USER_FILE = "users.json"


def hash_password(password):

    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


def load_users():

    if not os.path.exists(USER_FILE):

        return {}

    try:

        with open(
            USER_FILE,
            "r"
        ) as f:

            return json.load(f)

    except Exception:

        return {}


def save_users(users):

    with open(
        USER_FILE,
        "w"
    ) as f:

        json.dump(
            users,
            f,
            indent=2
        )


# ============================================================
# LOGIN STATE
# ============================================================

if "authenticated" not in st.session_state:

    st.session_state.authenticated = False


if "username" not in st.session_state:

    st.session_state.username = ""


if "events" not in st.session_state:

    st.session_state.events = []


if "sequence" not in st.session_state:

    st.session_state.sequence = []


if "cloud_index" not in st.session_state:

    st.session_state.cloud_index = 0


if "location_cache" not in st.session_state:

    st.session_state.location_cache = {}


if "manual_result" not in st.session_state:

    st.session_state.manual_result = None


# ============================================================
# LOGIN / SIGNUP
# ============================================================

if not st.session_state.authenticated:

    st.markdown(
        '<div class="login-title">'
        'AUTOMATED THREAT PRIORITIZATION'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="login-subtitle">'
        'AI-Powered Multi-Cloud Cybersecurity Platform'
        '</div>',
        unsafe_allow_html=True
    )


    left, center, right = st.columns(
        [1, 2, 1]
    )


    with center:

        login_tab, signup_tab = st.tabs(
            [
                "LOGIN",
                "CREATE ACCOUNT"
            ]
        )


        # ----------------------------------------------------
        # LOGIN
        # ----------------------------------------------------

        with login_tab:

            username = st.text_input(
                "Username",
                key="login_username"
            )


            password = st.text_input(
                "Password",
                type="password",
                key="login_password"
            )


            if st.button(
                "LOGIN",
                use_container_width=True
            ):

                users = load_users()


                if username not in users:

                    st.error(
                        "Account not found."
                    )


                elif users[username] != hash_password(
                    password
                ):

                    st.error(
                        "Incorrect username or password."
                    )


                else:

                    st.session_state.authenticated = True

                    st.session_state.username = (
                        username
                    )

                    st.rerun()


        # ----------------------------------------------------
        # SIGN UP
        # ----------------------------------------------------

        with signup_tab:

            new_username = st.text_input(
                "Choose username",
                key="signup_username"
            )


            new_password = st.text_input(
                "Password",
                type="password",
                key="signup_password"
            )


            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                key="signup_confirm"
            )


            if st.button(
                "CREATE ACCOUNT",
                use_container_width=True
            ):

                users = load_users()


                if not new_username:

                    st.error(
                        "Enter a username."
                    )


                elif len(new_password) < 6:

                    st.error(
                        "Password must contain at least 6 characters."
                    )


                elif new_password != confirm_password:

                    st.error(
                        "Passwords do not match."
                    )


                elif new_username in users:

                    st.error(
                        "Username already exists."
                    )


                else:

                    users[new_username] = (
                        hash_password(
                            new_password
                        )
                    )


                    save_users(users)


                    st.success(
                        "Account created successfully. "
                        "Please log in."
                    )


    # --------------------------------------------------------
    # STOP DASHBOARD
    # --------------------------------------------------------

    st.stop()


# ============================================================
# LOAD CACHED LOGS
# ============================================================

# ============================================================
# LOAD CACHED LOGS
# ============================================================

@st.cache_data(show_spinner="Loading security logs...")
def load_data():

    zip_filename = "combined_threats.zip"

    if not os.path.exists(zip_filename):
        raise FileNotFoundError(
            "combined_threats.zip was not found."
        )

    import zipfile

    with zipfile.ZipFile(zip_filename, "r") as z:

        csv_files = [
            name
            for name in z.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise FileNotFoundError(
                "No CSV file was found inside combined_threats.zip."
            )

        csv_name = csv_files[0]

        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    required = [
        "src",
        "payload_text",
        "type",
        "severity",
        "target"
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(missing)
        )

    for column in ["latitude", "longitude"]:
        if column not in df.columns:
            df[column] = np.nan

    if "city" not in df.columns:
        df["city"] = "Unknown"

    if "country_code" not in df.columns:
        df["country_code"] = "Unknown"

    # Reduce memory usage
    for column in [
        "src",
        "type",
        "severity",
        "target",
        "city",
        "country_code"
    ]:
        if column in df.columns:
            df[column] = df[column].astype("category")

    for column in ["latitude", "longitude"]:
        if column in df.columns:
            df[column] = df[column].astype("float32")

    return df

data = load_data()

data = load_data()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {

        "rf":
            joblib.load(
                "random_forest.pkl"
            ),

        "nn":
            joblib.load(
                "neural_network.pkl"
            ),

        "nn_scaler":
            joblib.load(
                "neural_network_scaler.pkl"
            ),

        "lstm":
            load_model(
                "lstm_model.keras"
            ),

        "lstm_encoder":
            joblib.load(
                "lstm_label_encoder.pkl"
            ),

        "lstm_scaler":
            joblib.load(
                "lstm_scaler.pkl"
            )

    }


    return models


models = load_models()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def ip_features(ip):

    try:

        parts = (
            str(ip)
            .strip()
            .split(".")
        )


        if len(parts) != 4:

            return [0, 0, 0, 0]


        values = [

            int(x)

            for x in parts

        ]


        if not all(

            0 <= x <= 255

            for x in values

        ):

            return [0, 0, 0, 0]


        return values


    except Exception:

        return [0, 0, 0, 0]


def payload_features(
    payload
):

    text = str(
        payload
    ).lower()


    return [

        int(
            "<script" in text
        ),

        int(
            "jndi" in text
        ),

        int(
            "union select" in text
        ),

        int(
            "login" in text
            or
            "wp-login" in text
        ),

        int(
            "mstshash" in text
        ),

        int(
            "wget" in text
            or
            "curl" in text
        ),

        int(
            "jsonrpc" in text
            or
            "eth_blocknumber" in text
        ),

        int(

            "select " in text

            and

            " from " in text

        ),

        int(
            "network port" in text
        )

    ]


def make_features(

    ip,

    payload,

    cloud

):

    values = (

        ip_features(ip)

        +

        payload_features(payload)

        +

        [

            int(
                cloud == "AWS"
            ),

            int(
                cloud == "Azure"
            ),

            int(
                cloud == "GCP"
            )

        ]

    )


    if len(values) != 16:

        raise ValueError(

            f"Model requires 16 features, "
            f"but {len(values)} were created."

        )


    return np.asarray(

        [values],

        dtype=np.float64

    )


# ============================================================
# RANDOM FOREST
# ============================================================

def calculate_rf(

    X,

    threat

):

    probabilities = (

        models["rf"]
        .predict_proba(
            X
        )[0]

    )


    classes = models["rf"].classes_


    match = np.where(
        classes == threat
    )[0]


    if len(match) == 0:

        return 0.0


    return float(

        probabilities[
            match[0]
        ]

    )


# ============================================================
# NEURAL NETWORK
# ============================================================

def calculate_nn(

    X,

    threat

):

    scaled = (

        models["nn_scaler"]
        .transform(X)

    )


    probabilities = (

        models["nn"]
        .predict_proba(
            scaled
        )[0]

    )


    classes = models["nn"].classes_


    match = np.where(
        classes == threat
    )[0]


    if len(match) == 0:

        return 0.0


    return float(

        probabilities[
            match[0]
        ]

    )


# ============================================================
# LSTM
# ============================================================

def make_lstm_row(

    threat,

    severity,

    cloud

):

    encoder = models[
        "lstm_encoder"
    ]


    if threat in encoder.classes_:

        threat_value = int(

            encoder.transform(
                [threat]
            )[0]

        )

    else:

        threat_value = 0


    cloud_value = {

        "AWS": 0,

        "Azure": 1,

        "GCP": 2

    }.get(

        cloud,

        0

    )


    severity_value = (

        SEVERITY_VALUE.get(

            severity,

            0.5

        )

    )


    return [

        threat_value,

        severity_value,

        cloud_value

    ]


def calculate_lstm(

    threat,

    severity,

    cloud

):

    current = make_lstm_row(

        threat,

        severity,

        cloud

    )


    history = (

        st.session_state.sequence

        +

        [current]

    )


    history = history[
        -TIMESTEPS:
    ]


    while len(history) < TIMESTEPS:

        history.insert(
            0,
            current
        )


    array = np.asarray(

        history,

        dtype=np.float32

    )


    scaled = (

        models[
            "lstm_scaler"
        ]

        .transform(
            array
        )

    )


    X = scaled.reshape(

        1,

        TIMESTEPS,

        3

    )


    probabilities = (

        models["lstm"]
        .predict(
            X,
            verbose=0
        )[0]

    )


    encoder = models[
        "lstm_encoder"
    ]


    if threat not in encoder.classes_:

        return 0.0


    index = int(

        encoder.transform(
            [threat]
        )[0]

    )


    return float(

        probabilities[
            index
        ]

    )


# ============================================================
# LOCATION LOOKUP
# ============================================================

def is_public_ip(
    ip
):

    try:

        address = ipaddress.ip_address(

            str(ip).strip()

        )


        return (

            not address.is_private

            and

            not address.is_loopback

            and

            not address.is_reserved

            and

            not address.is_multicast

        )


    except Exception:

        return False


def lookup_ip_location(
    ip
):

    ip = str(
        ip
    ).strip()


    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    if ip in st.session_state.location_cache:

        return (
            st.session_state
            .location_cache[ip]
        )


    if not is_public_ip(ip):

        result = {

            "success": False,

            "latitude": None,

            "longitude": None,

            "city": "Private Network",

            "country": "Local"

        }


        st.session_state.location_cache[
            ip
        ] = result


        return result


    # --------------------------------------------------------
    # FIRST PROVIDER
    # --------------------------------------------------------

    try:

        response = requests.get(

            f"https://ipapi.co/{ip}/json/",

            timeout=6

        )


        if response.status_code == 200:

            info = response.json()


            lat = info.get(
                "latitude"
            )

            lon = info.get(
                "longitude"
            )


            if (

                lat is not None

                and

                lon is not None

            ):

                result = {

                    "success": True,

                    "latitude":
                        float(lat),

                    "longitude":
                        float(lon),

                    "city":
                        info.get(
                            "city",
                            "Unknown"
                        ),

                    "country":
                        info.get(
                            "country_name",
                            "Unknown"
                        )

                }


                st.session_state.location_cache[
                    ip
                ] = result


                return result


    except Exception:

        pass


    # --------------------------------------------------------
    # SECOND PROVIDER
    # --------------------------------------------------------

    try:

        response = requests.get(

            f"https://ipwho.is/{ip}",

            timeout=6

        )


        if response.status_code == 200:

            info = response.json()


            if (

                info.get(
                    "success"
                )

                and

                info.get(
                    "latitude"
                ) is not None

                and

                info.get(
                    "longitude"
                ) is not None

            ):

                result = {

                    "success": True,

                    "latitude":
                        float(
                            info["latitude"]
                        ),

                    "longitude":
                        float(
                            info["longitude"]
                        ),

                    "city":
                        info.get(
                            "city",
                            "Unknown"
                        ),

                    "country":
                        info.get(
                            "country",
                            "Unknown"
                        )

                }


                st.session_state.location_cache[
                    ip
                ] = result


                return result


    except Exception:

        pass


    result = {

        "success":
            False,

        "latitude":
            None,

        "longitude":
            None,

        "city":
            "Location unavailable",

        "country":
            "Unknown"

    }


    st.session_state.location_cache[
        ip
    ] = result


    return result


# ============================================================
# FAST LIVE EVENT
#
# NO NETWORK CALLS
# ============================================================

def process_live_event(
    row
):

    ip = str(
        row["src"]
    ).strip()


    threat = str(
        row["type"]
    )


    severity = str(
        row["severity"]
    )


    cloud = str(
        row["target"]
    )


    payload = str(

        row.get(
            "payload_text",
            ""
        )

    )


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    X = make_features(

        ip,

        payload,

        cloud

    )


    # --------------------------------------------------------
    # RF
    # --------------------------------------------------------

    rf = calculate_rf(

        X,

        threat

    )


    # --------------------------------------------------------
    # NN
    # --------------------------------------------------------

    nn = calculate_nn(

        X,

        threat

    )


    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    lstm = calculate_lstm(

        threat,

        severity,

        cloud

    )


    # --------------------------------------------------------
    # AI SCORE
    # --------------------------------------------------------

    ai_score = (

        rf * 0.35

        +

        nn * 0.30

        +

        lstm * 0.35

    ) * 100


    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    severity_score = (

        SEVERITY_SCORE.get(

            severity,

            50

        )

    )


    # --------------------------------------------------------
    # REPEATED ATTACK
    # --------------------------------------------------------

    repeated = sum(

        e["source"] == ip

        for e
        in st.session_state.events

    )


    behaviour_score = min(

        repeated * 10,

        100

    )


    # --------------------------------------------------------
    # FINAL PRIORITY
    # --------------------------------------------------------

    final_score = (

        ai_score * 0.60

        +

        severity_score * 0.30

        +

        behaviour_score * 0.10

    )


    final_score = max(

        0,

        min(
            100,
            final_score
        )

    )


    if final_score >= 80:

        priority = "CRITICAL"

    elif final_score >= 60:

        priority = "HIGH"

    elif final_score >= 35:

        priority = "MEDIUM"

    else:

        priority = "LOW"


    # --------------------------------------------------------
    # DATASET COORDINATES ONLY
    #
    # No external lookup during live streaming.
    # --------------------------------------------------------

    latitude = row.get(
        "latitude",
        np.nan
    )


    longitude = row.get(
        "longitude",
        np.nan
    )


    if (

        pd.notna(latitude)

        and

        pd.notna(longitude)

    ):

        latitude = float(
            latitude
        )

        longitude = float(
            longitude
        )


        city = str(

            row.get(
                "city",
                "Dataset location"
            )

        )


        country = str(

            row.get(
                "country_code",
                "Unknown"
            )

        )


        location_source = (
            "Dataset"
        )

    else:

        latitude = np.nan

        longitude = np.nan


        city = "IP lookup required"

        country = str(

            row.get(
                "country_code",
                "Unknown"
            )

        )


        location_source = (
            "Not geolocated during live stream"
        )


    # --------------------------------------------------------
    # LSTM HISTORY
    # --------------------------------------------------------

    st.session_state.sequence.append(

        make_lstm_row(

            threat,

            severity,

            cloud

        )

    )


    st.session_state.sequence = (

        st.session_state.sequence[
            -TIMESTEPS:
        ]

    )


    # --------------------------------------------------------
    # EVENT
    # --------------------------------------------------------

    return {

        "time":
            datetime.now().strftime(
                "%H:%M:%S"
            ),

        "source":
            ip,

        "threat":
            threat,

        "severity":
            severity,

        "cloud":
            cloud,

        "rf":
            round(
                rf * 100,
                2
            ),

        "nn":
            round(
                nn * 100,
                2
            ),

        "lstm":
            round(
                lstm * 100,
                2
            ),

        "ai_score":
            round(
                ai_score,
                2
            ),

        "final_score":
            round(
                final_score,
                2
            ),

        "priority":
            priority,

        "city":
            city,

        "country":
            country,

        "latitude":
            latitude,

        "longitude":
            longitude,

        "location_source":
            location_source

    }


# ============================================================
# HEADER
# ============================================================

st.markdown(

    '<div class="main-title">'
    'AUTOMATED THREAT PRIORITIZATION'
    '</div>',

    unsafe_allow_html=True

)


st.markdown(

    '<div class="subtitle">'
    'AI-Powered Multi-Cloud Cybersecurity Dashboard'
    '</div>',

    unsafe_allow_html=True

)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.write(

        f"Logged in as "
        f"**{st.session_state.username}**"

    )


    st.divider()


    live_feed = st.toggle(

        "LIVE THREAT FEED",

        value=True

    )


    st.divider()


    st.write(
        "### DATASET"
    )


    st.write(

        f"Total: "
        f"{len(data):,}"

    )


    for cloud in CLOUDS:

        count = int(

            (
                data["target"]
                ==
                cloud
            ).sum()

        )


        st.write(

            f"{cloud}: "
            f"**{count:,}**"

        )


    st.divider()


    if st.button(

        "CLEAR LIVE FEED",

        use_container_width=True

    ):

        st.session_state.events = []

        st.session_state.sequence = []

        st.session_state.cloud_index = 0

        st.rerun()


    if st.button(

        "LOG OUT",

        use_container_width=True

    ):

        st.session_state.authenticated = False

        st.session_state.username = ""

        st.session_state.events = []

        st.session_state.sequence = []

        st.rerun()


# ============================================================
# LIVE ENGINE
#
# AWS → AZURE → GCP → AWS...
# One new event every second.
# ============================================================

@st.fragment(
    run_every="1s"
)
def live_engine():

    # --------------------------------------------------------
    # Determine available clouds
    # --------------------------------------------------------

    available_clouds = [

        cloud

        for cloud in CLOUDS

        if (

            "target" in data.columns

            and

            (
                data["target"]
                ==
                cloud
            ).any()

        )

    ]


    # --------------------------------------------------------
    # Generate event
    # --------------------------------------------------------

    if (

        live_feed

        and

        available_clouds

    ):

        current_index = (

            st.session_state.cloud_index

            %

            len(
                available_clouds
            )

        )


        cloud = available_clouds[
            current_index
        ]


        st.session_state.cloud_index += 1


        cloud_data = data[

            data["target"]
            ==
            cloud

        ]


        # Select one record
        row = (

            cloud_data

            .sample(
                1
            )

            .iloc[0]

        )


        event = process_live_event(
            row
        )


        st.session_state.events.insert(

            0,

            event

        )


        # Keep only latest 60

        st.session_state.events = (

            st.session_state.events[
                :60
            ]

        )


    # --------------------------------------------------------
    # NO DATA
    # --------------------------------------------------------

    events = (
        st.session_state.events
    )


    if not events:

        st.info(
            "Waiting for live attacks..."
        )

        return


    df = pd.DataFrame(
        events
    )


    # ========================================================
    # COUNTERS
    # ========================================================

    critical = sum(

        x["priority"]
        ==
        "CRITICAL"

        for x in events

    )


    high = sum(

        x["priority"]
        ==
        "HIGH"

        for x in events

    )


    medium = sum(

        x["priority"]
        ==
        "MEDIUM"

        for x in events

    )


    low = sum(

        x["priority"]
        ==
        "LOW"

        for x in events

    )


    c1, c2, c3, c4, c5 = st.columns(5)


    c1.metric(
        "LIVE THREATS",
        len(events)
    )


    c2.metric(
        "CRITICAL",
        critical
    )


    c3.metric(
        "HIGH",
        high
    )


    c4.metric(
        "MEDIUM",
        medium
    )


    c5.metric(
        "LOW",
        low
    )


    st.divider()


    # ========================================================
    # FOCUS THREAT
    # ========================================================

    focus = max(

        events,

        key=lambda x:
            x["final_score"]

    )


    st.markdown(

        '<div class="focus-box">'

        '<div class="focus-title">'
        '🚨 FOCUS ON THIS THREAT FIRST'
        '</div>'

        f'<h2>{focus["threat"]}</h2>'

        f'<b>Priority:</b> '
        f'{focus["priority"]}<br>'

        f'<b>Final Score:</b> '
        f'{focus["final_score"]}%<br>'

        f'<b>AI Score:</b> '
        f'{focus["ai_score"]}%<br>'

        f'<b>Source IP:</b> '
        f'{focus["source"]}<br>'

        f'<b>Location:</b> '
        f'{focus["city"]}, '
        f'{focus["country"]}<br>'

        f'<b>Cloud:</b> '
        f'{focus["cloud"]}<br>'

        f'<b>Severity:</b> '
        f'{focus["severity"]}'

        '</div>',

        unsafe_allow_html=True

    )


    # ========================================================
    # CLOUD STATUS
    # ========================================================

    st.subheader(
        "☁️ Multi-Cloud Threat Status"
    )


    cloud_columns = st.columns(3)


    for column, cloud in zip(

        cloud_columns,

        CLOUDS

    ):

        with column:

            st.markdown(
                f"## {cloud}"
            )


            cloud_events = [

                event

                for event
                in events

                if event["cloud"] == cloud

            ]


            st.metric(

                "Live Threats",

                len(
                    cloud_events
                )

            )


            dataset_count = int(

                (
                    data["target"]
                    ==
                    cloud
                ).sum()

            )


            st.caption(

                f"Dataset records: "
                f"{dataset_count:,}"

            )


            if cloud_events:

                highest = max(

                    cloud_events,

                    key=lambda x:
                        x["final_score"]

                )


                st.metric(

                    "Highest Risk",

                    f'{highest["final_score"]}%'

                )


                st.write(

                    f'**Threat:** '
                    f'{highest["threat"]}'

                )


                st.write(

                    f'**Priority:** '
                    f'{highest["priority"]}'

                )


                st.write(

                    f'**Severity:** '
                    f'{highest["severity"]}'

                )

            else:

                st.info(
                    "Waiting for event."
                )


    st.divider()


    # ========================================================
    # AI SCORES
    # ========================================================

    latest = events[0]


    st.subheader(
        "🤖 AI Threat Assessment"
    )


    a, b, c, d = st.columns(4)


    a.metric(

        "AI SCORE",

        f'{latest["ai_score"]}%'

    )


    b.metric(

        "Random Forest",

        f'{latest["rf"]}%'

    )


    c.metric(

        "Neural Network",

        f'{latest["nn"]}%'

    )


    d.metric(

        "LSTM",

        f'{latest["lstm"]}%'

    )


    # ========================================================
    # MAP
    # ========================================================

    st.subheader(
        "🌍 Live Attack Map"
    )


    map_df = df.dropna(

        subset=[
            "latitude",
            "longitude"
        ]

    ).copy()


    if len(map_df) > 0:

        map_df["radius"] = (

            map_df[
                "final_score"
            ]
            .clip(
                lower=10
            )
            * 1500

        )


        layer = pdk.Layer(

            "ScatterplotLayer",

            data=map_df,

            get_position=[

                "longitude",

                "latitude"

            ],

            get_radius="radius",

            get_fill_color=[

                255,

                40,

                60,

                190

            ],

            pickable=True,

            auto_highlight=True

        )


        view = pdk.ViewState(

            latitude=float(

                map_df[
                    "latitude"
                ].mean()

            ),

            longitude=float(

                map_df[
                    "longitude"
                ].mean()

            ),

            zoom=1.1

        )


        deck = pdk.Deck(

            layers=[
                layer
            ],

            initial_view_state=view,

            tooltip={

                "html":
                """
                <b>{threat}</b><br/>
                IP: {source}<br/>
                Cloud: {cloud}<br/>
                Severity: {severity}<br/>
                Priority: {priority}<br/>
                Score: {final_score}%<br/>
                Location: {city}, {country}
                """

            }

        )


        st.pydeck_chart(

            deck,

            use_container_width=True

        )


        st.caption(

            "Dataset coordinates are used when available. "
            "IP-derived coordinates are approximate."

        )


    else:

        st.warning(

            "The current live events do not have coordinates yet."

        )


    # ========================================================
    # THREAT QUEUE
    # ========================================================

    st.subheader(
        "🎯 Threat Priority Queue"
    )


    queue = df[

        [

            "time",

            "source",

            "threat",

            "severity",

            "cloud",

            "city",

            "country",

            "ai_score",

            "final_score",

            "priority"

        ]

    ].sort_values(

        "final_score",

        ascending=False

    )


    st.dataframe(

        queue,

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# RUN LIVE ENGINE
# ============================================================

live_engine()


# ============================================================
# MANUAL IP LOCATION
#
# This is intentionally outside the live stream.
# ============================================================

st.divider()


st.subheader(
    "🔎 Investigate Public IP Location"
)


manual_ip = st.text_input(

    "Enter public IP address",

    placeholder=
        "Example: 8.8.8.8"

)


if st.button(
    "LOOK UP LOCATION"
):

    manual_ip = (
        manual_ip
        .strip()
    )


    if not is_public_ip(
        manual_ip
    ):

        st.error(
            "Enter a valid public IP address."
        )

    else:

        with st.spinner(
            "Looking up IP location..."
        ):

            result = lookup_ip_location(
                manual_ip
            )


        st.session_state.manual_result = result


result = st.session_state.manual_result


if result:

    if result.get(
        "success"
    ):

        st.subheader(
            "IP Location"
        )


        a, b, c = st.columns(3)


        a.metric(

            "City",

            result.get(
                "city",
                "Unknown"
            )

        )


        b.metric(

            "Country",

            result.get(
                "country",
                "Unknown"
            )

        )


        c.metric(

            "Latitude",

            result.get(
                "latitude"
            )

        )


        st.write(

            "Longitude:",

            result.get(
                "longitude"
            )

        )


        st.info(

            "IP geolocation is an approximate network "
            "location, not an exact physical/GPS location."

        )


    else:

        st.warning(
            "Location could not be determined."
        )
