"""Streamlit frontend: seed-song picker, 5 aspect sliders, results with
inline audio players (all served CC audio, so streaming in the demo is
legal). Talks to the FastAPI backend over HTTP -- run both:

    uvicorn api.main:app --reload
    streamlit run web/app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE = os.getenv("MS_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Controllable Music Search", layout="wide")
st.title("Controllable Multi-Axis Music Search")
st.caption(
    "Pick a seed song, tune what \"similar\" means, and retrieve matches on your terms."
)

with st.sidebar:
    st.header("Seed")
    seed_track_id = st.text_input("Seed track ID", value="")
    user_id = st.text_input("User ID (optional, for personalized weights)", value="")

    st.header("Similarity axes")
    rhythm = st.slider("Drum groove / rhythm", 0.0, 1.0, 0.2)
    melody = st.slider("Melody / harmony", 0.0, 1.0, 0.2)
    timbre = st.slider("Instrumentation / timbre", 0.0, 1.0, 0.2)
    vocal = st.slider("Vocal character", 0.0, 1.0, 0.2)
    lyric = st.slider("Lyrical meaning", 0.0, 1.0, 0.2)

    top_k = st.slider("Results", 5, 50, 20)
    search_clicked = st.button("Search", type="primary")

st.subheader("Upload your own song")
uploaded = st.file_uploader("MP3/WAV", type=["mp3", "wav"])
if uploaded is not None and st.button("Process upload"):
    resp = requests.post(
        f"{API_BASE}/upload", files={"file": (uploaded.name, uploaded.getvalue())}, timeout=30
    )
    resp.raise_for_status()
    job = resp.json()
    st.session_state["upload_job_id"] = job["job_id"]
    st.info(f"Queued as job {job['job_id']}. Poll below once it's indexed.")

if "upload_job_id" in st.session_state and st.button("Check upload status"):
    resp = requests.get(f"{API_BASE}/upload/{st.session_state['upload_job_id']}", timeout=10)
    resp.raise_for_status()
    status = resp.json()
    st.json(status)
    if status["status"] == "done":
        seed_track_id = status["track_id"]

if search_clicked and seed_track_id:
    payload = {
        "seed_track_id": seed_track_id,
        "weights": {
            "rhythm": rhythm,
            "melody": melody,
            "timbre": timbre,
            "vocal": vocal,
            "lyric": lyric,
        },
        "top_k": top_k,
        "user_id": user_id or None,
    }
    try:
        resp = requests.post(f"{API_BASE}/search", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        st.error(f"Search failed: {exc}")
    else:
        st.caption(
            f"Weights used: {data['weights_used']}  ·  latency: {data['latency_ms']:.0f} ms"
        )
        for item in data["results"]:
            cols = st.columns([1, 3, 2])
            with cols[0]:
                st.write(f"**{item['track_id']}**")
                st.write(f"score: {item['score']:.3f}")
            with cols[1]:
                # CC-licensed MTG-Jamendo audio streams straight off the CDN;
                # `path` comes through the Qdrant payload (upsert passthrough).
                if item.get("path"):
                    st.audio(f"https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/{item['path']}")
                else:
                    st.caption("no audio path indexed")
            with cols[2]:
                st.bar_chart(item["per_aspect_score"])

# Lightweight pairwise feedback capture for personalization (Phase 4)
st.divider()
st.subheader("Which is more similar? (helps personalize your weights)")
col_a, col_b = st.columns(2)
candidate_a = col_a.text_input("Candidate A track ID")
candidate_b = col_b.text_input("Candidate B track ID")
choice = st.radio("More similar to the seed:", ["A", "B"], horizontal=True)
if st.button("Submit feedback") and seed_track_id and candidate_a and candidate_b and user_id:
    requests.post(
        f"{API_BASE}/feedback",
        json={
            "user_id": user_id,
            "seed_track_id": seed_track_id,
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "chose_a": choice == "A",
        },
        timeout=10,
    )
    st.success("Feedback recorded.")
