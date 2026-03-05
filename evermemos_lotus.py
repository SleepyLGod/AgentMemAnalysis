"""
EverMemOS Semantic Queries via LOTUS
=====================================
Executes Q1-Q7 from sem_queries.md using LOTUS operators on LOCOMO data.
Uses DeepSeek API for LLM calls and local SentenceTransformers for embeddings.

Usage:
    source .venv/bin/activate
    python evermemos_lotus.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load DeepSeek API key
load_dotenv()
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")

import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store import FaissVS

# ─────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────
LOCOMO_PATH = Path("evermemos/evaluation/data/locomo/locomo10.json")
CONV_INDEX = 0                  # which conversation to process
BOUNDARY_MSG_THRESHOLD = 15     # force boundary if >= N messages accumulated
TOPIC_SIM_THRESHOLD = 0.5       # cosine similarity threshold for topic matching
PROFILE_MIN_SEGMENTS = 2        # min segments in a topic before profile distillation

# ─────────────────────────────────────────────────────
# Setup LOTUS
# ─────────────────────────────────────────────────────
print("=" * 60)
print("Setting up LOTUS with DeepSeek + SentenceTransformers")
print("=" * 60)

lm = LM(
    model="deepseek/deepseek-chat",
    max_tokens=1024,
    temperature=0.0,
    max_batch_size=4,    # conservative to avoid rate limits
)
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
vs = FaissVS()
lotus.settings.configure(lm=lm, rm=rm, vs=vs)

# ─────────────────────────────────────────────────────
# Load LOCOMO data
# ─────────────────────────────────────────────────────
print("\nLoading LOCOMO data...")
with open(LOCOMO_PATH) as f:
    locomo_data = json.load(f)

conv = locomo_data[CONV_INDEX]
conversation = conv["conversation"]
speaker_a = conversation["speaker_a"]
speaker_b = conversation["speaker_b"]

# Collect all messages across all sessions in order
all_messages = []
session_idx = 1
while f"session_{session_idx}" in conversation:
    session_time = conversation.get(f"session_{session_idx}_date_time", f"Session {session_idx}")
    for msg in conversation[f"session_{session_idx}"]:
        all_messages.append({
            "speaker": msg["speaker"],
            "dia_id": msg["dia_id"],
            "text": msg["text"],
            "session_id": f"session_{session_idx}",
            "session_time": session_time,
        })
    session_idx += 1

print(f"Loaded conversation '{conv['sample_id']}': {speaker_a} & {speaker_b}")
print(f"  Total messages: {len(all_messages)} across {session_idx - 1} sessions")


def format_messages(messages: list[dict]) -> str:
    """Format a list of messages into a single conversation string."""
    lines = []
    for m in messages:
        lines.append(f"[{m['session_time']}] {m['speaker']}: {m['text']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────
# State: accumulates across the entire pipeline
# ─────────────────────────────────────────────────────
history_segments = []        # list of {segment_id, content, episode, subject, foresights, facts, topic_id}
topics = {}                  # topic_id -> {centroid: np.array, segments: [segment_id, ...]}
profiles = {}                # topic_id -> {traits: str}
segment_counter = 0
topic_counter = 0


# ═══════════════════════════════════════════════════════
# MAIN LOOP: Process messages one-by-one
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Starting EverMemOS pipeline")
print("=" * 60)

tumbling_window = []  # accumulates messages until boundary detected

for msg_idx, msg in enumerate(all_messages):
    tumbling_window.append(msg)
    print(f"\n--- Message {msg_idx + 1}/{len(all_messages)}: "
          f"{msg['speaker']}: {msg['text'][:60]}...")

    # ─────────────────────────────────────────────────
    # Q1: Boundary Detection
    # ─────────────────────────────────────────────────
    # Force boundary if message threshold reached
    force_boundary = len(tumbling_window) >= BOUNDARY_MSG_THRESHOLD

    if not force_boundary and len(tumbling_window) >= 3:
        # Use sem_filter to detect semantic boundary
        window_text = format_messages(tumbling_window)
        boundary_df = pd.DataFrame({"content": [window_text]})
        try:
            result = boundary_df.sem_filter(
                "The conversation in {content} has reached a natural semantic boundary "
                "due to a clear topic shift, a significant time gap between messages, "
                "or a logical conclusion of the discussion"
            )
            is_boundary = len(result) > 0  # sem_filter keeps rows that match
            print(f"  Q1 boundary check: {'BOUNDARY DETECTED' if is_boundary else 'continuing...'}")
        except Exception as e:
            print(f"  Q1 boundary check error: {e}")
            is_boundary = False
    elif force_boundary:
        is_boundary = True
        print(f"  Q1 FORCE boundary (>= {BOUNDARY_MSG_THRESHOLD} messages)")
    else:
        is_boundary = False
        print(f"  Q1 skipped (only {len(tumbling_window)} messages, need >= 3)")

    if not is_boundary:
        continue

    # ═════════════════════════════════════════════════
    # BOUNDARY REACHED — Seal window and process
    # ═════════════════════════════════════════════════
    segment_id = f"seg_{segment_counter}"
    segment_counter += 1
    window_content = format_messages(tumbling_window)
    n_msgs = len(tumbling_window)

    print(f"\n{'─' * 50}")
    print(f"  SEALED SEGMENT {segment_id} ({n_msgs} messages)")
    print(f"{'─' * 50}")

    # ─────────────────────────────────────────────────
    # Q2 + Q3: Episode Synthesis + Subject Extraction
    # ─────────────────────────────────────────────────
    print("\n  [Q2+Q3] Extracting episode + subject...")
    segment_df = pd.DataFrame({"content": [window_content]})
    try:
        episode_result = segment_df.sem_extract(
            input_cols=["content"],
            output_cols={
                "episode": "A concise third-person episodic narrative capturing the key events, "
                           "context, and emotional dynamics of this conversation segment",
                "subject": "The central subject or topic of this conversation in a few words",
            },
        )
        episode_text = episode_result["episode"].iloc[0] if "episode" in episode_result.columns else "N/A"
        subject_text = episode_result["subject"].iloc[0] if "subject" in episode_result.columns else "N/A"
    except Exception as e:
        print(f"  ERROR in Q2+Q3: {e}")
        episode_text = "Extraction failed"
        subject_text = "Extraction failed"

    print(f"  Episode: {episode_text[:120]}...")
    print(f"  Subject: {subject_text}")

    # ─────────────────────────────────────────────────
    # Q4: Foresight Extraction
    # ─────────────────────────────────────────────────
    print("\n  [Q4] Extracting foresights...")
    try:
        foresight_result = segment_df.sem_extract(
            input_cols=["content"],
            output_cols={
                "prediction": "A time-bounded future prediction or planned action mentioned in the conversation",
                "timeframe": "When this prediction or action is expected to happen",
            },
        )
        foresights = []
        if "prediction" in foresight_result.columns:
            for _, row in foresight_result.iterrows():
                pred = row.get("prediction", "N/A")
                tf = row.get("timeframe", "N/A")
                if pred and pred != "N/A" and str(pred).strip():
                    foresights.append({"prediction": pred, "timeframe": tf})
    except Exception as e:
        print(f"  ERROR in Q4: {e}")
        foresights = []

    if foresights:
        for f in foresights:
            print(f"  → {f['prediction']} (timeframe: {f['timeframe']})")
    else:
        print(f"  (no foresights detected)")

    # ─────────────────────────────────────────────────
    # Q5: Fact / EventLog Extraction
    # ─────────────────────────────────────────────────
    print("\n  [Q5] Extracting facts...")
    try:
        fact_result = segment_df.sem_extract(
            input_cols=["content"],
            output_cols={
                "atomic_fact": "A discrete atomic factual event from the conversation "
                               "(who did what, when, with specific details)",
                "timestamp": "When this event happened or was mentioned",
            },
        )
        facts = []
        if "atomic_fact" in fact_result.columns:
            for _, row in fact_result.iterrows():
                af = row.get("atomic_fact", "N/A")
                ts = row.get("timestamp", "N/A")
                if af and af != "N/A" and str(af).strip():
                    facts.append({"atomic_fact": af, "timestamp": ts})
    except Exception as e:
        print(f"  ERROR in Q5: {e}")
        facts = []

    if facts:
        for f in facts[:5]:  # show first 5
            print(f"  → {f['atomic_fact']} (at: {f['timestamp']})")
        if len(facts) > 5:
            print(f"  ... and {len(facts) - 5} more facts")
    else:
        print(f"  (no facts extracted)")

    # ─────────────────────────────────────────────────
    # Q6: Topic / Thematic Clustering
    # ─────────────────────────────────────────────────
    print("\n  [Q6] Topic clustering...")

    # Get embedding for this episode
    episode_embedding = rm([episode_text])[0]

    assigned_topic = None
    best_sim = -1.0

    for tid, topic_info in topics.items():
        centroid = topic_info["centroid"]
        # Cosine similarity
        sim = float(np.dot(episode_embedding, centroid) /
                     (np.linalg.norm(episode_embedding) * np.linalg.norm(centroid) + 1e-9))
        if sim > best_sim:
            best_sim = sim
            if sim >= TOPIC_SIM_THRESHOLD:
                assigned_topic = tid

    if assigned_topic is None:
        # Create new topic
        assigned_topic = f"topic_{topic_counter}"
        topic_counter += 1
        topics[assigned_topic] = {
            "centroid": episode_embedding,
            "segments": [],
            "segment_count": 0,
        }
        print(f"  NEW topic: {assigned_topic} (no match >= {TOPIC_SIM_THRESHOLD})")
    else:
        # Update centroid (running average)
        old_centroid = topics[assigned_topic]["centroid"]
        n = topics[assigned_topic]["segment_count"]
        topics[assigned_topic]["centroid"] = (old_centroid * n + episode_embedding) / (n + 1)
        print(f"  MERGED into {assigned_topic} (sim={best_sim:.3f})")

    topics[assigned_topic]["segments"].append(segment_id)
    topics[assigned_topic]["segment_count"] += 1

    # Store segment
    segment_record = {
        "segment_id": segment_id,
        "content": window_content,
        "episode": episode_text,
        "subject": subject_text,
        "foresights": foresights,
        "facts": facts,
        "topic_id": assigned_topic,
    }
    history_segments.append(segment_record)

    # ─────────────────────────────────────────────────
    # Q7: Profile Distillation (conditional)
    # ─────────────────────────────────────────────────
    topic_seg_count = topics[assigned_topic]["segment_count"]
    if topic_seg_count >= PROFILE_MIN_SEGMENTS:
        print(f"\n  [Q7] Profile distillation for {assigned_topic} "
              f"({topic_seg_count} segments >= {PROFILE_MIN_SEGMENTS})...")

        # Collect all episodes in this topic
        topic_episodes = [
            seg for seg in history_segments
            if seg["topic_id"] == assigned_topic
        ]
        episodes_df = pd.DataFrame({
            "episode": [seg["episode"] for seg in topic_episodes],
            "topic_id": [assigned_topic] * len(topic_episodes),
        })

        # Add existing profile as context if available
        existing_traits = profiles.get(assigned_topic, {}).get("traits", "No existing profile yet.")
        episodes_df["existing_traits"] = existing_traits

        try:
            profile_result = episodes_df.sem_agg(
                "Given the existing user profile: {existing_traits}, and these conversation episodes: "
                "{episode}, distill updated stable user traits including personality, interests, "
                "skills, values, and behavioral patterns. Return a JSON-like structured profile.",
                group_by=["topic_id"],
            )
            new_traits = profile_result["_output"].iloc[0] if "_output" in profile_result.columns else "N/A"
            profiles[assigned_topic] = {"traits": new_traits}
            print(f"  Profile: {str(new_traits)[:200]}...")
        except Exception as e:
            print(f"  ERROR in Q7: {e}")
    else:
        print(f"\n  [Q7] Skipped: {assigned_topic} has {topic_seg_count} segments "
              f"(need >= {PROFILE_MIN_SEGMENTS})")

    # Clear tumbling window
    tumbling_window = []
    print()

# ═══════════════════════════════════════════════════════
# Handle remaining messages in tumbling window
# ═══════════════════════════════════════════════════════
if tumbling_window:
    print(f"\n{'=' * 60}")
    print(f"Remaining {len(tumbling_window)} messages in buffer (forcing final segment)")
    print(f"{'=' * 60}")
    # Force process remaining — same as above but simplified
    segment_id = f"seg_{segment_counter}"
    segment_counter += 1
    window_content = format_messages(tumbling_window)

    segment_df = pd.DataFrame({"content": [window_content]})
    try:
        episode_result = segment_df.sem_extract(
            input_cols=["content"],
            output_cols={
                "episode": "A concise third-person episodic narrative",
                "subject": "The central topic",
            },
        )
        episode_text = episode_result["episode"].iloc[0] if "episode" in episode_result.columns else "N/A"
        subject_text = episode_result["subject"].iloc[0] if "subject" in episode_result.columns else "N/A"
    except Exception as e:
        episode_text = f"Error: {e}"
        subject_text = "Error"

    segment_record = {
        "segment_id": segment_id,
        "content": window_content,
        "episode": episode_text,
        "subject": subject_text,
        "foresights": [],
        "facts": [],
        "topic_id": "topic_final",
    }
    history_segments.append(segment_record)
    print(f"  Final episode: {episode_text[:120]}...")


# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"\nTotal segments created: {len(history_segments)}")
print(f"Total topics created: {len(topics)}")
print(f"Total profiles distilled: {len(profiles)}")

print("\n--- Segments ---")
for seg in history_segments:
    print(f"  [{seg['segment_id']}] topic={seg['topic_id']} | "
          f"subject={seg['subject'][:50]} | "
          f"foresights={len(seg['foresights'])} | facts={len(seg['facts'])}")

print("\n--- Topics ---")
for tid, info in topics.items():
    print(f"  [{tid}] segments={info['segments']}")

print("\n--- Profiles ---")
for tid, pinfo in profiles.items():
    print(f"  [{tid}] {str(pinfo['traits'])[:150]}...")

print("\n--- LLM Usage ---")
lm.print_total_usage()
