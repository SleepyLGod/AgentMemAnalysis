"""
EverMemOS Semantic Queries via LOTUS
=====================================
Executes Q1-Q7 from sem_queries.md using LOTUS operators on LOCOMO data.
Uses DeepSeek API for LLM calls and local SentenceTransformers for embeddings.

Features added:
- JSON logging of every operator's input/output
- Limited processing size (e.g. 40 messages) to avoid rate limits
- Token usage and cache hit reporting

Usage:
    source .venv/bin/activate
    python evermemos_lotus.py

LOTUS optimizations:
gold plan (optimal LLM) + approximate plan with γ (accuracy target), δ (fault tolerance) target + proxy scorer (cheaper scorer) + sampling + CI (confidence interval)
- filter: proxy-oracle cascade: 
    learn a cascade between a small proxy model and the full LLM.
    on a sample, jointly run proxy + LLM to find score thresholds with guaranteed precision/recall; at runtime, only “uncertain” tuples go to the LLM
- join: 
    treat join as a semantic filter over tuple pairs and use two alternative proxies: pure embedding similarity, or “predict‑then‑compare” where the LLM first predicts the key then uses similarity.​
- sim-filter: produce similarity score solely based on embedding similarity, only efficient when tuple pairs with high semantic similarity between the right and left join key are more likely to pass the predicate.
- project-sim-filter: sem_map on left table, using LLM to predict the correlated value in the right table => compute embedding similarity between predicted value and the right join key
    On sampled pairs, estimate which proxy + threshold meets the precision/recall targets with the fewest LLM calls, then use that cascade for the full join
- group by:
    Keep the expensive LLM‑based clustering to find group centers, but optimize the assignment step.​
    Use embedding similarity to centers as a proxy and only call the LLM for tuples whose similarity is below a learned threshold, preserving overall labeling accuracy.​
- top-k: 
    Use an LLM‑based quick‑select as the gold algorithm to reduce pairwise comparisons vs full sorting.​
    Further cut comparisons by choosing pivots using embedding similarity to the query, so pivots are likely near the top‑k region when similarity correlates with the ranking criterion.
- system: 
    Batch LLM calls aggressively (e.g., in each quick‑select round or join phase) using an efficient inference engine.​
    Use vector indexes (e.g., FAISS) to speed up similarity search for sem_search, semantic joins, and clustering in group‑by.

"""

import json
import os
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")

import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store import FaissVS
from lotus_opt_config import build_cascade_args, load_opt_config

# ─────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────
LOCOMO_PATH = Path("../evermemos/evaluation/data/locomo/locomo10.json")
CONV_INDEX = 0                  # which conversation to process
MAX_MESSAGES = 40               # truncate to first N messages for testing
BOUNDARY_MSG_THRESHOLD = 15     # force boundary if >= N messages accumulated
TOPIC_SIM_THRESHOLD = 0.5       # cosine similarity threshold for topic matching
TOPIC_MAX_GAP_DAYS = 7          # only applied when both timestamps are parseable
PROFILE_MIN_SEGMENTS = 2        # min segments in a topic before profile distillation
OPT_CONFIG = load_opt_config()

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_JSON_PATH = Path(f"lotus_execution_log_{timestamp_str}.json")

# ─────────────────────────────────────────────────────
# Setup LOTUS
# ─────────────────────────────────────────────────────
print("=" * 60)
print("Setting up LOTUS with DeepSeek + SentenceTransformers")
print("=" * 60)

# We use DeepSeek Chat. Set retry max and rate limits to be safe.
# We also enable cache so we don't pay for re-runs.
lm = LM(
    model="deepseek/deepseek-chat",
    max_tokens=1024,
    temperature=0.0,
    max_batch_size=4,    # conservative to avoid rate limits
)
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
vs = FaissVS()
helper_lm = None
effective_proxy_model = OPT_CONFIG.proxy_model
if OPT_CONFIG.proxy_model == "helper_lm":
    if OPT_CONFIG.helper_model:
        helper_lm = LM(
            model=OPT_CONFIG.helper_model,
            max_tokens=1024,
            temperature=0.0,
            max_batch_size=4,
        )
        print(f"[OPT] Helper LM enabled: {OPT_CONFIG.helper_model}")
    else:
        effective_proxy_model = "embedding"
        print(
            "[OPT] LOTUS_OPT_PROXY_MODEL=helper_lm but LOTUS_HELPER_MODEL is unset. "
            "Falling back to embedding proxy."
        )

# Enable LOTUS cache (it creates a .lotus_cache directory by default)
configure_kwargs = {"lm": lm, "rm": rm, "vs": vs, "enable_cache": True}
if helper_lm is not None:
    configure_kwargs["helper_lm"] = helper_lm
lotus.settings.configure(**configure_kwargs)
print("LOTUS cache enabled (saves tokens on re-runs).")
FILTER_CASCADE_ARGS = (
    build_cascade_args(OPT_CONFIG, effective_proxy_model)
    if OPT_CONFIG.filter_cascade_enabled
    else None
)
JOIN_CASCADE_ARGS = (
    build_cascade_args(OPT_CONFIG, effective_proxy_model)
    if OPT_CONFIG.join_cascade_enabled
    else None
)
print(
    "[OPT] filter_cascade={}, join_cascade={}, topk_cascade={} (join/top-k unused in evermemos script)".format(
        OPT_CONFIG.filter_cascade_enabled,
        OPT_CONFIG.join_cascade_enabled,
        OPT_CONFIG.topk_cascade_enabled,
    )
)

# ─────────────────────────────────────────────────────
# Setup Logging
# ─────────────────────────────────────────────────────
exec_log = {
    "start_time": datetime.now().isoformat(),
    "config": {
        "max_messages": MAX_MESSAGES,
        "boundary_threshold": BOUNDARY_MSG_THRESHOLD,
        "topic_threshold": TOPIC_SIM_THRESHOLD,
        "optimizations": {
            "filter_cascade_enabled": OPT_CONFIG.filter_cascade_enabled,
            "join_cascade_enabled": OPT_CONFIG.join_cascade_enabled,
            "topk_cascade_enabled": OPT_CONFIG.topk_cascade_enabled,
            "proxy_model_requested": OPT_CONFIG.proxy_model,
            "proxy_model_effective": effective_proxy_model,
            "recall_target": OPT_CONFIG.recall_target,
            "precision_target": OPT_CONFIG.precision_target,
            "failure_probability": OPT_CONFIG.failure_probability,
            "sampling_percentage": OPT_CONFIG.sampling_percentage,
            "min_join_cascade_size": OPT_CONFIG.min_join_cascade_size,
            "helper_model": OPT_CONFIG.helper_model,
        },
    },
    "operations": [],
    "summary": {}
}

def log_operation(op_type, df_in, prompt, df_out=pd.DataFrame(), error=None, extra=None):
    """Log the inputs, outputs, and any errors of an operator."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "operator": op_type,
        "prompt": prompt,
        "input_rows": len(df_in),
        "inputs": df_in.to_dict(orient="records"),
    }
    
    if error:
        record["status"] = "ERROR"
        record["error"] = str(error)
        record["traceback"] = traceback.format_exc()
    else:
        record["status"] = "SUCCESS"
        record["output_rows"] = len(df_out)
        record["outputs"] = df_out.to_dict(orient="records")
    if extra:
        record.update(extra)
        
    exec_log["operations"].append(record)
    
    # Save incrementally
    with open(LOG_JSON_PATH, "w") as f:
        json.dump(exec_log, f, indent=2, ensure_ascii=False)


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
print(f"Total messages available: {len(all_messages)}")

# LIMIT DATA
all_messages = all_messages[:MAX_MESSAGES]
print(f"TRUNCATED to {len(all_messages)} messages for this test.")

def format_messages(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        lines.append(f"[{m['session_time']}] {m['speaker']}: {m['text']}")
    return "\n".join(lines)


def parse_session_time(session_time: str | None) -> datetime | None:
    """
    Parse LOCOMO session time strings like "1:56 pm on 8 May, 2023".
    Returns None if parsing fails so downstream can safely skip time-gap checks.
    """
    if not session_time or not isinstance(session_time, str):
        return None

    normalized = session_time.strip()
    normalized = normalized.replace(" am on ", " AM on ").replace(" pm on ", " PM on ")

    fmts = [
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %b, %Y",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None


# ─────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────
history_segments = []        
topics = {}                  
profiles = {}                
segment_counter = 0
topic_counter = 0
tumbling_window = []  

print("\n" + "=" * 60)
print("Starting EverMemOS pipeline")
print("=" * 60)

for msg_idx, msg in enumerate(all_messages):
    tumbling_window.append(msg)
    print(f"\n--- Message {msg_idx + 1}/{len(all_messages)}: {msg['speaker']}: {msg['text'][:60]}...")

    # Q1: Boundary Detection
    force_boundary = len(tumbling_window) >= BOUNDARY_MSG_THRESHOLD

    if not force_boundary and len(tumbling_window) >= 3:
        window_text = format_messages(tumbling_window)
        boundary_df = pd.DataFrame({"content": [window_text]})
        filter_prompt = (
            "The conversation in {content} has reached a natural semantic boundary "
            "due to a clear topic shift, a significant time gap between messages, "
            "or a logical conclusion of the discussion"
        )
        
        try:
            q1_cascade_args = FILTER_CASCADE_ARGS
            q1_stats = None
            boundary_df_for_run = boundary_df

            if q1_cascade_args is not None and effective_proxy_model == "embedding":
                try:
                    boundary_df_for_run = boundary_df.sem_index("content", "ever_q1_boundary_content_idx")
                except Exception as idx_err:
                    print(
                        "  [OPT] Q1 cascade index setup failed "
                        f"({idx_err}). Falling back to standard sem_filter."
                    )
                    q1_cascade_args = None

            if q1_cascade_args is not None:
                try:
                    result, q1_stats = boundary_df_for_run.sem_filter(
                        filter_prompt,
                        cascade_args=q1_cascade_args,
                        return_stats=True,
                    )
                except Exception as cascade_err:
                    print(
                        "  [OPT] Q1 cascade sem_filter failed "
                        f"({cascade_err}). Falling back to standard sem_filter."
                    )
                    result = boundary_df.sem_filter(filter_prompt)
                    q1_cascade_args = None
                    q1_stats = None
            else:
                result = boundary_df_for_run.sem_filter(filter_prompt)

            extra = {"cascade_enabled": q1_cascade_args is not None}
            if q1_stats is not None:
                extra["cascade_stats"] = q1_stats
            log_operation("Q1_sem_filter", boundary_df, filter_prompt, result, extra=extra)
            is_boundary = len(result) > 0
            print(f"  Q1 boundary check: {'BOUNDARY DETECTED' if is_boundary else 'continuing...'}")
        except Exception as e:
            print(f"  Q1 boundary check error: {e}")
            log_operation("Q1_sem_filter", boundary_df, filter_prompt, error=e)
            is_boundary = False
            
    elif force_boundary:
        is_boundary = True
        print(f"  Q1 FORCE boundary (>= {BOUNDARY_MSG_THRESHOLD} messages)")
    else:
        is_boundary = False
        print(f"  Q1 skipped (need >= 3 messages)")

    if not is_boundary:
        continue

    # ═════════════════════════════════════════════════
    # BOUNDARY REACHED
    # ═════════════════════════════════════════════════
    segment_id = f"seg_{segment_counter}"
    segment_counter += 1
    window_content = format_messages(tumbling_window)
    n_msgs = len(tumbling_window)
    segment_timestamp = parse_session_time(tumbling_window[-1].get("session_time"))

    print(f"\n{'─' * 50}")
    print(f"  SEALED SEGMENT {segment_id} ({n_msgs} messages)")
    print(f"{'─' * 50}")

    segment_df = pd.DataFrame({"content": [window_content]})

    # Q2 + Q3: Episode synthesis & Subject Extraction
    print("\n  [Q2+Q3] sem_extract (episode + subject)...")
    episode_text, subject_text = "N/A", "N/A"
    try:
        extract_cols = {
            "episode": "A concise third-person episodic narrative capturing key events",
            "subject": "The central subject or topic of this conversation in a few words"
        }
        episode_result = segment_df.sem_extract(
            input_cols=["content"],
            output_cols=extract_cols,
        )
        log_operation("Q2_Q3_sem_extract", segment_df, extract_cols, episode_result)
        
        if "episode" in episode_result.columns:
            episode_text = str(episode_result["episode"].iloc[0])
        if "subject" in episode_result.columns:
            subject_text = str(episode_result["subject"].iloc[0])
    except Exception as e:
        print(f"  ERROR in Q2+Q3: {e}")
        log_operation("Q2_Q3_sem_extract", segment_df, extract_cols, error=e)
        episode_text = "Extraction failed"

    print(f"  Episode: {episode_text[:100]}...")
    print(f"  Subject: {subject_text}")

    # Q4: Foresights
    print("\n  [Q4] sem_extract (foresights)...")
    foresights = []
    try:
        f_cols = {
            "prediction": "A future prediction or planned action mentioned",
            "timeframe": "When this prediction or action is expected to happen"
        }
        foresight_result = segment_df.sem_extract(input_cols=["content"], output_cols=f_cols)
        log_operation("Q4_sem_extract", segment_df, f_cols, foresight_result)
        
        if "prediction" in foresight_result.columns:
            for _, row in foresight_result.iterrows():
                p = str(row.get("prediction", ""))
                if p and p.lower() != "n/a" and p.lower() != "none" and len(p.strip()) > 3:
                    foresights.append({"prediction": p, "timeframe": str(row.get("timeframe", ""))})
    except Exception as e:
        print(f"  ERROR in Q4: {e}")
        log_operation("Q4_sem_extract", segment_df, f_cols, error=e)

    for f in foresights: print(f"  → {f['prediction']} ({f['timeframe']})")

    # Q5: Facts
    print("\n  [Q5] sem_extract (facts)...")
    facts = []
    try:
        fact_cols = {
            "atomic_fact": "A discrete atomic factual event (who did what, when)",
            "timestamp": "When this event happened or was mentioned"
        }
        fact_result = segment_df.sem_extract(input_cols=["content"], output_cols=fact_cols)
        log_operation("Q5_sem_extract", segment_df, fact_cols, fact_result)
        
        if "atomic_fact" in fact_result.columns:
            for _, row in fact_result.iterrows():
                a = str(row.get("atomic_fact", ""))
                if a and a.lower() != "n/a" and a.lower() != "none" and len(a.strip()) > 3:
                    facts.append({"atomic_fact": a, "timestamp": str(row.get("timestamp", ""))})
    except Exception as e:
        print(f"  ERROR in Q5: {e}")
        log_operation("Q5_sem_extract", segment_df, fact_cols, error=e)

    for f in facts: print(f"  → {f['atomic_fact']} ({f['timestamp']})")


    # Q6: Topic Clustering
    print("\n  [Q6] Topic clustering (SentenceTransformers)...")
    episode_embedding = rm([episode_text])[0]
    assigned_topic = None
    best_sim = -1.0
    best_gap_days = None

    for tid, topic_info in topics.items():
        centroid = topic_info["centroid"]
        sim = float(np.dot(episode_embedding, centroid) /
                     (np.linalg.norm(episode_embedding) * np.linalg.norm(centroid) + 1e-9))

        topic_timestamp = topic_info.get("last_timestamp")
        gap_days = None
        time_gap_ok = True
        if segment_timestamp is not None and topic_timestamp is not None:
            gap_days = abs((segment_timestamp - topic_timestamp).total_seconds()) / 86400.0
            time_gap_ok = gap_days <= TOPIC_MAX_GAP_DAYS

        if sim > best_sim:
            best_sim = sim
            best_gap_days = gap_days
            if sim >= TOPIC_SIM_THRESHOLD and time_gap_ok:
                assigned_topic = tid

    gap_text = "N/A" if best_gap_days is None else f"{best_gap_days:.2f}d"
    if assigned_topic is None:
        assigned_topic = f"topic_{topic_counter}"
        topic_counter += 1
        topics[assigned_topic] = {
            "centroid": episode_embedding,
            "segments": [],
            "segment_count": 0,
            "last_timestamp": segment_timestamp,
        }
        print(f"  NEW topic: {assigned_topic} (sim={best_sim:.3f}, gap={gap_text})")
    else:
        old_centroid = topics[assigned_topic]["centroid"]
        n = topics[assigned_topic]["segment_count"]
        topics[assigned_topic]["centroid"] = (old_centroid * n + episode_embedding) / (n + 1)
        if segment_timestamp is not None:
            topics[assigned_topic]["last_timestamp"] = segment_timestamp
        print(f"  MERGED into {assigned_topic} (sim={best_sim:.3f}, gap={gap_text})")

    topics[assigned_topic]["segments"].append(segment_id)
    topics[assigned_topic]["segment_count"] += 1

    history_segments.append({
        "segment_id": segment_id,
        "content": window_content,
        "episode": episode_text,
        "subject": subject_text,
        "foresights": foresights,
        "facts": facts,
        "topic_id": assigned_topic,
        "segment_time": segment_timestamp.isoformat() if segment_timestamp else None,
    })

    # Q7: Profile Distillation
    topic_seg_count = topics[assigned_topic]["segment_count"]
    if topic_seg_count % PROFILE_MIN_SEGMENTS == 0:  # e.g. every 2 segments
        print(f"\n  [Q7] sem_agg (Profile distillation for {assigned_topic})...")
        topic_episodes = [seg["episode"] for seg in history_segments if seg["topic_id"] == assigned_topic]
        
        episodes_df = pd.DataFrame({
            "episode": topic_episodes,
            "topic_id": [assigned_topic] * len(topic_episodes)
        })
        existing_traits = profiles.get(assigned_topic, {}).get("traits", "No existing profile.")
        episodes_df["existing_traits"] = existing_traits
        
        agg_prompt = (
            "Given the existing user profile: {existing_traits}, and these conversation episodes: "
            "{episode}, distill updated stable user traits including personality, interests, "
            "skills, and behavioral patterns. Output a clean summary."
        )
        try:
            profile_result = episodes_df.sem_agg(agg_prompt, group_by=["topic_id"])
            log_operation("Q7_sem_agg", episodes_df, agg_prompt, profile_result)
            
            if "_output" in profile_result.columns:
                new_traits = str(profile_result["_output"].iloc[0])
                profiles[assigned_topic] = {"traits": new_traits}
                print(f"  Profile updated.")
        except Exception as e:
            print(f"  ERROR in Q7: {e}")
            log_operation("Q7_sem_agg", episodes_df, agg_prompt, error=e)
            
    # clear window
    tumbling_window = []

# Finalize
exec_log["end_time"] = datetime.now().isoformat()
exec_log["summary"] = {
    "total_segments": len(history_segments),
    "total_topics": len(topics),
    "virtual_cost_usd": lm.stats.virtual_usage.total_cost,
    "physical_cost_usd": lm.stats.physical_usage.total_cost,
    "cache_hits": lm.stats.cache_hits,
    "operator_cache_hits": lm.stats.operator_cache_hits,
}
with open(LOG_JSON_PATH, "w") as f:
    json.dump(exec_log, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"Logged all I/O to: {LOG_JSON_PATH}")
print(f"Total segments created: {len(history_segments)}")
lm.print_total_usage()
