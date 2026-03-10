"""
Zep / Graphiti Semantic Queries via LOTUS
=====================================
Executes Zep's memory graph insertion queries using LOTUS operators on LOCOMO data.
Simulates:
- Fixed Window message accumulation
- Q1/Q4: Entity (Node) and Fact (Edge) Extraction via sem_extract
- Q2: Entity Resolution via sem_join
- Q5/Q6: Edge Deduplication and Contradiction via sem_join
- Q3/Q7/Q8: Summary and Community Updates via sem_map (simulating Zep's N-to-1 race condition loop)

Usage:
    source .venv/bin/activate
    python zep_lotus.py
"""

import json
import os
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")

import lotus
from lotus.models.lm import LM
from lotus.vector_store import FaissVS
from rm_factory import build_rm
from sim_config import (
    build_cascade_args,
    build_opt_config,
    build_resolved_config_snapshot,
    get_pipeline_helper_model,
    get_pipeline_main_model,
    get_pipeline_settings,
    load_sim_config,
)

# ─────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────
PIPELINE_NAME = "zep"
SIM_CONFIG = load_sim_config()
PIPELINE_CONFIG = get_pipeline_settings(SIM_CONFIG, PIPELINE_NAME)
MAIN_MODEL_CONFIG = get_pipeline_main_model(SIM_CONFIG, PIPELINE_NAME)
HELPER_MODEL_CONFIG = get_pipeline_helper_model(SIM_CONFIG, PIPELINE_NAME)
OPTIMIZATION_CONFIG = SIM_CONFIG.optimizations

LOCOMO_PATH = SIM_CONFIG.global_config.locomo_path
CONV_INDEX = SIM_CONFIG.global_config.conv_index
MAX_MESSAGES = PIPELINE_CONFIG.max_messages
SLIDING_WINDOW_SIZE = PIPELINE_CONFIG.sliding_window_size
AUTO_BUILD_COMMUNITIES_AFTER_MSG = PIPELINE_CONFIG.auto_build_communities_after_msg
OPT_CONFIG = build_opt_config(
    optimizations=OPTIMIZATION_CONFIG,
    helper_model=HELPER_MODEL_CONFIG,
)
RESOLVED_CONFIG = build_resolved_config_snapshot(SIM_CONFIG, PIPELINE_NAME)

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_JSON_PATH = Path(f"zep_execution_log_{timestamp_str}.json")

# ─────────────────────────────────────────────────────
# Setup LOTUS
# ─────────────────────────────────────────────────────
print("=" * 60)
print("Setting up LOTUS for Zep / Graphiti Pipeline")
print("=" * 60)

lm = LM(
    model=MAIN_MODEL_CONFIG.model,
    max_tokens=MAIN_MODEL_CONFIG.max_tokens,
    temperature=MAIN_MODEL_CONFIG.temperature,
    max_batch_size=MAIN_MODEL_CONFIG.max_batch_size,
    **MAIN_MODEL_CONFIG.kwargs,
)
rm = build_rm(SIM_CONFIG.rm_model)
print(f"[RM] backend={SIM_CONFIG.rm_model.backend} model={SIM_CONFIG.rm_model.model}")
vs = FaissVS()
helper_lm = None
effective_proxy_model = OPT_CONFIG.proxy_model
if OPT_CONFIG.proxy_model == "helper_lm":
    if OPT_CONFIG.helper_model:
        helper_lm = LM(
            model=HELPER_MODEL_CONFIG.model,
            max_tokens=HELPER_MODEL_CONFIG.max_tokens,
            temperature=HELPER_MODEL_CONFIG.temperature,
            max_batch_size=HELPER_MODEL_CONFIG.max_batch_size,
            **HELPER_MODEL_CONFIG.kwargs,
        )
        print(f"[OPT] Helper LM enabled: {HELPER_MODEL_CONFIG.model}")
    else:
        effective_proxy_model = "embedding"
        print(
            "[OPT] proxy_model=helper_lm but helper model is not configured. "
            "Falling back to embedding proxy."
        )

configure_kwargs = {
    "lm": lm,
    "rm": rm,
    "vs": vs,
    "enable_cache": SIM_CONFIG.global_config.enable_cache,
}
if helper_lm is not None:
    configure_kwargs["helper_lm"] = helper_lm
lotus.settings.configure(**configure_kwargs)
print(f"LOTUS cache enabled={SIM_CONFIG.global_config.enable_cache}.")
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
    "[OPT] filter_cascade={}, join_cascade={}, topk_cascade={} (top-k unused in zep script)".format(
        OPT_CONFIG.filter_cascade_enabled,
        OPT_CONFIG.join_cascade_enabled,
        OPT_CONFIG.topk_cascade_enabled,
    )
)

# ─────────────────────────────────────────────────────
# Logging utility
# ─────────────────────────────────────────────────────
exec_log = {
    "start_time": datetime.now().isoformat(),
    "config": {
        "config_path": str(SIM_CONFIG.config_path),
        "resolved_config": RESOLVED_CONFIG,
        "locomo_path": str(LOCOMO_PATH),
        "conv_index": CONV_INDEX,
        "max_messages": MAX_MESSAGES,
        "window": SLIDING_WINDOW_SIZE,
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

def log_operation(op_type, prompt, df_in, df_out=pd.DataFrame(), error=None, extra=None):
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
        record["outputs"] = df_out.to_dict(orient="records")
    if extra:
        record.update(extra)
    exec_log["operations"].append(record)
    with open(LOG_JSON_PATH, "w") as f:
        json.dump(exec_log, f, indent=2, ensure_ascii=False)


def sem_join_with_optional_cascade(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    join_prompt: str,
) -> tuple[pd.DataFrame, dict[str, Any] | None, bool]:
    if JOIN_CASCADE_ARGS is None:
        return left_df.sem_join(right_df, join_prompt), None, False

    full_join_size = len(left_df) * len(right_df)
    min_join_size = int(getattr(JOIN_CASCADE_ARGS, "min_join_cascade_size", 0) or 0)

    # sem_join cascade only triggers when join cardinality reaches min_join_cascade_size.
    if full_join_size < min_join_size:
        joined_df = left_df.sem_join(right_df, join_prompt)
        return joined_df, {
            "cascade_skipped": "full_join_below_threshold",
            "full_join_size": full_join_size,
            "min_join_cascade_size": min_join_size,
        }, False

    try:
        join_result = left_df.sem_join(
            right_df,
            join_prompt,
            cascade_args=JOIN_CASCADE_ARGS,
            return_stats=True,
        )

        # LOTUS may return either:
        # - (joined_df, stats) when cascade stats are available
        # - joined_df only when stats are not produced
        if isinstance(join_result, tuple):
            if len(join_result) == 2:
                joined_df, join_stats = join_result
                return joined_df, join_stats, True
            raise ValueError(f"Unexpected sem_join return tuple length: {len(join_result)}")

        if isinstance(join_result, pd.DataFrame):
            return join_result, {
                "cascade_returned_no_stats": True,
                "full_join_size": full_join_size,
                "min_join_cascade_size": min_join_size,
            }, False

        raise TypeError(f"Unexpected sem_join return type: {type(join_result).__name__}")
    except Exception as cascade_err:
        print(
            "    [OPT] Join cascade failed "
            f"({cascade_err}). Falling back to standard sem_join."
        )
        joined_df = left_df.sem_join(right_df, join_prompt)
        return joined_df, {"cascade_fallback_error": str(cascade_err)}, False


# ─────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────
print("\nLoading LOCOMO data...")
with open(LOCOMO_PATH) as f:
    locomo_data = json.load(f)

messages = []
conv = locomo_data[CONV_INDEX]["conversation"]
session_idx = 1
while f"session_{session_idx}" in conv:
    for msg in conv[f"session_{session_idx}"]:
        messages.append({"speaker": msg["speaker"], "text": msg["text"]})
    session_idx += 1

messages = messages[:MAX_MESSAGES]
print(f"Loaded {len(messages)} messages for processing.")


# ─────────────────────────────────────────────────────
# Global Memory DB (Simulates Zep's Neo4j Graph)
# ─────────────────────────────────────────────────────
# DataFrames to act as Node, Edge, and Community tables
history_entities_df = pd.DataFrame(columns=["uuid", "name", "summary"])
history_edges_df = pd.DataFrame(columns=["uuid", "src_name", "tgt_name", "fact", "invalidated"])
history_communities_df = pd.DataFrame(columns=["uuid", "name", "summary"])
community_membership = {}  # dict mapping entity_uuid -> community_uuid

ent_counter = 0
edge_counter = 0
comm_counter = 0


def infer_dominant_neighbor_community(
    entity_name: str,
    active_edges: pd.DataFrame,
    entity_name_to_uuid: dict[str, str],
    membership: dict[str, str],
) -> str | None:
    """
    Mimic Graphiti insertion behavior: if the entity itself has no community,
    infer from neighboring entities' existing communities (mode).
    """
    community_counts: dict[str, int] = {}

    for _, edge in active_edges.iterrows():
        src_name = str(edge.get("src_name", ""))
        tgt_name = str(edge.get("tgt_name", ""))

        neighbor_name = None
        if src_name == entity_name:
            neighbor_name = tgt_name
        elif tgt_name == entity_name:
            neighbor_name = src_name

        if not neighbor_name:
            continue

        neighbor_uuid = entity_name_to_uuid.get(neighbor_name)
        if not neighbor_uuid:
            continue

        neighbor_comm = membership.get(neighbor_uuid)
        if not neighbor_comm:
            continue

        community_counts[neighbor_comm] = community_counts.get(neighbor_comm, 0) + 1

    if not community_counts:
        return None

    return max(community_counts.items(), key=lambda kv: kv[1])[0]


def build_initial_communities_once(
    entities_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    communities_df: pd.DataFrame,
    membership: dict[str, str],
    comm_counter_start: int,
) -> tuple[pd.DataFrame, dict[str, str], int, int]:
    """
    Simulate Graphiti's standalone build_communities flow:
    build community snapshots from current graph topology so later insertion updates can work.
    """
    if len(entities_df) == 0:
        return communities_df, membership, comm_counter_start, 0

    active_edges = edges_df[edges_df["invalidated"] == False]
    if len(active_edges) == 0:
        return communities_df, membership, comm_counter_start, 0

    name_to_uuid = {
        str(row["name"]): str(row["uuid"]) for _, row in entities_df.iterrows()
    }
    uuid_to_summary = {
        str(row["uuid"]): str(row.get("summary", "")) for _, row in entities_df.iterrows()
    }

    # Undirected adjacency for connected-component community seeds.
    adj: dict[str, set[str]] = {}
    for _, edge in active_edges.iterrows():
        s_name = str(edge.get("src_name", ""))
        t_name = str(edge.get("tgt_name", ""))
        s_uuid = name_to_uuid.get(s_name)
        t_uuid = name_to_uuid.get(t_name)
        if not s_uuid or not t_uuid:
            continue
        adj.setdefault(s_uuid, set()).add(t_uuid)
        adj.setdefault(t_uuid, set()).add(s_uuid)

    visited: set[str] = set()
    components: list[list[str]] = []

    for start in adj.keys():
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        comp: list[str] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nei in adj.get(cur, set()):
                if nei not in visited:
                    visited.add(nei)
                    stack.append(nei)
        if len(comp) > 0:
            components.append(comp)

    built_count = 0
    comm_counter = comm_counter_start

    for comp in components:
        summaries = [
            uuid_to_summary.get(u, "") for u in comp if uuid_to_summary.get(u, "").strip()
        ]
        if len(summaries) == 0:
            continue

        summary_input = pd.DataFrame({"member_summaries": ["\n".join(summaries)]})
        summary_prompt = (
            "Given these entity summaries from one graph community:\n{member_summaries}\n"
            "Write one concise community summary."
        )
        try:
            summary_res = summary_input.sem_map(summary_prompt)
            log_operation("Q_BUILD_sem_map_summary", summary_prompt, summary_input, summary_res)
            community_summary = str(summary_res["_map"].iloc[0]).strip()
        except Exception as e:
            log_operation("Q_BUILD_sem_map_summary", summary_prompt, summary_input, error=e)
            community_summary = summaries[0]

        name_input = pd.DataFrame({"fuse": [community_summary]})
        name_prompt = "Create a short one-sentence title (5 words max) summarizing: {fuse}"
        try:
            name_res = name_input.sem_map(name_prompt)
            log_operation("Q_BUILD_sem_map_name", name_prompt, name_input, name_res)
            community_name = str(name_res["_map"].iloc[0]).strip()
        except Exception as e:
            log_operation("Q_BUILD_sem_map_name", name_prompt, name_input, error=e)
            community_name = "Community"

        c_uuid = f"comm_{comm_counter}"
        comm_counter += 1
        communities_df.loc[len(communities_df)] = [c_uuid, community_name, community_summary]
        for u in comp:
            membership[u] = c_uuid
        built_count += 1

    return communities_df, membership, comm_counter, built_count

print("\n" + "=" * 60)
print("Starting Zep Pipeline")
print("=" * 60)

# Zep processes EVERY incoming request, even before the window reaches full length.
# It always uses the latest up-to-10 messages as context for that request.
auto_build_done = False
for msg_idx in range(len(messages)):
    w_start = max(0, msg_idx - SLIDING_WINDOW_SIZE + 1)
    window_msgs = messages[w_start:msg_idx+1]
    
    window_text = "\n".join([f"{m['speaker']}: {m['text']}" for m in window_msgs])
    print(f"\n--- Zep Processing Msg {msg_idx} | Context Window [{w_start}:{msg_idx}] (Len: {len(window_msgs)}) ---")
    
    window_df = pd.DataFrame({"content": [window_text]})

    # ══════════════════════════════════════════════════════════
    # PHASE 1: ENTITY (NODE) EXTRACTION AND RESOLUTION
    # ══════════════════════════════════════════════════════════

    print("  [Q1] Node Extraction (sem_extract)...")
    ent_cols = {"name": "entity name", "summary": "brief entity description based on current context"}
    try:
        recent_entities = window_df.sem_extract(input_cols=["content"], output_cols=ent_cols)
        log_operation("Q1_sem_extract", ent_cols, window_df, recent_entities)
    except Exception as e:
        print(f"    Error in Q1: {e}")
        log_operation("Q1_sem_extract", ent_cols, window_df, error=e)
        continue

    # Filter out empty extractions
    if "name" not in recent_entities.columns:
        continue
    recent_entities = recent_entities[recent_entities["name"].astype(str).str.strip() != ""]
    if len(recent_entities) == 0:
        continue

    print("  [Q2] Node Resolution (sem_join)...")
    resolved_entities = [] # list of dicts: {"uuid", "name", "summary_updated"}
    
    if len(history_entities_df) == 0:
        print("    History is empty. Auto-adding current entities.")
        for _, r in recent_entities.iterrows():
            e_name = str(r["name"])
            e_summ = str(r.get("summary", ""))
            e_uuid = f"ent_{ent_counter}"
            ent_counter += 1
            history_entities_df.loc[len(history_entities_df)] = [e_uuid, e_name, e_summ]
            resolved_entities.append({"uuid": e_uuid, "name": e_name, "summary_updated": e_summ})
    else:
        # Zep resolves entities by doing a cross-check between recent and history
        # LOTUS's true sem_join implementation:
        try:
            join_prompt = "{name:left} and {name:right} refer to the identical real-world object or concept."
            # Note: We do an inner join to find matches. If a recent entity has no match, it's new.
            matches, q2_join_stats, q2_cascade_used = sem_join_with_optional_cascade(
                recent_entities, history_entities_df, join_prompt
            )
            extra = {"cascade_enabled": q2_cascade_used}
            if q2_join_stats is not None:
                extra["cascade_stats"] = q2_join_stats
            log_operation("Q2_sem_join", join_prompt, recent_entities, matches, extra=extra)
            
            matched_recent_names = matches["name_left"].tolist() if "name_left" in matches.columns else []
            
            for _, r in recent_entities.iterrows():
                e_name = str(r["name"])
                e_summ = str(r.get("summary", ""))
                
                # Check if this recent entity matched any history entity
                match_row = matches[matches["name_left"] == e_name] if "name_left" in matches.columns else pd.DataFrame()
                
                if len(match_row) > 0:
                    # Entity exists -> Update Summary (Q3)
                    e_uuid = match_row.iloc[0]["uuid"]
                    hist_summ = match_row.iloc[0]["summary_right"]
                    print(f"    [Q3] Entity EXISTS ({e_name}). Merging summaries (sem_map)...")
                    
                    merge_df = pd.DataFrame({"ctx": [window_text], "hist_sum": [hist_summ], "new_sum": [e_summ]})
                    merge_prompt = "Context: {ctx}\nOld: {hist_sum}\nNew: {new_sum}\nSynthesize an updated concise summary for the entity."
                    merged_res = merge_df.sem_map(merge_prompt)
                    log_operation("Q3_sem_map", merge_prompt, merge_df, merged_res)
                    
                    updated_summ = str(merged_res["_map"].iloc[0])
                    
                    # Update global DB
                    history_entities_df.loc[history_entities_df["uuid"] == e_uuid, "summary"] = updated_summ
                    resolved_entities.append({"uuid": e_uuid, "name": e_name, "summary_updated": updated_summ})
                else:
                    # Entity is new -> Add
                    print(f"    Entity NEW ({e_name}). Adding.")
                    e_uuid = f"ent_{ent_counter}"
                    ent_counter += 1
                    history_entities_df.loc[len(history_entities_df)] = [e_uuid, e_name, e_summ]
                    resolved_entities.append({"uuid": e_uuid, "name": e_name, "summary_updated": e_summ})
                    
        except Exception as e:
            print(f"    Error in Q2/Q3: {e}")
            log_operation("Q2_sem_join", "Entity Resolution", recent_entities, error=e)
            continue


    # ══════════════════════════════════════════════════════════
    # PHASE 2: FACT (EDGE) EXTRACTION AND RESOLUTION
    # ══════════════════════════════════════════════════════════

    print("  [Q4] Fact/Edge Extraction (sem_extract)...")
    # Zep explicitly uses resolved resolved entity names to guide fact extraction
    resolved_names = ", ".join([e["name"] for e in resolved_entities])
    edge_input_df = pd.DataFrame({"content": [window_text], "entities": [resolved_names]})
    edge_cols = {
        "src_name": "source entity",
        "tgt_name": "target entity",
        "fact": "factual relationship description"
    }
    try:
        recent_edges = edge_input_df.sem_extract(input_cols=["content", "entities"], output_cols=edge_cols)
        log_operation("Q4_sem_extract", edge_cols, edge_input_df, recent_edges)
    except Exception as e:
        print(f"    Error in Q4: {e}")
        log_operation("Q4_sem_extract", edge_cols, edge_input_df, error=e)
        continue

    if "fact" not in recent_edges.columns:
        continue
    recent_edges = recent_edges[recent_edges["fact"].astype(str).str.strip() != ""]
    if len(recent_edges) == 0:
        continue
        
    for _, edge_row in recent_edges.iterrows():
        s_name = str(edge_row.get("src_name", ""))
        t_name = str(edge_row.get("tgt_name", ""))
        fact_desc = str(edge_row.get("fact", ""))
        
        # Zep simulates graph relationships. Skip ill-formed edges.
        if not s_name or not t_name or s_name == "None" or t_name == "None":
            continue
            
        print(f"    Evaluating Edge: {s_name} -> {t_name} ({fact_desc})")
        
        if len(history_edges_df) == 0:
            print("      History is empty. Auto-adding edge.")
            e_uuid = f"edge_{edge_counter}"
            edge_counter += 1
            history_edges_df.loc[len(history_edges_df)] = [e_uuid, s_name, t_name, fact_desc, False]
            continue
            
        # Zep checks logic against active historical edges using sem_join
        active_hist_edges = history_edges_df[history_edges_df["invalidated"] == False]
        cur_edge_df = pd.DataFrame({"fact": [fact_desc]})
        
        is_dup, is_contra = False, False
        
        # [Q5] Duplicate Detection
        # Strict topology pre-filter: only compare edges with identical src/tgt endpoints.
        topology_hist_edges = active_hist_edges[
            (active_hist_edges["src_name"] == s_name) & (active_hist_edges["tgt_name"] == t_name)
        ]
        dup_prompt = "Fact '{fact:left}' represents absolutely identical factual information as Fact '{fact:right}'"
        try:
            if len(topology_hist_edges) > 0:
                dup_matches, q5_join_stats, q5_cascade_used = sem_join_with_optional_cascade(
                    cur_edge_df, topology_hist_edges, dup_prompt
                )
            else:
                dup_matches = pd.DataFrame()
                q5_join_stats, q5_cascade_used = None, False
            extra = {"cascade_enabled": q5_cascade_used}
            if q5_join_stats is not None:
                extra["cascade_stats"] = q5_join_stats
            log_operation("Q5_sem_join", dup_prompt, cur_edge_df, dup_matches, extra=extra)
            if len(dup_matches) > 0:
                is_dup = True
                print("      [Q5] DUP DETECTED.")
        except Exception as e:
            print(f"      Q5 Error: {e}")
            log_operation("Q5_sem_join", dup_prompt, cur_edge_df, error=e)

        # [Q6] Contradiction Detection
        if not is_dup:
            contra_prompt = "Fact '{fact:left}' fundamentally contradicts or supersedes Fact '{fact:right}'"
            try:
                # Assuming loose topology, cross-check against all active
                contra_matches, q6_join_stats, q6_cascade_used = sem_join_with_optional_cascade(
                    cur_edge_df, active_hist_edges, contra_prompt
                )
                extra = {"cascade_enabled": q6_cascade_used}
                if q6_join_stats is not None:
                    extra["cascade_stats"] = q6_join_stats
                log_operation("Q6_sem_join", contra_prompt, cur_edge_df, contra_matches, extra=extra)
                if len(contra_matches) > 0:
                    is_contra = True
                    print(f"      [Q6] CONTRADICTION DETECTED. Invalidating {len(contra_matches)} old facts.")
                    # Invalidate old facts
                    for uuid_to_inv in contra_matches["uuid"].tolist():
                        history_edges_df.loc[history_edges_df["uuid"] == uuid_to_inv, "invalidated"] = True
            except Exception as e:
                print(f"      Q6 Error: {e}")
                log_operation("Q6_sem_join", contra_prompt, cur_edge_df, error=e)

        # Insert if not dup
        if not is_dup:
            print("      Adding New Edge.")
            e_uuid = f"edge_{edge_counter}"
            edge_counter += 1
            history_edges_df.loc[len(history_edges_df)] = [e_uuid, s_name, t_name, fact_desc, False]


    # ══════════════════════════════════════════════════════════
    # PHASE 3: COMMUNITY UPDATE (SIMULATING ZEP'S N->1 RACE LOOP)
    # ══════════════════════════════════════════════════════════
    # Optional one-time bootstrap: run standalone community build so insertion updates can start working.
    if (
        not auto_build_done
        and len(history_communities_df) == 0
        and (msg_idx + 1) >= AUTO_BUILD_COMMUNITIES_AFTER_MSG
    ):
        print(
            f"  [BUILD] Triggering one-time build_communities snapshot at msg {msg_idx + 1}..."
        )
        (
            history_communities_df,
            community_membership,
            comm_counter,
            built_count,
        ) = build_initial_communities_once(
            history_entities_df,
            history_edges_df,
            history_communities_df,
            community_membership,
            comm_counter,
        )
        auto_build_done = True
        print(f"  [BUILD] Communities built: {built_count}")

    # Zep insertion only updates EXISTING communities. It does not create/rebuild communities here.
    # If no communities were built beforehand, this phase effectively does nothing.
    active_edges_for_community = history_edges_df[history_edges_df["invalidated"] == False]
    entity_name_to_uuid = {
        str(row["name"]): str(row["uuid"]) for _, row in history_entities_df.iterrows()
    }
    
    for ent in resolved_entities:
        e_uuid = ent["uuid"]
        e_name = ent["name"]
        e_summ = ent["summary_updated"]
        
        c_uuid = community_membership.get(e_uuid)
        
        if c_uuid is None:
            # Try to infer from neighboring entities' existing communities (mode).
            inferred_c_uuid = infer_dominant_neighbor_community(
                e_name,
                active_edges_for_community,
                entity_name_to_uuid,
                community_membership,
            )
            if inferred_c_uuid is not None:
                c_uuid = inferred_c_uuid
                community_membership[e_uuid] = c_uuid
                print(f"  [Q7] Node {e_name} assigned to existing community {c_uuid} via neighbors.")

        if c_uuid is None:
            print(f"  [Q7] Node {e_name} has no existing community context. Skipping update.")
            continue

        c_rows = history_communities_df[history_communities_df["uuid"] == c_uuid]
        if len(c_rows) == 0:
            print(f"  [Q7] Community {c_uuid} missing in DB snapshot. Skipping update.")
            continue

        # Update existing community via pairwise summary fusion (known N->1 write limitation).
        print(f"  [Q7/Q8] Updating existing Community for Node {e_name}...")
        c_hist_summ = c_rows.iloc[0]["summary"]

        comm_df = pd.DataFrame({"e_sum": [e_summ], "c_sum": [c_hist_summ]})
        fuse_prompt = "Entity info: {e_sum}\nCommunity info: {c_sum}\nSynthesize these two into a single succinct community summary."

        try:
            fuse_res = comm_df.sem_map(fuse_prompt)
            log_operation("Q7_sem_map_community", fuse_prompt, comm_df, fuse_res)
            fused_summary = str(fuse_res["_map"].iloc[0])

            # Q8 Name generation
            name_df = pd.DataFrame({"fuse": [fused_summary]})
            name_prompt = "Create a short one-sentence title (5 words max) summarizing: {fuse}"
            name_res = name_df.sem_map(name_prompt)
            log_operation("Q8_sem_map_naming", name_prompt, name_df, name_res)

            new_name = str(name_res["_map"].iloc[0])

            history_communities_df.loc[history_communities_df["uuid"] == c_uuid, "summary"] = fused_summary
            history_communities_df.loc[history_communities_df["uuid"] == c_uuid, "name"] = new_name
        except Exception as e:
            print(f"    Error in Community Update: {e}")
            log_operation("Q7_Q8_sem_map", fuse_prompt, comm_df, error=e)


# Finalize
exec_log["end_time"] = datetime.now().isoformat()
exec_log["summary"] = {
    "total_nodes_db": len(history_entities_df),
    "total_edges_db": len(history_edges_df),
    "active_edges_db": len(history_edges_df[history_edges_df["invalidated"] == False]),
    "total_communities_db": len(history_communities_df),
    "virtual_cost_usd": lm.stats.virtual_usage.total_cost,
    "physical_cost_usd": lm.stats.physical_usage.total_cost,
    "cache_hits": lm.stats.cache_hits,
    "operator_cache_hits": lm.stats.operator_cache_hits,
}
with open(LOG_JSON_PATH, "w") as f:
    json.dump(exec_log, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("ZEP PIPELINE COMPLETE")
print("=" * 60)
print(f"Final Global Nodes: {len(history_entities_df)}")
print(f"Final Global Edges (Active): {len(history_edges_df[history_edges_df['invalidated'] == False])}")
print(f"Final Communities: {len(history_communities_df)}")
lm.print_total_usage()
