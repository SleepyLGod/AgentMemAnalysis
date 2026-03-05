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

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")

import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store import FaissVS

# ─────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────
LOCOMO_PATH = Path("../evermemos/evaluation/data/locomo/locomo10.json")
CONV_INDEX = 0                  
MAX_MESSAGES = 15               # Limited to avoid long runs
SLIDING_WINDOW_SIZE = 10        # Simulate Zep's Limit 10 context window retrieved per message

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_JSON_PATH = Path(f"zep_execution_log_{timestamp_str}.json")

# ─────────────────────────────────────────────────────
# Setup LOTUS
# ─────────────────────────────────────────────────────
print("=" * 60)
print("Setting up LOTUS for Zep / Graphiti Pipeline")
print("=" * 60)

lm = LM(model="deepseek/deepseek-chat", max_tokens=1000, temperature=0.0, max_batch_size=2)
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
vs = FaissVS()
lotus.settings.configure(lm=lm, rm=rm, vs=vs, enable_cache=True)
print("LOTUS cache enabled.")

# ─────────────────────────────────────────────────────
# Logging utility
# ─────────────────────────────────────────────────────
exec_log = {
    "start_time": datetime.now().isoformat(),
    "config": {"max_messages": MAX_MESSAGES, "window": SLIDING_WINDOW_SIZE},
    "operations": [],
    "summary": {}
}

def log_operation(op_type, prompt, df_in, df_out=pd.DataFrame(), error=None):
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
    exec_log["operations"].append(record)
    with open(LOG_JSON_PATH, "w") as f:
        json.dump(exec_log, f, indent=2, ensure_ascii=False)


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

print("\n" + "=" * 60)
print("Starting Zep Pipeline")
print("=" * 60)

# Zep processes EVERY single incoming message individually, but pulls the last up-to-10 messages as context.
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
            matches = recent_entities.sem_join(history_entities_df, join_prompt)
            log_operation("Q2_sem_join", join_prompt, recent_entities, matches)
            
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
        # Strict topology constraint typically applied in DB before sem_join, but we simulate full semantic
        dup_prompt = "Fact '{fact:left}' represents absolutely identical factual information as Fact '{fact:right}'"
        try:
            dup_matches = cur_edge_df.sem_join(active_hist_edges, dup_prompt)
            if len(dup_matches) > 0:
                is_dup = True
                print("      [Q5] DUP DETECTED.")
        except Exception as e:
            print(f"      Q5 Error: {e}")

        # [Q6] Contradiction Detection
        if not is_dup:
            contra_prompt = "Fact '{fact:left}' fundamentally contradicts or supersedes Fact '{fact:right}'"
            try:
                # Assuming loose topology, cross-check against all active
                contra_matches = cur_edge_df.sem_join(active_hist_edges, contra_prompt)
                if len(contra_matches) > 0:
                    is_contra = True
                    print(f"      [Q6] CONTRADICTION DETECTED. Invalidating {len(contra_matches)} old facts.")
                    # Invalidate old facts
                    for uuid_to_inv in contra_matches["uuid"].tolist():
                        history_edges_df.loc[history_edges_df["uuid"] == uuid_to_inv, "invalidated"] = True
            except Exception as e:
                print(f"      Q6 Error: {e}")

        # Insert if not dup
        if not is_dup:
            print("      Adding New Edge.")
            e_uuid = f"edge_{edge_counter}"
            edge_counter += 1
            history_edges_df.loc[len(history_edges_df)] = [e_uuid, s_name, t_name, fact_desc, False]


    # ══════════════════════════════════════════════════════════
    # PHASE 3: COMMUNITY UPDATE (SIMULATING ZEP'S N->1 RACE LOOP)
    # ══════════════════════════════════════════════════════════
    # Zep updates communities by iterating per-entity, synthesizing the community summary pairwise (no grouping)
    
    for ent in resolved_entities:
        e_uuid = ent["uuid"]
        e_name = ent["name"]
        e_summ = ent["summary_updated"]
        
        c_uuid = community_membership.get(e_uuid)
        
        if c_uuid is None:
            # Create a new mini-community for this entity
            print(f"  [Q7] Creating new Community for Node {e_name}...")
            c_uuid = f"comm_{comm_counter}"
            comm_counter += 1
            community_membership[e_uuid] = c_uuid
            # Simulate Q8 Naming later
            history_communities_df.loc[len(history_communities_df)] = [c_uuid, "TBD", e_summ]
            
        else:
            # Update existing community using the flawed Zep pairwise sem_map
            print(f"  [Q7/Q8] Updating Community for Node {e_name} (Zep N-to-1 Flaw Simulation)...")
            c_row = history_communities_df[history_communities_df["uuid"] == c_uuid].iloc[0]
            c_hist_summ = c_row["summary"]
            
            # Simulated zep pairwise merge (sem_map instead of proper sem_agg group-by)
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
