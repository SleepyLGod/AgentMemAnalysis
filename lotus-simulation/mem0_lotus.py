"""
Mem0 Semantic Queries via LOTUS
=====================================
Executes Mem0 memory graph insertion queries using LOTUS operators on LOCOMO data.
Simulates:
- Q1/Q5/Q6: Fact, Entity, and Relation Extraction
- Q2/Q7: Historical similarity search using vector representations
- Q3+Q4, Q8, Q9: Resolution via semantic maps/filters

Applies LOTUS optimization concepts (e.g. Join sim-filter proxy) to reduce LLM calls.

Usage:
    source .venv/bin/activate
    python mem0_lotus.py
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

# Proxy Optimization Thresholds (Simulates LLM offloading)
SIM_NOOP_THRESHOLD = 0.92       # If cosine sim >= 0.92, auto-assume NOOP (Exact match)

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_JSON_PATH = Path(f"mem0_execution_log_{timestamp_str}.json")

# ─────────────────────────────────────────────────────
# Setup LOTUS
# ─────────────────────────────────────────────────────
print("=" * 60)
print("Setting up LOTUS for Mem0 Pipeline")
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
    "config": {"max_messages": MAX_MESSAGES},
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
# Global Memory DB (Simulates Mem0 Vector Stores)
# ─────────────────────────────────────────────────────
# To use LOTUS sem_sim_join, we need DataFrames. We will maintain these organically.
history_facts_df = pd.DataFrame(columns=["fact_id", "fact"])
history_entities_df = pd.DataFrame(columns=["entity_id", "entity_name", "entity_type"])
history_relations_df = pd.DataFrame(
    columns=["rel_id", "src_entity_id", "src_entity", "relation", "dst_entity_id", "dst_entity"]
)

fact_counter = 0
entity_counter = 0
rel_counter = 0

print("\n" + "=" * 60)
print("Starting Mem0 Pipeline")
print("=" * 60)

for msg_idx, msg in enumerate(messages):
    text_content = f"{msg['speaker']}: {msg['text']}"
    print(f"\n--- Mem0 Message {msg_idx + 1}/{len(messages)}: {text_content[:60]}...")
    
    msg_df = pd.DataFrame({"text": [text_content]})

    # ══════════════════════════════════════════════════════════
    # PHASE 1: EXTRACTION
    # ══════════════════════════════════════════════════════════

    # [Q1] Fact Extraction
    print("  [Q1] Extracting Facts...")
    fact_cols = {"fact": "a standalone, self-contained fact from the conversation statement"}
    try:
        current_facts = msg_df.sem_extract(input_cols=["text"], output_cols=fact_cols)
        log_operation("Q1_sem_extract", fact_cols, msg_df, current_facts)
        # Filter valid extractions
        cf_list = []
        if "fact" in current_facts.columns:
            cf_list = [str(r["fact"]) for _, r in current_facts.iterrows() if str(r.get("fact", "")) not in ("", "N/A", "None")]
    except Exception as e:
        print(f"    Error: {e}")
        log_operation("Q1_sem_extract", fact_cols, msg_df, error=e)
        cf_list = []
        current_facts = pd.DataFrame()

    for f in cf_list: print(f"    → {f}")

    # [Q5] Entity Extraction
    print("  [Q5] Extracting Entities...")
    ent_cols = {"entity_name": "entity name", "entity_type": "entity type"}
    try:
        current_entities = msg_df.sem_extract(input_cols=["text"], output_cols=ent_cols)
        log_operation("Q5_sem_extract", ent_cols, msg_df, current_entities)
        
        ce_list = []
        if "entity_name" in current_entities.columns:
            for _, r in current_entities.iterrows():
                if str(r.get("entity_name", "")) not in ("", "N/A", "None"):
                    ce_list.append({"entity_name": str(r["entity_name"]), "entity_type": str(r.get("entity_type", ""))})
    except Exception as e:
        print(f"    Error: {e}")
        log_operation("Q5_sem_extract", ent_cols, msg_df, error=e)
        ce_list = []
        current_entities = pd.DataFrame()
        
    for e in ce_list: print(f"    → {e['entity_name']} ({e['entity_type']})")

    # [Q6] Relation Extraction (requires entities)
    cr_list = []
    if len(ce_list) > 0:
        print("  [Q6] Extracting Relations...")
        # Convert entities list to string to pass as context
        ent_names = ", ".join([e["entity_name"] for e in ce_list])
        rel_input_df = pd.DataFrame({"text": [text_content], "entity_name": [ent_names]})
        rel_cols = {
            "src_entity_name": "source entity",
            "relation_description": "relationship",
            "dest_entity_name": "destination entity"
        }
        try:
            current_relations = rel_input_df.sem_extract(input_cols=["text", "entity_name"], output_cols=rel_cols)
            log_operation("Q6_sem_extract", rel_cols, rel_input_df, current_relations)
            
            if "relation_description" in current_relations.columns:
                for _, r in current_relations.iterrows():
                    if str(r.get("relation_description", "")) not in ("", "N/A", "None"):
                        cr_list.append({
                            "src_entity": str(r.get("src_entity_name", "")),
                            "relation": str(r["relation_description"]),
                            "dst_entity": str(r.get("dest_entity_name", ""))
                        })
        except Exception as e:
            print(f"    Error: {e}")
            log_operation("Q6_sem_extract", rel_cols, rel_input_df, error=e)

        for rel in cr_list: print(f"    → {rel['src_entity']} -[{rel['relation']}]-> {rel['dst_entity']}")

    # ══════════════════════════════════════════════════════════
    # PHASE 2: RESOLUTION / SEMANTIC JOIN
    # ══════════════════════════════════════════════════════════

    # ================= FACT RESOLUTION (Q2, Q3+Q4) =================
    if len(cf_list) > 0:
        print("  [Q2] Vector Recall for Facts...")
        # Convert current facts to DF format required by LOTUS
        cf_df = pd.DataFrame({"cur_fact": cf_list})
        
        if len(history_facts_df) == 0:
            print("    HISTORY EMPTY. Adding facts auto.")
            for f in cf_list:
                history_facts_df.loc[len(history_facts_df)] = [f"f_{fact_counter}", f]
                fact_counter += 1
        else:
            # We must set index on history to search. Since history grows, we recreate the index.
            try:
                # Need to use sem_sim_join over the history facts.
                # Format: df.sem_sim_join(other_df, left_on, right_on)
                history_facts_df = history_facts_df.sem_index("fact", "mem0_fact_idx")
                selected_history = cf_df.sem_sim_join(
                    history_facts_df,
                    left_on="cur_fact",
                    right_on="fact",
                    K=5,
                )
                log_operation("Q2_sem_sim_join", "left_on=cur_fact, right_on=fact, K=5", cf_df, selected_history)
                
                print("  [Q3+Q4] LLM Fact Resolution...")
                # Evaluate each new fact against up-to-K recalled candidates.
                for cf_val in cf_list:
                    candidate_rows = (
                        selected_history[selected_history["cur_fact"] == cf_val]
                        if "cur_fact" in selected_history.columns
                        else pd.DataFrame()
                    )

                    if len(candidate_rows) == 0:
                        print(f"    AUTO-ADD (No History Match): {cf_val}")
                        history_facts_df.loc[len(history_facts_df)] = [f"f_{fact_counter}", cf_val]
                        fact_counter += 1
                        continue

                    final_action = "ADD"
                    target_fact_id = None
                    merged_fact = cf_val

                    for _, row in candidate_rows.iterrows():
                        hf_val = str(row.get("fact", row.get("fact_right", "")))
                        hf_id = row.get("fact_id", row.get("fact_id_right"))
                        hf_id = None if pd.isna(hf_id) else str(hf_id)
                        if not hf_val or hf_val.lower() in ("none", "nan"):
                            continue

                        # Proxy Optimization: exact-ish semantic match -> NOOP.
                        emb_c = rm([cf_val])[0]
                        emb_h = rm([hf_val])[0]
                        sim = float(
                            np.dot(emb_c, emb_h)
                            / (np.linalg.norm(emb_c) * np.linalg.norm(emb_h) + 1e-9)
                        )
                        if sim >= SIM_NOOP_THRESHOLD:
                            final_action = "NOOP"
                            print(f"    PROXY-OPTIMIZED NOOP (Sim {sim:.3f}): {cf_val}")
                            break

                        pair_df = pd.DataFrame({"current_fact": [cf_val], "history_fact": [hf_val]})
                        prompt = (
                            "Given new fact '{current_fact}' and existing fact '{history_fact}', "
                            "return action (ADD/UPDATE/DELETE/NOOP). "
                            "If contradictory: DELETE. If adding details: UPDATE. If redundant: NOOP. Else: ADD."
                        )
                        try:
                            res_df = pair_df.sem_map(prompt)
                            log_operation("Q3+Q4_sem_map", prompt, pair_df, res_df)
                            action_raw = str(res_df["_map"].iloc[0]).strip().upper()
                            print(f"    LLM ACTION ({action_raw}) for fact: {cf_val}")

                            if "DELETE" in action_raw:
                                final_action = "DELETE"
                                target_fact_id = hf_id
                                break
                            if "UPDATE" in action_raw:
                                final_action = "UPDATE"
                                target_fact_id = hf_id
                                merge_prompt = (
                                    "Merge '{history_fact}' and '{current_fact}' into one comprehensive, "
                                    "non-duplicated fact sentence."
                                )
                                merge_res = pair_df.sem_map(merge_prompt)
                                log_operation("Q4_sem_map_merge", merge_prompt, pair_df, merge_res)
                                merged_fact = str(merge_res["_map"].iloc[0]).strip() or cf_val
                                break
                            if "NOOP" in action_raw:
                                final_action = "NOOP"
                                break
                            # ADD keeps scanning for stronger evidence (UPDATE/DELETE/NOOP).
                        except Exception as e:
                            print(f"    Error in LLM resolution: {e}")
                            log_operation("Q3+Q4_sem_map", prompt, pair_df, error=e)

                    if final_action == "NOOP":
                        continue

                    if final_action == "UPDATE" and target_fact_id:
                        history_facts_df.loc[
                            history_facts_df["fact_id"] == target_fact_id, "fact"
                        ] = merged_fact
                        print(f"    UPDATED fact_id={target_fact_id}")
                        continue

                    if final_action == "DELETE" and target_fact_id:
                        history_facts_df = history_facts_df[
                            history_facts_df["fact_id"] != target_fact_id
                        ].reset_index(drop=True)
                        print(f"    DELETED fact_id={target_fact_id}")
                        # Mem0 keeps the new contradictory fact as the replacement memory.
                        history_facts_df.loc[len(history_facts_df)] = [f"f_{fact_counter}", cf_val]
                        fact_counter += 1
                        continue

                    # Default and explicit ADD.
                    history_facts_df.loc[len(history_facts_df)] = [f"f_{fact_counter}", cf_val]
                    fact_counter += 1
                        
            except Exception as e:
                print(f"    Error in Q2 Join: {e}")
                log_operation("Q2_sem_sim_join", "left_on=cur_fact, right_on=fact", cf_df, error=e)


    # ================= ENTITY ALIGNMENT (Q7, Q8) =================
    resolved_entity_ids = {}
    if len(ce_list) > 0:
        print("  [Q7] Vector Recall for Entities...")
        ce_df = pd.DataFrame(ce_list) # columns: entity_name, entity_type
        
        if len(history_entities_df) == 0:
            print("    HISTORY EMPTY. Adding entities auto.")
            for e in ce_list:
                e_id = f"e_{entity_counter}"
                history_entities_df.loc[len(history_entities_df)] = [
                    e_id,
                    e["entity_name"],
                    e["entity_type"],
                ]
                entity_counter += 1
                resolved_entity_ids[e["entity_name"]] = e_id
        else:
            try:
                history_entities_df = history_entities_df.sem_index("entity_name", "mem0_ent_idx")
                # Top 1 recall
                candidates = ce_df.sem_sim_join(history_entities_df, left_on="entity_name", right_on="entity_name", K=1, lsuffix="_left", rsuffix="_right")
                log_operation("Q7_sem_sim_join", "K=1 entity recall", ce_df, candidates)
                
                print("  [Q8] LLM Entity Alignment (sem_filter)...")
                # Drop NAs from join to avoid LLM errors
                valid_candidates = candidates.dropna(subset=["entity_name_right"])
                
                if len(valid_candidates) > 0:
                    # Rename columns strictly for the langex template 
                    filter_df = pd.DataFrame({
                        "cur_ent": valid_candidates["entity_name_left"],
                        "hist_ent": valid_candidates["entity_name_right"],
                        "hist_id": valid_candidates["entity_id"],
                    })
                    
                    filter_prompt = "{cur_ent} and {hist_ent} refer to the exact same real-world entity."
                    try:
                        aligned = filter_df.sem_filter(filter_prompt)
                        log_operation("Q8_sem_filter", filter_prompt, filter_df, aligned)
                        
                        # Any that survived the filter are 'SAME', others are added as new entities.
                        matched_map = {}
                        if "cur_ent" in aligned.columns and "hist_id" in aligned.columns:
                            for _, arow in aligned.iterrows():
                                cur_ent = str(arow.get("cur_ent", ""))
                                hist_id = str(arow.get("hist_id", ""))
                                if cur_ent and hist_id:
                                    matched_map[cur_ent] = hist_id

                        for e in ce_list:
                            cur_ent = e["entity_name"]
                            cur_type = e.get("entity_type", "unknown")
                            if cur_ent in resolved_entity_ids:
                                continue

                            if cur_ent in matched_map:
                                resolved_entity_ids[cur_ent] = matched_map[cur_ent]
                                print(f"    LLM: {cur_ent} is SAME as history.")
                                continue

                            exact_rows = history_entities_df[history_entities_df["entity_name"] == cur_ent]
                            if len(exact_rows) > 0:
                                resolved_entity_ids[cur_ent] = str(exact_rows.iloc[0]["entity_id"])
                                continue

                            print(f"    LLM: {cur_ent} is DIFFERENT. Adding.")
                            new_id = f"e_{entity_counter}"
                            history_entities_df.loc[len(history_entities_df)] = [
                                new_id,
                                cur_ent,
                                cur_type,
                            ]
                            entity_counter += 1
                            resolved_entity_ids[cur_ent] = new_id
                    except Exception as e:
                        print(f"    Error in Q8 Filter: {e}")
                        log_operation("Q8_sem_filter", filter_prompt, filter_df, error=e)
                else:
                    print("    No valid entity candidates to filter. Auto-adding.")
                    for e in ce_list:
                        cur_ent = e["entity_name"]
                        cur_type = e.get("entity_type", "unknown")
                        if cur_ent in resolved_entity_ids:
                            continue
                        exact_rows = history_entities_df[history_entities_df["entity_name"] == cur_ent]
                        if len(exact_rows) > 0:
                            resolved_entity_ids[cur_ent] = str(exact_rows.iloc[0]["entity_id"])
                            continue
                        new_id = f"e_{entity_counter}"
                        history_entities_df.loc[len(history_entities_df)] = [
                            new_id,
                            cur_ent,
                            cur_type,
                        ]
                        entity_counter += 1
                        resolved_entity_ids[cur_ent] = new_id
            except Exception as e:
                print(f"    Error in Q7 Join: {e}")
                log_operation("Q7_sem_sim_join", "K=1 entity recall", ce_df, error=e)

    # ================= RELATION RESOLUTION (Q9) =================
    if len(cr_list) > 0:
        print("  [Q9] 1-hop Candidate Recall + LLM Relation Resolution...")
        for rel in cr_list:
            src_name = str(rel.get("src_entity", ""))
            dst_name = str(rel.get("dst_entity", ""))
            rel_text = str(rel.get("relation", ""))
            if not src_name or not dst_name or not rel_text:
                continue

            src_id = resolved_entity_ids.get(src_name)
            if src_id is None:
                src_rows = history_entities_df[history_entities_df["entity_name"] == src_name]
                if len(src_rows) > 0:
                    src_id = str(src_rows.iloc[0]["entity_id"])
                else:
                    src_id = f"e_{entity_counter}"
                    history_entities_df.loc[len(history_entities_df)] = [src_id, src_name, "unknown"]
                    entity_counter += 1
                resolved_entity_ids[src_name] = src_id

            dst_id = resolved_entity_ids.get(dst_name)
            if dst_id is None:
                dst_rows = history_entities_df[history_entities_df["entity_name"] == dst_name]
                if len(dst_rows) > 0:
                    dst_id = str(dst_rows.iloc[0]["entity_id"])
                else:
                    dst_id = f"e_{entity_counter}"
                    history_entities_df.loc[len(history_entities_df)] = [dst_id, dst_name, "unknown"]
                    entity_counter += 1
                resolved_entity_ids[dst_name] = dst_id

            # Mem0 graph flow: relation candidates are 1-hop relations around src/dst entities.
            rel_candidates = history_relations_df[
                (history_relations_df["src_entity_id"] == src_id)
                | (history_relations_df["dst_entity_id"] == dst_id)
            ]
            candidate_log_in = pd.DataFrame(
                {
                    "src_entity_id": [src_id],
                    "dst_entity_id": [dst_id],
                    "relation": [rel_text],
                }
            )
            log_operation("Q9_candidate_filter", "1-hop candidates by src_id OR dst_id", candidate_log_in, rel_candidates)

            if len(rel_candidates) == 0:
                print(f"    AUTO-ADD RELATION (No 1-hop candidate): {src_name} -[{rel_text}]-> {dst_name}")
                history_relations_df.loc[len(history_relations_df)] = [
                    f"r_{rel_counter}",
                    src_id,
                    src_name,
                    rel_text,
                    dst_id,
                    dst_name,
                ]
                rel_counter += 1
                continue

            final_action = "NEW"
            target_rel_id = None

            for _, crow in rel_candidates.iterrows():
                hist_rel = str(crow.get("relation", ""))
                rel_pair_df = pd.DataFrame(
                    {"current_relation": [rel_text], "history_relation": [hist_rel]}
                )
                rel_prompt = (
                    "Given new relation '{current_relation}' and existing relation '{history_relation}', "
                    "Does the new relation contradict the old one (CONTRADICTS), add detail (AUGMENTS), or is it unrelated (NEW)? "
                    "Return exactly one word: CONTRADICTS, AUGMENTS, or NEW."
                )
                try:
                    rel_res = rel_pair_df.sem_map(rel_prompt)
                    log_operation("Q9_sem_map", rel_prompt, rel_pair_df, rel_res)
                    rel_action = str(rel_res["_map"].iloc[0]).strip().upper()
                    print(f"    LLM RELATION ACTION ({rel_action}) for: {src_name} -[{rel_text}]-> {dst_name}")

                    if "CONTRADICT" in rel_action:
                        final_action = "CONTRADICTS"
                        target_rel_id = str(crow.get("rel_id", ""))
                        break
                    if "AUGMENT" in rel_action:
                        final_action = "AUGMENTS"
                        target_rel_id = str(crow.get("rel_id", ""))
                        break
                except Exception as e:
                    print(f"    Error in LLM relation resolution: {e}")
                    log_operation("Q9_sem_map", rel_prompt, rel_pair_df, error=e)

            if final_action == "CONTRADICTS" and target_rel_id:
                # Delete contradicted old relation, then add new one.
                history_relations_df = history_relations_df[
                    history_relations_df["rel_id"] != target_rel_id
                ].reset_index(drop=True)
                history_relations_df.loc[len(history_relations_df)] = [
                    f"r_{rel_counter}",
                    src_id,
                    src_name,
                    rel_text,
                    dst_id,
                    dst_name,
                ]
                rel_counter += 1
                continue

            if final_action == "AUGMENTS" and target_rel_id:
                history_relations_df.loc[
                    history_relations_df["rel_id"] == target_rel_id,
                    ["src_entity_id", "src_entity", "relation", "dst_entity_id", "dst_entity"],
                ] = [src_id, src_name, rel_text, dst_id, dst_name]
                continue

            # NEW
            history_relations_df.loc[len(history_relations_df)] = [
                f"r_{rel_counter}",
                src_id,
                src_name,
                rel_text,
                dst_id,
                dst_name,
            ]
            rel_counter += 1

# Finalize
exec_log["end_time"] = datetime.now().isoformat()
exec_log["summary"] = {
    "total_facts_db": len(history_facts_df),
    "total_entities_db": len(history_entities_df),
    "virtual_cost_usd": lm.stats.virtual_usage.total_cost,
    "physical_cost_usd": lm.stats.physical_usage.total_cost,
    "cache_hits": lm.stats.cache_hits,
}
with open(LOG_JSON_PATH, "w") as f:
    json.dump(exec_log, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("MEM0 PIPELINE COMPLETE")
print("=" * 60)
print(f"Final Global Facts: {len(history_facts_df)}")
print(f"Final Global Entities: {len(history_entities_df)}")
lm.print_total_usage()
