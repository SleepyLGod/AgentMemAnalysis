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
history_relations_df = pd.DataFrame(columns=["rel_id", "src_entity", "relation", "dst_entity"])

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
                selected_history = cf_df.sem_sim_join(history_facts_df, left_on="cur_fact", right_on="fact", K=1)
                log_operation("Q2_sem_sim_join", "left_on=cur_fact, right_on=fact, K=1", cf_df, selected_history)
                
                print("  [Q3+Q4] LLM Fact Resolution...")
                # Apply Proxy Optimization: check embedding similarity to skip LLM if > threshold
                for _, row in selected_history.iterrows():
                    cf_val = str(row.get("cur_fact"))
                    hf_val = str(row.get("fact"))
                    if not pd.isna(hf_val) and hf_val:
                        # Compute similarity manually for the proxy check
                        emb_c = rm([cf_val])[0]
                        emb_h = rm([hf_val])[0]
                        sim = float(np.dot(emb_c, emb_h) / (np.linalg.norm(emb_c) * np.linalg.norm(emb_h) + 1e-9))
                        
                        if sim >= SIM_NOOP_THRESHOLD:
                            print(f"    PROXY-OPTIMIZED NOOP (Sim {sim:.3f}): {cf_val}")
                            continue
                            
                        # If not skipped by proxy, call LLM
                        pair_df = pd.DataFrame({"current_fact": [cf_val], "history_fact": [hf_val]})
                        prompt = (
                            "Given new fact '{current_fact}' and existing fact '{history_fact}', "
                            "return action (ADD/UPDATE/DELETE/NOOP). "
                            "If contradictory: DELETE. If adding details: UPDATE. If redundant: NOOP. Else: ADD."
                        )
                        try:
                            # Use sem_map to just get the action
                            res_df = pair_df.sem_map(prompt)
                            log_operation("Q3+Q4_sem_map", prompt, pair_df, res_df)
                            action = str(res_df["_map"].iloc[0]).strip().upper()
                            
                            print(f"    LLM ACTION ({action}) for fact: {cf_val}")
                            
                            # Execute Action (Simplified state management)
                            if "UPDATE" in action or "ADD" in action:
                                history_facts_df.loc[len(history_facts_df)] = [f"f_{fact_counter}", cf_val]
                                fact_counter += 1
                            elif "DELETE" in action:
                                # Mock delete
                                pass
                        except Exception as e:
                            print(f"    Error in LLM resolution: {e}")
                            log_operation("Q3_Q4_sem_map", prompt, pair_df, error=e)
                    else:
                        # No history match found, just ADD
                        print(f"    AUTO-ADD (No History Match): {cf_val}")
                        history_facts_df.loc[len(history_facts_df)] = [f"f_{fact_counter}", cf_val]
                        fact_counter += 1
                        
            except Exception as e:
                print(f"    Error in Q2 Join: {e}")
                log_operation("Q2_sem_sim_join", "left_on=cur_fact, right_on=fact", cf_df, error=e)


    # ================= ENTITY ALIGNMENT (Q7, Q8) =================
    if len(ce_list) > 0:
        print("  [Q7] Vector Recall for Entities...")
        ce_df = pd.DataFrame(ce_list) # columns: entity_name, entity_type
        
        if len(history_entities_df) == 0:
            print("    HISTORY EMPTY. Adding entities auto.")
            for e in ce_list:
                history_entities_df.loc[len(history_entities_df)] = [f"e_{entity_counter}", e["entity_name"], e["entity_type"]]
                entity_counter += 1
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
                        "hist_ent": valid_candidates["entity_name_right"]
                    })
                    
                    filter_prompt = "{cur_ent} and {hist_ent} refer to the exact same real-world entity."
                    try:
                        aligned = filter_df.sem_filter(filter_prompt)
                        log_operation("Q8_sem_filter", filter_prompt, filter_df, aligned)
                        
                        # Any that survived the filter are 'SAME', we NOOP them.
                        # Those that didn't pass, we ADD.
                        matched_curents = aligned["cur_ent"].tolist() if "cur_ent" in aligned.columns else []
                        
                        for cur_ent in filter_df["cur_ent"].tolist():
                            if cur_ent in matched_curents:
                                print(f"    LLM: {cur_ent} is SAME as history.")
                            else:
                                print(f"    LLM: {cur_ent} is DIFFERENT. Adding.")
                                # We'd add to DB
                                history_entities_df.loc[len(history_entities_df)] = [f"e_{entity_counter}", cur_ent, "unknown"]
                                entity_counter += 1
                    except Exception as e:
                        print(f"    Error in Q8 Filter: {e}")
                        log_operation("Q8_sem_filter", filter_prompt, filter_df, error=e)
                else:
                    print("    No valid entity candidates to filter. Auto-adding.")
            except Exception as e:
                print(f"    Error in Q7 Join: {e}")
                log_operation("Q7_sem_sim_join", "K=1 entity recall", ce_df, error=e)

    # ================= RELATION RESOLUTION (Q9) =================
    if len(cr_list) > 0:
        print("  [Q9] Vector Recall + LLM Resolution for Relations...")
        # Convert current relations
        cr_df = pd.DataFrame(cr_list) 
        
        if len(history_relations_df) == 0:
            print("    HISTORY EMPTY. Adding relations auto.")
            for rel in cr_list:
                history_relations_df.loc[len(history_relations_df)] = [f"r_{rel_counter}", rel["src_entity"], rel["relation"], rel["dst_entity"]]
                rel_counter += 1
        else:
            try:
                history_relations_df = history_relations_df.sem_index("relation", "mem0_rel_idx")
                rel_candidates = cr_df.sem_sim_join(history_relations_df, left_on="relation", right_on="relation", K=1, lsuffix="_left", rsuffix="_right")
                log_operation("Q9_sem_sim_join", "left_on=relation, right_on=relation, K=1", cr_df, rel_candidates)
                
                print("  [Q9] LLM Relation Resolution (sem_map)...")
                for _, row in rel_candidates.iterrows():
                    cr_val = str(row.get("relation_left")) if "relation_left" in row else str(row.get("cur_rel_desc", ""))
                    # LOTUS without suffix might just keep 'relation' from right table if 'relation_left' not specified, 
                    # fallback to stringing the SRC-REL-DST
                    if not cr_val:
                        cr_val = f"{row.get('src_entity_left')} -> {row.get('relation')} -> {row.get('dst_entity_left')}"
                    
                    hr_val = str(row.get("relation_right")) if "relation_right" in row else str(row.get("relation", ""))
                    
                    if not pd.isna(hr_val) and hr_val:
                        rel_pair_df = pd.DataFrame({"current_relation": [cr_val], "history_relation": [hr_val]})
                        rel_prompt = (
                            "Given new relation '{current_relation}' and existing relation '{history_relation}', "
                            "Does the new relation contradict the old one (CONTRADICTS), add detail (AUGMENTS), or is it unrelated (NEW)? "
                            "Return exactly one word: CONTRADICTS, AUGMENTS, or NEW."
                        )
                        try:
                            # Use sem_map to get action
                            rel_res = rel_pair_df.sem_map(rel_prompt)
                            log_operation("Q9_sem_map", rel_prompt, rel_pair_df, rel_res)
                            rel_action = str(rel_res["_map"].iloc[0]).strip().upper()
                            
                            print(f"    LLM RELATION ACTION ({rel_action}) for: {cr_val}")
                            
                            # Execute Action
                            if "AUGMENTS" in rel_action or "NEW" in rel_action:
                                src_ent = row.get("src_entity_left", "Unknown")
                                dst_ent = row.get("dst_entity_left", "Unknown")
                                history_relations_df.loc[len(history_relations_df)] = [f"r_{rel_counter}", src_ent, cr_val, dst_ent]
                                rel_counter += 1
                        except Exception as e:
                            print(f"    Error in LLM relation resolution: {e}")
                            log_operation("Q9_sem_map", rel_prompt, rel_pair_df, error=e)
                    else:
                        print(f"    AUTO-ADD RELATION (No Match): {cr_val}")
                        history_relations_df.loc[len(history_relations_df)] = [f"r_{rel_counter}", row.get("src_entity_left"), cr_val, row.get("dst_entity_left")]
                        rel_counter += 1
            except Exception as e:
                print(f"    Error in Q9 Relation Join: {e}")
                log_operation("Q9_sem_sim_join", "relation sim join", cr_df, error=e)

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
