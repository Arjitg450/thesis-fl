#!/usr/bin/env python3
"""
LLM-based SHAP explainer for Keras/FL client models.

This script:
- Prints prediction immediately
- Generates LLM explanation afterward
- Avoids PFCP / non-PFCP confusion
- Produces stable, explainable outputs
"""

import argparse
import pandas as pd
import os
import requests
from pathlib import Path

# ================================
# CONFIG
# ================================

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen2.5:32b-instruct"

CLASS_MAP = {
    0: "benign",
    1: "bruteForce",
    2: "ddos_tcp_ddos",
    3: "pfcp_session_deletion",
    4: "pfcp_session_establishment",
    5: "pfcp_session_modification",
}

PFCP_CLASSES = {
    "pfcp_session_deletion",
    "pfcp_session_establishment",
    "pfcp_session_modification",
}


# ================================
# PROMPT BUILDER
# ================================

def build_prompt(shap_df, shap_summary_path, topk=10):
    """Build structured prompt for LLM with attack-aware PFCP semantics."""

    top_features = shap_df.head(topk)
    features = top_features["feature"].tolist()
    importances = top_features["shap_abs_mean"].tolist()

    prediction = "unknown"
    true_label = "unknown"
    confidence = 0.0

    try:
        folder = Path(shap_summary_path).parent
        inf_files = list(folder.glob("inference_*.csv"))
        if inf_files:
            inf_df = pd.read_csv(inf_files[0])
            if "pred" in inf_df.columns:
                prediction = int(inf_df["pred"].iloc[0])
            if "true" in inf_df.columns:
                true_label = inf_df["true"].iloc[0]
            prob_cols = [c for c in inf_df.columns if c.startswith("p_")]
            if prob_cols:
                confidence = float(inf_df[prob_cols].iloc[0].max())
    except Exception:
        pass

    pred_name = CLASS_MAP.get(prediction, str(prediction))
    true_name = CLASS_MAP.get(true_label, str(true_label))

    # IMPORTANT: PFCP is ALWAYS malicious in this threat model
    PFCP_ATTACK_CLASSES = {
        "pfcp_session_deletion",
        "pfcp_session_establishment",
        "pfcp_session_modification",
    }

    is_pfcp_attack = pred_name in PFCP_ATTACK_CLASSES
    is_correct = pred_name == true_name

    # prompt = [
    #     "You are a cybersecurity expert specializing in 5G core network attacks and explainable AI.",
    #     "",
    #     "CRITICAL CONTEXT:",
    #     "• The 5G core is compromised.",
    #     "• The SMF (Session Management Function) is malicious.",
    #     "• ANY PFCP traffic (establishment, modification, deletion) is considered an ATTACK.",
    #     "• These are NOT benign control-plane operations.",
    #     "",
    #     "Your task is to explain the model's prediction using SHAP feature attributions.",
    #     "",
    #     "STRICT RULES:",
    #     "- Treat all PFCP classes as malicious behavior.",
    #     "- Never describe PFCP traffic as normal or expected.",
    #     "- If prediction is incorrect, explain which attack class it was confused with.",
    #     "- Do NOT hallucinate protocol behavior.",
    #     "",
    #     f"Predicted class: {pred_name}",
    #     f"True class: {true_name}",
    #     "",
    #     "Evaluation:",
    #     f"- Prediction correct: {is_correct}",
    #     f"- Attack type: {'PFCP-based attack' if is_pfcp_attack else 'Non-PFCP attack'}",
    #     "",
    #     "Output format (STRICT — exactly 4 numbered sections):",
    #     "",
    #     "1. Prediction Assessment:",
    #     "   State whether the prediction correctly identifies the attack type.",
    #     "",
    #     "2. Key Contributing Factors:",
    #     "   List 3–5 SHAP features that most influenced the decision.",
    #     "",
    #     "3. Technical Interpretation:",
    #     "   Explain how the observed feature behavior reflects a malicious PFCP attack.",
    #     "",
    #     "4. Confidence Analysis:",
    #     f"   Interpret the {confidence*100:.1f}% confidence in the context of attack certainty.",
    #     "",
    #     "Feature semantics (use only when relevant):",
    #     "- fe_session_mod_req_resp_imbalance: asymmetry in PFCP request/response behavior.",
    #     "- fe_src2dst_min_over_mean_ps: abnormal control-plane packet sizing.",
    #     "- dst2src_stddev_ps: instability in UPF responses.",
    #     "- fe_dst2src_syn_rate: anomalous signaling initiation frequency.",
    #     "",
    #     "Top contributing features (ranked by SHAP importance):",
    # ]
    prompt = [
        "You are a cybersecurity expert specializing in 5G core network security, PFCP protocol analysis, and explainable AI.",
        "",
        "CRITICAL CONTEXT:",
        "• The system analyzes 5G core network traffic.",
        "• Attacks may be PFCP-based or non-PFCP-based.",
        "• PFCP operates over UDP (typically port 8805).",
        "• TCP-based attacks use standard TCP semantics.",
        "",
        "ATTACK CLASS DEFINITIONS:",
        "• PFCP attacks: pfcp_session_establishment, pfcp_session_modification, pfcp_session_deletion.",
        "• Non-PFCP attacks: bruteForce, ddos_tcp_ddos.",
        "",
        "INTERPRETATION RULES (CRITICAL):",
        "1. If the predicted class is a PFCP attack:",
        "   - Treat all PFCP traffic as malicious.",
        "   - Interpret PFCP features as control-plane abuse.",
        "   - DO NOT apply TCP semantics.",
        "   - TCP-named features represent abstract signaling indicators only.",
        "",
        "2. If the predicted class is a NON-PFCP attack:",
        "   - TCP semantics (SYN, ACK, FIN, PSH) MAY be used.",
        "   - PFCP-related features should not be emphasized.",
        "",
        "3. NEVER mix PFCP semantics with TCP semantics in the same explanation.",
        "",
        "FEATURE FAMILY DEFINITIONS:",
        "• pfcp_behavior:",
        "  - Direct indicators of PFCP message activity.",
        "  - Includes session establishment, modification, and deletion.",
        "",
        "• flow_shape:",
        "  - Packet size consistency or irregularity.",
        "  - Abnormal values suggest malformed or artificial traffic.",
        "",
        "• burstiness:",
        "  - Measures timing irregularities and traffic bursts.",
        "  - Indicates automated or attack-driven behavior.",
        "",
        "• directionality:",
        "  - Measures imbalance between source and destination traffic.",
        "",
        "• tcp_semantics:",
        "  - Used ONLY for non-PFCP attacks.",
        "  - Represents TCP connection behavior.",
        "",
        "PREDICTION CONTEXT:",
        f"Predicted class: {pred_name}",
        f"True class: {true_name}",
        "",
        "Evaluation:",
        f"- Prediction correct: {is_correct}",
        f"- Attack family: {'PFCP' if is_pfcp_attack else 'Non-PFCP'}",
        "",
        "OUTPUT FORMAT (STRICT — EXACTLY 4 SECTIONS):",
        "",
        "1. Prediction Assessment:",
        "   State whether the prediction correctly identifies the attack type.",
        "",
        "2. Key Contributing Factors:",
        "   List 3–5 SHAP features and briefly explain their influence.",
        "",
        "3. Technical Interpretation:",
        "   Explain how the observed feature behavior reflects the predicted attack.",
        "   - Use PFCP semantics ONLY for PFCP attacks.",
        "   - Use TCP semantics ONLY for non-PFCP attacks.",
        "",
        "4. Confidence Analysis:",
        f"   Interpret the {confidence*100:.1f}% confidence realistically (not as certainty).",
        "",
        "REFERENCE FEATURE INTERPRETATION:",
        "",
        "PFCP FEATURES:",
        "- fe_session_est_req_per_packet: excessive PFCP session establishment attempts.",
        "- fe_session_mod_req_resp_imbalance: imbalance in modification requests and responses.",
        "- fe_session_del_req_per_packet: frequent session teardown behavior.",
        "",
        "NON-PFCP (TCP) FEATURES:",
        "- fe_*_syn_rate: connection initiation intensity.",
        "- fe_*_ack_rate: acknowledgment behavior.",
        "- fe_*_rst_rate: abnormal connection resets.",
        "- fe_*_psh_rate: aggressive data pushing.",
        "",
        "FLOW FEATURES:",
        "- fe_*_piat_*: burst timing irregularities.",
        "- fe_*_min_over_mean_ps: packet size irregularity.",
        "",
        "FINAL RULE:",
        "Your explanation must be protocol-aware, attack-specific, and suitable for academic or security review.",
        "Do NOT hallucinate protocol behavior.",
        "",
        "Top contributing features (ranked by SHAP importance):",
    ]


    for f, imp in zip(features, importances):
        prompt.append(f"- {f}: importance {imp:.6f}")

    return "\n".join(prompt)


# ================================
# OLLAMA CALL
# ================================

def query_ollama(prompt, model=DEFAULT_OLLAMA_MODEL, ollama_url=DEFAULT_OLLAMA_URL):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_k": 40,
            "top_p": 0.85
        },
    }

    try:
        r = requests.post(ollama_url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"Error contacting Ollama: {e}"


# ================================
# MAIN
# ================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap-summary", required=True)
    ap.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    shap_df = pd.read_csv(args.shap_summary)

    # -------------------------------
    # PRINT PREDICTION FIRST
    # -------------------------------
    prediction = "unknown"
    true_label = "unknown"

    try:
        folder = Path(args.shap_summary).parent
        inf_files = list(folder.glob("inference_*.csv"))
        if inf_files:
            inf_df = pd.read_csv(inf_files[0])
            if "pred" in inf_df.columns:
                prediction = int(inf_df["pred"].iloc[0])
            if "true" in inf_df.columns:
                true_label = inf_df["true"].iloc[0]
    except Exception:
        pass

    pred_name = CLASS_MAP.get(prediction, str(prediction))
    true_name = CLASS_MAP.get(true_label, str(true_label))

    print(f"\nPrediction: {prediction} ({pred_name})")
    print(f"True: {true_name}\n")

    # -------------------------------
    # LLM EXPLANATION
    # -------------------------------
    print("Generating LLM explanation...\n", flush=True)

    prompt = build_prompt(
        shap_df,
        shap_summary_path=args.shap_summary,
        topk=args.topk,
    )

    explanation = query_ollama(
        prompt,
        model=args.ollama_model,
        ollama_url=args.ollama_url,
    )

    print("LLM Explanation:\n")
    print(explanation)

    out_path = os.path.splitext(args.shap_summary)[0] + "_llm_explanation.txt"
    with open(out_path, "w") as f:
        f.write(explanation)

    print(f"\nExplanation saved to: {out_path}")


if __name__ == "__main__":
    main()
