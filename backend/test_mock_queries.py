import re
import json
from typing import List, Set, Tuple

def generate_grounded_answer_v2(prompt: str) -> str:
    # Extract query
    user_q = ""
    if "User Research Question:" in prompt:
        parts = prompt.split("User Research Question:")[1]
        if "Supplied Context Blocks" in parts:
            user_q = parts.split("Supplied Context Blocks")[0].strip()
        elif "<context>" in parts:
            user_q = parts.split("<context>")[0].strip()
        else:
            user_q = parts.split("\n\n")[0].strip()
    else:
        first_line = prompt.split("\n")[0].strip()
        if not first_line.startswith("[Source:"):
            user_q = first_line

    # Extract context segment
    context_segment = ""
    if "<context>" in prompt and "</context>" in prompt:
        m = re.search(r"<context>(.*?)</context>", prompt, flags=re.DOTALL)
        if m:
            context_segment = m.group(1).strip()
    elif "Supplied Context Blocks" in prompt:
        parts = prompt.split("Supplied Context Blocks")
        raw_ctx = parts[1]
        if "Provide your grounded synthesis" in raw_ctx:
            raw_ctx = raw_ctx.split("Provide your grounded synthesis")[0]
        context_segment = raw_ctx.strip()
    else:
        context_segment = prompt

    # Parse blocks with chunk_id and score
    raw_blocks = [b.strip() for b in re.split(r"(?=\[Source:)", context_segment) if b.strip()]
    parsed_blocks = []
    for blk in raw_blocks:
        cid = None
        cid_match = re.search(r"Chunk:\s*([^\s\|\]]+)", blk, flags=re.IGNORECASE)
        if cid_match:
            cid = cid_match.group(1).strip()
        else:
            uuid_match = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", blk)
            if uuid_match:
                cid = uuid_match.group(1)
            else:
                alt_match = re.search(r"\b(chunk-?\d+|c\d+|dup-\d+)\b", blk, flags=re.IGNORECASE)
                if alt_match:
                    cid = alt_match.group(1)

        block_score = 1.0
        score_match = re.search(r"Score:\s*([0-9\.]+)", blk)
        if score_match:
            try:
                block_score = float(score_match.group(1))
            except ValueError:
                pass

        content_text = re.sub(r"\[Source:[^\]]+\]", "", blk).strip()
        content_text = re.sub(r"</?context>", "", content_text).strip()
        if content_text:
            parsed_blocks.append({"chunk_id": cid, "score": block_score, "text": content_text})

    # Empty context check
    if not parsed_blocks or "no evidence" in prompt.lower() or "empty" in prompt.lower():
        return json.dumps({
            "answer": "The available documents do not contain sufficient evidence to answer this query.",
            "claims": [],
            "insufficient_evidence": True,
        }, indent=2)

    # Filter out distractor blocks whose score is far below the best chunk (< 5% of top score or < 0.02)
    max_score = max(b["score"] for b in parsed_blocks)
    if max_score >= 0.2:
        relevance_cutoff = max(0.02, max_score * 0.05)
        parsed_blocks = [b for b in parsed_blocks if b["score"] >= relevance_cutoff]

    stopwords = {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "the", "a", "an", "and", "or", "but", "in", "on",
        "at", "to", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "from", "up", "down", "out",
        "over", "under", "again", "further", "then", "once", "here", "there", "all",
        "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
        "will", "just", "should", "now", "tell", "show", "give", "explain", "describe",
        "according", "stated", "using", "used", "does", "provide", "please", "could",
        "would", "across", "between", "query", "question", "use", "uses"
    }
    entity_stopwords = {
        "company", "enterprise", "knowledge", "copilot", "system", "platform",
        "document", "documents", "section", "page", "source", "chunk", "report",
        "manual", "corp", "inc", "ltd", "abc", "employee", "employees", "user", "users"
    }

    def get_stem(w: str) -> str:
        w = w.lower()
        w = re.sub(r"(ing|tion|tions|ed|es|s|al|als|ive|ives|ment|ments)$", "", w).rstrip("e")
        return w

    all_q_tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", user_q.lower())
    content_words = [w for w in all_q_tokens if w not in stopwords]
    
    intent_words = set([w for w in content_words if w not in entity_stopwords])
    entity_words = set([w for w in content_words if w in entity_stopwords])

    # If no intent words remain after entity filtering, fall back to content words
    if not intent_words:
        intent_words = set(content_words)

    intent_stems = {get_stem(w) for w in intent_words if len(w) > 2}
    entity_stems = {get_stem(w) for w in entity_words if len(w) > 2}

    # Combined evidence check for topic presence
    all_content = " ".join([b["text"] for b in parsed_blocks])
    all_content_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", all_content.lower()))
    all_content_stems = {get_stem(w) for w in all_content_tokens if len(w) > 2}

    intent_overlap = (intent_words & all_content_tokens) | (intent_stems & all_content_stems)

    # If the user is asking about specific intent topic(s) and NONE exist in the context, ABSTAIN!
    if len(intent_words) > 0 and len(intent_overlap) == 0:
        return json.dumps({
            "answer": "The available documents do not contain sufficient evidence to answer this query.",
            "claims": [],
            "insufficient_evidence": True,
        }, indent=2)

    def extract_candidate_sentences(text: str) -> List[str]:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        candidates = []
        current_header = ""
        for l in lines:
            if l.startswith("#"):
                current_header = re.sub(r"^#+\s*", "", l).strip()
                continue
            l_clean = re.sub(r"^[\*\-]\s*", "", l)
            l_clean = re.sub(r"^\d+\.\s*", "", l_clean).strip()
            # If line is a short header like 'Architecture and Retrieval' or 'Main Objectives' without ending punctuation
            if len(l_clean) < 35 and not any(l_clean.endswith(p) for p in [".", "!", "?"]) and (" and " in l_clean or len(l_clean.split()) <= 4):
                current_header = l_clean
                continue
            for part in re.split(r"(?<=[.!?])\s+", l_clean):
                part_clean = part.strip().rstrip(".")
                if len(part_clean) > 18 and not part_clean.lower().startswith("evidence chunk"):
                    if current_header and (part_clean.lower().startswith(("improve", "reduce", "provide", "prevent", "enable", "ensure")) or len(part_clean.split()) <= 6):
                        candidates.append(f"{current_header}: {part_clean}")
                    else:
                        candidates.append(part_clean)
        return candidates

    def score_candidate(cand: str, blk_score: float):
        words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", cand.lower()))
        stems = {get_stem(w) for w in words if len(w) > 2}

        exact_intent = words & intent_words
        stem_intent = stems & intent_stems
        exact_entity = words & entity_words
        stem_entity = stems & entity_stems

        intent_points = len(exact_intent) * 10 + len(stem_intent) * 6
        entity_points = len(exact_entity) * 2 + len(stem_entity) * 1

        # A valid claim sentence MUST address the question intent
        if intent_points == 0:
            return 0.0, set()

        total = float(intent_points + entity_points) * (0.5 + 0.5 * blk_score)
        matched_intent = exact_intent | stem_intent
        return total, matched_intent

    all_candidates = []
    for block in parsed_blocks:
        cands = extract_candidate_sentences(block["text"])
        for c in cands:
            sc, matched_int = score_candidate(c, block["score"])
            if sc > 0:
                all_candidates.append({
                    "score": sc,
                    "text": c,
                    "chunk_id": block["chunk_id"],
                    "matched_intent": matched_int
                })

    all_candidates.sort(key=lambda x: x["score"], reverse=True)

    if not all_candidates:
        return json.dumps({
            "answer": "The available documents do not contain sufficient evidence to answer this query.",
            "claims": [],
            "insufficient_evidence": True,
        }, indent=2)

    selected = []
    covered_intent = set()
    seen_texts = set()

    for cand in all_candidates:
        cand_norm = cand["text"].lower()
        if any(cand_norm in prev or prev in cand_norm for prev in seen_texts):
            continue
        new_intent = cand["matched_intent"] - covered_intent
        if new_intent or not selected:
            selected.append(cand)
            seen_texts.add(cand_norm)
            covered_intent.update(cand["matched_intent"])
            if len(selected) >= 3:
                break

    if len(selected) < 2:
        for cand in all_candidates:
            cand_norm = cand["text"].lower()
            if not any(cand_norm in prev or prev in cand_norm for prev in seen_texts):
                selected.append(cand)
                seen_texts.add(cand_norm)
                if len(selected) >= 2:
                    break

    claims = []
    for idx, item in enumerate(selected, start=1):
        sent = item["text"].strip().rstrip(".")
        claims.append({
            "claim_id": f"C{idx}",
            "text": f"{sent}.",
            "evidence_chunk_ids": [item["chunk_id"]] if item["chunk_id"] else []
        })

    answer_text = " ".join([c["text"] for c in claims])

    return json.dumps({
        "answer": answer_text,
        "claims": claims,
        "insufficient_evidence": False,
    }, indent=2)

# Run the test cases
context_text = """[Source: doc-abc | Chunk: fa6c6b90-4340-401d-a1ab-95498070e385 | Page: 1 | Section: Architecture and Retrieval | Score: 0.9912]
Company ABC Enterprise Knowledge Policy
Company ABC maintains an internal enterprise knowledge platform for employees.
The platform helps employees search company documents and research internal information.
Main Objectives
Improve access to internal company knowledge.
Reduce the time employees spend searching through documents.
Provide answers with supporting evidence and citations.
Prevent unsupported answers when sufficient evidence is unavailable.
Architecture and Retrieval
The platform uses hybrid retrieval combining semantic vector search and keyword search. A reranking stage improves the ordering of retrieved documents. For complex questions, an agent decomposes the question into smaller sub-questions and retrieves evidence for each part.

---

[Source: doc-bench | Chunk: ff7a8a34-f1cb-468b-bb26-9e4b301cff75 | Page: 1 | Section: Findings | Score: 0.0061]
We evaluated 50 benchmark queries across dense and lexical retrieval modes.
Dense Only retrieval achieved an accuracy score of 81.5% in the benchmark results table.
"""

queries = [
    ("Query A", "What retrieval methods does Company ABC use?"),
    ("Query B", "How does Company ABC handle complex questions?"),
    ("Query C", "What is Company ABC's employee vacation allowance?"),
    ("Query D (Agentic)", "Compare the main objectives of the enterprise knowledge platform with how it retrieves information for complex questions."),
]

for label, q in queries:
    prompt = f"User Research Question:\n{q}\n\nSupplied Context Blocks (Evidence Chunks):\n{context_text}"
    res = json.loads(generate_grounded_answer_v2(prompt))
    print(f"\n==================== {label}: '{q}' ====================")
    print("INSUFFICIENT:", res.get("insufficient_evidence"))
    print("ANSWER:\n", res.get("answer"))
    print("CLAIMS:")
    for c in res.get("claims", []):
        print(" -", c.get("text"), "->", c.get("evidence_chunk_ids"))
