#!/usr/bin/env python3
"""English Coach - Persistent AI coaching agent."""

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from flask import Flask, render_template, request, Response, stream_with_context
import anthropic

app = Flask(__name__)
client = anthropic.Anthropic(
    api_key=os.environ.get("MINIMAX_API_KEY"),
    base_url="https://api.minimax.io/anthropic",
)

DB_PATH = os.path.join(os.path.dirname(__file__), "coach.db")


# ─── Database ────────────────────────────────────────────────────────────────

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                mode TEXT DEFAULT 'conversation',
                message_count INTEGER DEFAULT 0,
                summary TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_type TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL DEFAULT 'grammar',
                example TEXT,
                correction TEXT,
                explanation TEXT,
                count INTEGER DEFAULT 1,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                status TEXT DEFAULT 'active'
            );

            CREATE TABLE IF NOT EXISTS learner_profile (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS coaching_plan (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                focus_skill TEXT NOT NULL,
                skill_type TEXT DEFAULT 'active',
                weekly_goal TEXT,
                exercises TEXT,
                encouragement TEXT,
                sessions_since_update INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS job_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT NOT NULL,
                payload TEXT DEFAULT '{}',
                status TEXT DEFAULT 'pending',
                created_at TEXT NOT NULL,
                claimed_at TEXT
            );
        """)


init_db()


# ─── Profile ─────────────────────────────────────────────────────────────────

def get_profile():
    with get_db() as db:
        rows = db.execute("SELECT key, value FROM learner_profile").fetchall()
    profile = {
        "level": "unknown",
        "sessions": 0,
        "strengths": [],
        "active_score": 50,
        "passive_score": 50,
        "vocabulary_looked_up": [],
        "last_updated": None,
    }
    for row in rows:
        try:
            profile[row["key"]] = json.loads(row["value"])
        except Exception:
            profile[row["key"]] = row["value"]
    return profile


def save_profile(profile):
    with get_db() as db:
        for key, value in profile.items():
            db.execute(
                "INSERT OR REPLACE INTO learner_profile (key, value) VALUES (?, ?)",
                (key, json.dumps(value))
            )


def increment_sessions():
    with get_db() as db:
        row = db.execute(
            "SELECT value FROM learner_profile WHERE key='sessions'"
        ).fetchone()
        count = json.loads(row["value"]) + 1 if row else 1
        db.execute(
            "INSERT OR REPLACE INTO learner_profile (key, value) VALUES ('sessions', ?)",
            (json.dumps(count),)
        )
        db.execute(
            "UPDATE coaching_plan SET sessions_since_update = sessions_since_update + 1 WHERE is_active = 1"
        )
    return count


# ─── Sessions ────────────────────────────────────────────────────────────────

def create_session(mode="conversation"):
    with get_db() as db:
        cursor = db.execute(
            "INSERT INTO sessions (started_at, mode) VALUES (?, ?)",
            (datetime.now().isoformat(), mode)
        )
        return cursor.lastrowid


def save_message(session_id, role, content):
    with get_db() as db:
        db.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now().isoformat())
        )
        db.execute(
            "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
            (session_id,)
        )


def get_last_session_summary():
    with get_db() as db:
        row = db.execute(
            "SELECT summary FROM sessions WHERE summary IS NOT NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return row["summary"] if row else None


# ─── Errors ──────────────────────────────────────────────────────────────────

def get_top_errors(limit=8):
    with get_db() as db:
        rows = db.execute(
            """SELECT error_type, category, count, status, last_seen, example, correction, explanation
               FROM errors ORDER BY count DESC, last_seen DESC LIMIT ?""",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def upsert_error(error_type, category, example=None, correction=None, explanation=None):
    now = datetime.now().strftime("%Y-%m-%d")
    with get_db() as db:
        row = db.execute(
            "SELECT id FROM errors WHERE error_type = ?", (error_type,)
        ).fetchone()
        if row:
            db.execute(
                "UPDATE errors SET count = count + 1, last_seen = ?, "
                "example = COALESCE(?, example), correction = COALESCE(?, correction), "
                "explanation = COALESCE(?, explanation), status = 'active' WHERE id = ?",
                (now, example, correction, explanation, row["id"])
            )
        else:
            db.execute(
                """INSERT INTO errors (error_type, category, example, correction, explanation, first_seen, last_seen)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (error_type, category, example, correction, explanation, now, now)
            )


# ─── Coaching Plan ───────────────────────────────────────────────────────────

def get_active_plan():
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM coaching_plan WHERE is_active = 1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if not row:
        return None
    plan = dict(row)
    if plan.get("exercises"):
        try:
            plan["exercises"] = json.loads(plan["exercises"])
        except Exception:
            plan["exercises"] = []
    return plan


def should_update_plan():
    with get_db() as db:
        row = db.execute(
            "SELECT sessions_since_update FROM coaching_plan WHERE is_active = 1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return (not row) or (row["sessions_since_update"] >= 3)


def generate_coaching_plan(profile, errors):
    error_summary = "\n".join(
        f"- {e['error_type']} (seen {e['count']}x, category: {e['category']})"
        for e in errors[:5]
    ) or "No errors recorded yet — first session."

    prompt = f"""You are an English coach creating a personalized study plan.

Learner profile:
- Level: {profile.get('level', 'unknown')}
- Sessions completed: {profile.get('sessions', 0)}
- Active English score (production/writing/speaking): {profile.get('active_score', 50)}/100
- Passive English score (comprehension/reading): {profile.get('passive_score', 50)}/100
- Strengths: {', '.join(profile.get('strengths', [])) or 'not yet assessed'}

Top recurring errors:
{error_summary}

Create a focused coaching plan targeting the learner's biggest weakness. Return ONLY valid JSON:
{{
  "focus_skill": "specific skill name (e.g. 'Third Conditional Sentences')",
  "skill_type": "active or passive",
  "weekly_goal": "one clear, measurable goal sentence",
  "exercises": [
    {{"title": "Exercise name", "description": "What to do (1-2 sentences)", "type": "active or passive"}},
    {{"title": "Exercise name", "description": "What to do (1-2 sentences)", "type": "active or passive"}},
    {{"title": "Exercise name", "description": "What to do (1-2 sentences)", "type": "active or passive"}}
  ],
  "encouragement": "short motivating message referencing their specific progress (1 sentence)"
}}"""

    try:
        response = client.messages.create(
            model="MiniMax-M2.5",
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        text_block = next((b for b in response.content if hasattr(b, "text")), None)
        if not text_block:
            return
        text = text_block.text.strip()
        if "```" in text:
            text = text.split("```")[1].lstrip("json").strip()
        plan_data = json.loads(text)

        now = datetime.now().isoformat()
        with get_db() as db:
            db.execute("UPDATE coaching_plan SET is_active = 0")
            db.execute(
                """INSERT INTO coaching_plan
                   (created_at, focus_skill, skill_type, weekly_goal, exercises, encouragement, sessions_since_update, is_active)
                   VALUES (?, ?, ?, ?, ?, ?, 0, 1)""",
                (
                    now,
                    plan_data.get("focus_skill", "General English"),
                    plan_data.get("skill_type", "active"),
                    plan_data.get("weekly_goal", ""),
                    json.dumps(plan_data.get("exercises", [])),
                    plan_data.get("encouragement", ""),
                )
            )
    except Exception:
        pass


# ─── Job Queue ────────────────────────────────────────────────────────────────

def enqueue_job(job_type, payload=None):
    """Add a job to the persistent queue."""
    with get_db() as db:
        db.execute(
            "INSERT INTO job_queue (job_type, payload, status, created_at) VALUES (?, ?, 'pending', ?)",
            (job_type, json.dumps(payload or {}), datetime.now().isoformat())
        )


def claim_job():
    """Atomically claim a pending job. Returns job dict or None."""
    # Reclaim jobs stuck in 'processing' for over 10 minutes first
    cutoff = (datetime.now() - timedelta(minutes=10)).isoformat()
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            "UPDATE job_queue SET status='pending', claimed_at=NULL "
            "WHERE status='processing' AND claimed_at < ?",
            (cutoff,)
        )
        conn.commit()

        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM job_queue WHERE status='pending' ORDER BY id LIMIT 1"
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE job_queue SET status='processing', claimed_at=? WHERE id=?",
                (datetime.now().isoformat(), row["id"])
            )
            conn.commit()
            return dict(row)
        conn.commit()
        return None
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return None
    finally:
        conn.close()


def finish_job(job_id, success=True):
    with get_db() as db:
        db.execute(
            "UPDATE job_queue SET status=? WHERE id=?",
            ("done" if success else "failed", job_id)
        )


def run_job(job):
    """Execute a job by type."""
    payload = json.loads(job.get("payload") or "{}")
    job_type = job["job_type"]
    try:
        if job_type == "analyze_session":
            _do_analyze_session(
                payload.get("messages", []),
                payload.get("session_id"),
                payload.get("mode", "conversation"),
            )
        elif job_type == "generate_plan":
            profile = get_profile()
            errors = get_top_errors(5)
            generate_coaching_plan(profile, errors)
        finish_job(job["id"], True)
    except Exception:
        finish_job(job["id"], False)


def worker_loop():
    """Persistent background worker that processes queued jobs."""
    while True:
        try:
            job = claim_job()
            if job:
                run_job(job)
            else:
                time.sleep(2)
        except Exception:
            time.sleep(5)


_worker_started = False
_worker_lock = threading.Lock()


def start_worker():
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            t = threading.Thread(target=worker_loop, daemon=True)
            t.start()
            _worker_started = True


start_worker()


# ─── Context Builder ─────────────────────────────────────────────────────────

def build_coach_context():
    profile = get_profile()
    errors = get_top_errors(5)
    plan = get_active_plan()
    last_summary = get_last_session_summary()

    lines = ["[LEARNER CONTEXT]"]
    if profile["level"] != "unknown":
        lines.append(f"Level: {profile['level']}")
    lines.append(f"Sessions completed: {profile.get('sessions', 0)}")
    lines.append(f"Active English score (writing/speaking): {profile.get('active_score', 50)}/100")
    lines.append(f"Passive English score (reading/listening): {profile.get('passive_score', 50)}/100")
    if profile.get("strengths"):
        lines.append(f"Strengths: {', '.join(profile['strengths'])}")

    if errors:
        lines.append("\nRecurring errors to watch and address:")
        for e in errors:
            lines.append(f"  - {e['error_type']} (seen {e['count']}x) [{e['category']}]")
            if e.get("example"):
                lines.append(f"    e.g. \"{e['example']}\" -> \"{e.get('correction', '?')}\"")

    if plan:
        lines.append(f"\nCurrent coaching focus: {plan['focus_skill']} ({plan['skill_type']} skill)")
        lines.append(f"Weekly goal: {plan.get('weekly_goal', '')}")

    if last_summary:
        lines.append(f"\nLast session: {last_summary}")

    return "\n".join(lines)


# ─── Background Analysis ─────────────────────────────────────────────────────

def _do_analyze_session(messages, session_id, mode):
    """Core analysis logic — runs inside a job worker."""
    convo_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages[-12:]
        if isinstance(m.get("content"), str)
    )
    profile = get_profile()

    prompt = f"""Analyze this English coaching conversation and extract structured insights.

Current learner profile:
- Level: {profile.get('level', 'unknown')}
- Active score: {profile.get('active_score', 50)}/100
- Passive score: {profile.get('passive_score', 50)}/100

Conversation:
{convo_text}

Return ONLY valid JSON:
{{
  "level": "beginner|elementary|intermediate|upper-intermediate|advanced|unknown",
  "active_score": <integer 0-100: ability to PRODUCE English: write, speak, construct sentences>,
  "passive_score": <integer 0-100: ability to UNDERSTAND English: read, comprehend, infer meaning>,
  "strengths": ["specific strength 1", "specific strength 2"],
  "errors": [
    {{
      "error_type": "concise name e.g. 'missing definite article'",
      "category": "grammar|vocabulary|style|structure",
      "example": "exact wrong phrase from the conversation",
      "correction": "corrected version",
      "explanation": "brief rule explanation (max 15 words)"
    }}
  ],
  "session_summary": "1-2 sentence summary of what was practiced and the key takeaway"
}}

Only include errors actually observed in the conversation. Be specific."""

    response = client.messages.create(
        model="MiniMax-M2.5",
        max_tokens=900,
        messages=[{"role": "user", "content": prompt}],
    )
    text_block = next((b for b in response.content if hasattr(b, "text")), None)
    if not text_block:
        raise ValueError("No text block in response")
    text = text_block.text.strip()
    if "```" in text:
        text = text.split("```")[1].lstrip("json").strip()
    data = json.loads(text)

    # Update profile
    profile["level"] = data.get("level", profile["level"])
    profile["active_score"] = max(0, min(100, int(data.get("active_score", profile["active_score"]))))
    profile["passive_score"] = max(0, min(100, int(data.get("passive_score", profile["passive_score"]))))
    profile["strengths"] = data.get("strengths", profile.get("strengths", []))[:4]
    profile["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    save_profile(profile)

    # Record errors
    for error in data.get("errors", []):
        upsert_error(
            error.get("error_type", "unknown error"),
            error.get("category", "grammar"),
            error.get("example"),
            error.get("correction"),
            error.get("explanation"),
        )

    # Save session summary
    if data.get("session_summary"):
        with get_db() as db:
            db.execute(
                "UPDATE sessions SET summary = ?, ended_at = ? WHERE id = ?",
                (data["session_summary"], datetime.now().isoformat(), session_id)
            )

    increment_sessions()

    if should_update_plan():
        enqueue_job("generate_plan")


# ─── System Prompts ───────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "conversation": """You are Alex, an experienced and encouraging English coach.
Your role: have natural, engaging conversations while actively coaching the learner.

Rules:
- Respond naturally to what the user says — be a real conversation partner
- At the END of your response, add a **Coach's Note** section ONLY when you spot errors or something worth highlighting
- In Coach's Note: gently correct errors with examples, suggest better phrasing, or compliment good usage
- Keep Coach's Notes to 1-3 bullet points maximum
- If this is a returning learner, reference their progress or last session when relevant
- Adapt your language complexity to the learner's level from their profile
- Actively work on their focus skill from the coaching plan when opportunities arise""",

    "grammar": """You are an expert English grammar coach.
Response format:
1. **Corrected Version** — full corrected text
2. **Corrections** — each error: original -> corrected + clear rule explanation
3. **Pattern Note** — if you see a recurring pattern, name it specifically
4. **Overall Assessment** — 2-3 sentences: what's strong and the top priority to fix

Be thorough, specific, and encouraging. Always explain the "why" behind each rule.""",

    "vocabulary": """You are an English vocabulary coach.
For any word or phrase:
1. **Definition** — clear, simple definition
2. **Pronunciation** — phonetic spelling + tips
3. **Examples** — 3 natural sentences (easy to harder)
4. **Synonyms & Antonyms** — 3-4 each with nuance notes
5. **Collocations** — most common word combinations
6. **Usage Tips** — formal/informal, common mistakes, register
7. **Memory Hook** — a memorable trick to remember it""",

    "writing": """You are a professional English writing coach.
Response format:
1. **Summary** — overall impression (2-3 sentences)
2. **Strengths** — specific, concrete praise
3. **Priority Improvements** — ranked list with examples from the text
4. **Revised Excerpt** — rewrite one key paragraph showing the improvements
5. **Action Items** — 3 specific practice tasks based on this particular writing

Consider: clarity, structure, vocabulary range, grammar, tone, and flow.""",

    "practice": """You are an English practice coach delivering targeted exercises.

Your role:
- Generate exercises that directly target the learner's recurring errors and current focus skill
- For ACTIVE exercises: writing prompts, sentence construction, error correction tasks
- For PASSIVE exercises: reading passages with comprehension questions, vocabulary in context, inference tasks
- Always state which skill each exercise targets and why it matters for this learner
- Give clear, specific instructions
- When the learner submits answers, give detailed, encouraging feedback
- Reference their coaching plan's focus skill and weekly goal

Start each session by offering 2-3 exercise options targeting their weakest areas, then let them choose.""",
}


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/memory", methods=["GET"])
def get_memory():
    with get_db() as db:
        pending_count = db.execute(
            "SELECT COUNT(*) FROM job_queue WHERE status IN ('pending', 'processing')"
        ).fetchone()[0]
    return {
        "profile": get_profile(),
        "errors": get_top_errors(10),
        "plan": get_active_plan(),
        "jobs_pending": pending_count > 0,
    }


@app.route("/api/memory/reset", methods=["POST"])
def reset_memory():
    with get_db() as db:
        db.execute("DELETE FROM learner_profile")
        db.execute("DELETE FROM errors")
        db.execute("UPDATE coaching_plan SET is_active = 0")
        db.execute("DELETE FROM job_queue WHERE status IN ('pending', 'processing')")
    return {"ok": True}


@app.route("/api/history", methods=["GET"])
def get_history():
    limit = int(request.args.get("limit", 15))
    with get_db() as db:
        rows = db.execute(
            """SELECT id, started_at, mode, message_count, summary
               FROM sessions WHERE message_count > 0
               ORDER BY id DESC LIMIT ?""",
            (limit,)
        ).fetchall()
    return {"sessions": [dict(r) for r in rows]}


@app.route("/api/session/<int:session_id>", methods=["GET"])
def get_session_messages(session_id):
    with get_db() as db:
        rows = db.execute(
            "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,)
        ).fetchall()
    return {"messages": [dict(r) for r in rows]}


@app.route("/api/plan/regenerate", methods=["POST"])
def regenerate_plan():
    enqueue_job("generate_plan")
    return {"ok": True}


@app.route("/api/progress", methods=["GET"])
def get_progress():
    with get_db() as db:
        sessions = db.execute(
            "SELECT id, started_at, mode, message_count FROM sessions WHERE message_count > 0 ORDER BY id DESC LIMIT 20"
        ).fetchall()
        errors = db.execute(
            "SELECT error_type, category, count, status FROM errors ORDER BY count DESC LIMIT 10"
        ).fetchall()
    return {
        "sessions": [dict(s) for s in sessions],
        "error_trends": [dict(e) for e in errors],
    }


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    mode = data.get("mode", "conversation")
    messages = data.get("messages", [])
    session_id = data.get("session_id")

    if not messages:
        return {"error": "No messages provided"}, 400

    if not session_id:
        session_id = create_session(mode)

    last_user_msg = next((m for m in reversed(messages) if m["role"] == "user"), None)
    if last_user_msg:
        save_message(session_id, "user", last_user_msg["content"])

    context = build_coach_context()
    base_system = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["conversation"])
    system = f"{base_system}\n\n{context}".strip()

    if mode == "vocabulary" and last_user_msg:
        word = last_user_msg["content"].strip()
        if word and len(word) < 80:
            profile = get_profile()
            vocab = profile.get("vocabulary_looked_up", [])
            if word not in vocab:
                profile["vocabulary_looked_up"] = (vocab + [word])[-50:]
                save_profile(profile)

    def generate():
        try:
            yield f"data: {json.dumps({'session_id': session_id})}\n\n"

            full_text = ""
            with client.messages.stream(
                model="MiniMax-M2.5",
                max_tokens=2048,
                system=system,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    full_text += text
                    yield f"data: {json.dumps({'text': text})}\n\n"

            yield "data: [DONE]\n\n"

            save_message(session_id, "assistant", full_text)

            if mode in ("conversation", "grammar", "writing", "practice") and len(messages) >= 2:
                all_msgs = list(messages) + [{"role": "assistant", "content": full_text}]
                enqueue_job("analyze_session", {
                    "messages": all_msgs,
                    "session_id": session_id,
                    "mode": mode,
                })

        except anthropic.AuthenticationError:
            yield f"data: {json.dumps({'error': 'Invalid API key. Set MINIMAX_API_KEY.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\nEnglish Coach running at http://localhost:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
