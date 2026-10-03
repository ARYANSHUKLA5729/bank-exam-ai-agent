import os
import sqlite3
from datetime import datetime, date, timedelta

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "study.db")
SUBJECTS = ["Quant", "Reasoning", "English", "General Awareness",
            "Computer", "Mock Test", "Revision"]


# ---------- database ----------
def conn():
    return sqlite3.connect(DB)


def init_db():
    with conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS sessions(
            id INTEGER PRIMARY KEY, day TEXT, subject TEXT, topic TEXT,
            seconds INTEGER, questions INTEGER DEFAULT 0, correct INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS tasks(
            id INTEGER PRIMARY KEY, day TEXT, subject TEXT, title TEXT,
            done INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
        """)


def get_setting(key, default=None):
    with conn() as c:
        row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_setting(key, value):
    with conn() as c:
        if value is None:
            c.execute("DELETE FROM settings WHERE key=?", (key,))
        else:
            c.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (key, str(value)))


def sessions_df():
    with conn() as c:
        return pd.read_sql_query("SELECT * FROM sessions", c)


def tasks_df():
    with conn() as c:
        return pd.read_sql_query("SELECT * FROM tasks", c)


def add_session(subject, topic, seconds, questions=0, correct=0, day=None):
    day = day or date.today().isoformat()
    with conn() as c:
        c.execute(
            "INSERT INTO sessions(day,subject,topic,seconds,questions,correct) VALUES(?,?,?,?,?,?)",
            (day, subject, topic, int(seconds), int(questions), int(correct)))


# ---------- stats ----------
def fmt(seconds):
    seconds = int(seconds)
    h, m = seconds // 3600, (seconds % 3600) // 60
    return f"{h}h {m:02d}m"


def total_between(df, start, end):
    if df.empty:
        return 0
    d = pd.to_datetime(df["day"]).dt.date
    return int(df[(d >= start) & (d <= end)]["seconds"].sum())


def compute_streak(df, min_seconds):
    if df.empty:
        return 0
    per_day = df.groupby("day")["seconds"].sum()
    good = {date.fromisoformat(d) for d, s in per_day.items() if s >= min_seconds}
    d = date.today()
    if d not in good:          # today not finished yet, don't break the streak
        d -= timedelta(days=1)
    streak = 0
    while d in good:
        streak += 1
        d -= timedelta(days=1)
    return streak


def stats_summary():
    """Plain-text snapshot of the data, used as context for the AI."""
    df, tdf = sessions_df(), tasks_df()
    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    target_h = float(get_setting("target_hours", 4))
    lines = [f"Today: {today.isoformat()}. Daily target: {target_h}h."]
    lines.append(f"Today studied: {fmt(total_between(df, today, today))}")
    lines.append(f"This week: {fmt(total_between(df, week_start, today))}")
    lines.append(f"This month: {fmt(total_between(df, today.replace(day=1), today))}")
    lines.append(f"All time: {fmt(df['seconds'].sum() if not df.empty else 0)}")
    lines.append("Last 7 days:")
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        lines.append(f"  {d.isoformat()} ({d.strftime('%a')}): {fmt(total_between(df, d, d))}")
    if not df.empty:
        lines.append("Subject totals (all time):")
        for s, sec in df.groupby("subject")["seconds"].sum().sort_values(ascending=False).items():
            lines.append(f"  {s}: {fmt(sec)}")
        q, c = int(df["questions"].sum()), int(df["correct"].sum())
        if q:
            lines.append(f"Questions solved: {q}, correct: {c}, accuracy: {c / q:.0%}")
    if not tdf.empty:
        recent = tdf[pd.to_datetime(tdf["day"]).dt.date >= week_start]
        lines.append(f"Tasks this week: {int(recent['done'].sum())}/{len(recent)} completed")
        pending = tdf[(tdf["day"] == today.isoformat()) & (tdf["done"] == 0)]
        if not pending.empty:
            lines.append("Pending today: " + "; ".join(f"{r.subject} - {r.title}" for r in pending.itertuples()))
    lines.append(f"Streak: {compute_streak(df, float(get_setting('streak_hours', 2)) * 3600)} days")
    return "\n".join(lines)


# ---------- AI ----------
def ask_ai(user_msg, history=None):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return "Set the GEMINI_API_KEY environment variable to enable the AI coach."
    try:
        from google import genai
        client = genai.Client(api_key=key)
        system = ("You are a strict but friendly study accountability coach for a student "
                  "preparing for banking exams. Use ONLY the data below for numbers; never "
                  "invent figures. Be concise and practical.\n\nSTUDY DATA:\n" + stats_summary())
        convo = "\n".join(f"{r}: {t}" for r, t in (history or [])[-6:])
        prompt = f"{system}\n\nConversation so far:\n{convo}\n\nUser: {user_msg}"
        resp = client.models.generate_content(
            model=get_setting("gemini_model", "gemini-2.5-flash"), contents=prompt)
        return resp.text
    except Exception as e:
        return f"AI error: {e}"


# ---------- UI ----------
st.set_page_config(page_title="Study Coach", page_icon="🎯", layout="wide")
init_db()
st.title("🎯 Banking Study Coach")

with st.sidebar:
    st.header("Settings")
    target = st.number_input("Daily target (hours)", 0.5, 16.0,
                             float(get_setting("target_hours", 4)), 0.5)
    streak_h = st.number_input("Hours needed for a streak day", 0.5, 16.0,
                               float(get_setting("streak_hours", 2)), 0.5)
    model = st.text_input("Gemini model", get_setting("gemini_model", "gemini-2.5-flash"))
    set_setting("target_hours", target)
    set_setting("streak_hours", streak_h)
    set_setting("gemini_model", model)

tab_dash, tab_timer, tab_tasks, tab_ai = st.tabs(["📊 Dashboard", "⏱️ Timer", "✅ Tasks", "🤖 AI Coach"])

# ----- Dashboard -----
with tab_dash:
    df, tdf = sessions_df(), tasks_df()
    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    today_s = total_between(df, today, today)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Today", fmt(today_s))
    c2.metric("This week", fmt(total_between(df, week_start, today)))
    c3.metric("This month", fmt(total_between(df, today.replace(day=1), today)))
    c4.metric("All time", fmt(df["seconds"].sum() if not df.empty else 0))
    c5.metric("🔥 Streak", f"{compute_streak(df, streak_h * 3600)} days")

    pct = min(today_s / (target * 3600), 1.0)
    st.progress(pct, text=f"Daily target: {fmt(today_s)} / {target:g}h ({pct:.0%})")

    if not tdf.empty:
        tt = tdf[tdf["day"] == today.isoformat()]
        if len(tt):
            st.write(f"**Tasks today:** {int(tt['done'].sum())}/{len(tt)} completed "
                     f"({tt['done'].sum() / len(tt):.0%})")

    if not df.empty and df["questions"].sum() > 0:
        q, c = int(df["questions"].sum()), int(df["correct"].sum())
        st.write(f"**Questions solved:** {q} · **Correct:** {c} · **Accuracy:** {c / q:.1%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Last 7 days (hours)")
        days = [today - timedelta(days=i) for i in range(6, -1, -1)]
        chart = pd.DataFrame({
            "day": [d.strftime("%a %d") for d in days],
            "hours": [round(total_between(df, d, d) / 3600, 2) for d in days],
        }).set_index("day")
        st.bar_chart(chart)
    with right:
        st.subheader("Time by subject")
        if df.empty:
            st.info("No sessions yet. Start the timer!")
        else:
            by_sub = df.groupby("subject")["seconds"].sum().sort_values(ascending=False)
            st.bar_chart((by_sub / 3600).round(2))
            for s, sec in by_sub.items():
                st.write(f"{s}: **{fmt(sec)}**")

# ----- Timer -----
with tab_timer:
    start = get_setting("active_start")
    if start:
        subj, topic = get_setting("active_subject"), get_setting("active_topic", "")
        elapsed = (datetime.now() - datetime.fromisoformat(start)).total_seconds()
        st.success(f"🟢 Running: **{subj}** {('· ' + topic) if topic else ''}")
        st.metric("Elapsed", fmt(elapsed) + f" ({int(elapsed % 60):02d}s)")
        st.button("🔄 Refresh time")
        with st.form("stop_form"):
            qs = st.number_input("Questions solved this session", 0, 1000, 0)
            cor = st.number_input("Correct", 0, 1000, 0)
            if st.form_submit_button("⏹️ Stop & save"):
                add_session(subj, topic, elapsed, qs, min(cor, qs))
                for k in ("active_start", "active_subject", "active_topic"):
                    set_setting(k, None)
                st.rerun()
        if st.button("Discard session"):
            for k in ("active_start", "active_subject", "active_topic"):
                set_setting(k, None)
            st.rerun()
    else:
        with st.form("start_form"):
            subj = st.selectbox("Subject", SUBJECTS)
            topic = st.text_input("Topic (e.g. Percentage)")
            if st.form_submit_button("▶️ Start session"):
                set_setting("active_start", datetime.now().isoformat())
                set_setting("active_subject", subj)
                set_setting("active_topic", topic)
                st.rerun()

    with st.expander("➕ Log a past session manually"):
        with st.form("manual_form"):
            m_day = st.date_input("Date", date.today())
            m_sub = st.selectbox("Subject", SUBJECTS, key="m_sub")
            m_top = st.text_input("Topic", key="m_top")
            m_min = st.number_input("Minutes", 1, 600, 30)
            m_q = st.number_input("Questions solved", 0, 1000, 0, key="m_q")
            m_c = st.number_input("Correct", 0, 1000, 0, key="m_c")
            if st.form_submit_button("Save"):
                add_session(m_sub, m_top, m_min * 60, m_q, min(m_c, m_q), m_day.isoformat())
                st.rerun()

# ----- Tasks -----
with tab_tasks:
    with st.form("task_form", clear_on_submit=True):
        c1, c2 = st.columns([1, 3])
        t_sub = c1.selectbox("Subject", SUBJECTS)
        t_title = c2.text_input("Task (e.g. Percentage practice set)")
        if st.form_submit_button("Add task for today") and t_title.strip():
            with conn() as c:
                c.execute("INSERT INTO tasks(day,subject,title) VALUES(?,?,?)",
                          (date.today().isoformat(), t_sub, t_title.strip()))
            st.rerun()

    tdf = tasks_df()
    todays = tdf[tdf["day"] == date.today().isoformat()] if not tdf.empty else tdf
    if todays.empty:
        st.info("No tasks planned for today.")
    else:
        done_n = int(todays["done"].sum())
        st.progress(done_n / len(todays), text=f"{done_n}/{len(todays)} completed")
        for r in todays.itertuples():
            checked = st.checkbox(f"{r.subject} — {r.title}", value=bool(r.done), key=f"task_{r.id}")
            if checked != bool(r.done):
                with conn() as c:
                    c.execute("UPDATE tasks SET done=? WHERE id=?", (int(checked), r.id))
                st.rerun()

    with st.expander("Carry over unfinished tasks from yesterday"):
        y = (date.today() - timedelta(days=1)).isoformat()
        if st.button("Copy to today"):
            with conn() as c:
                rows = c.execute("SELECT subject,title FROM tasks WHERE day=? AND done=0", (y,)).fetchall()
                c.executemany("INSERT INTO tasks(day,subject,title) VALUES(?,?,?)",
                              [(date.today().isoformat(), s, t) for s, t in rows])
            st.rerun()

# ----- AI Coach -----
with tab_ai:
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if st.button("📈 Generate weekly report"):
        with st.spinner("Analysing your week..."):
            report = ask_ai("Write my weekly progress report: total hours, average per day, "
                            "best and worst day, subject balance, task completion, and 3 specific "
                            "things to do next week.")
        st.session_state.chat.append(("assistant", report))
    for role, text in st.session_state.chat:
        with st.chat_message(role):
            st.write(text)
    if msg := st.chat_input("Ask about your progress, or what to study today"):
        st.session_state.chat.append(("user", msg))
        with st.spinner("Thinking..."):
            reply = ask_ai(msg, st.session_state.chat[:-1])
        st.session_state.chat.append(("assistant", reply))
        st.rerun()