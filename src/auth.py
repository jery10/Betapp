"""
Single-user auth gate for BetPredict.
Verifies credentials against TipKing's Supabase users table.
Only the ALLOWED_EMAIL can log in.
"""
import os
import streamlit as st
from supabase import create_client
from werkzeug.security import check_password_hash

ALLOWED_EMAIL = "jeremiahakinlabi@gmail.com"

_sb = None

def _get_db():
    global _sb
    if _sb is None:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")
        if url and key:
            _sb = create_client(url, key)
    return _sb


def _verify(email: str, password: str) -> tuple[bool, str]:
    if email.lower().strip() != ALLOWED_EMAIL:
        return False, "Access restricted."
    try:
        db = _get_db()
        if db is None:
            return False, "Database unavailable."
        res = db.table("users").select("password_hash").eq("email", email.lower().strip()).execute()
        if not res.data:
            return False, "Account not found."
        if not check_password_hash(res.data[0]["password_hash"], password):
            return False, "Incorrect password."
        return True, ""
    except Exception as e:
        return False, f"Error: {e}"


def require_login():
    """
    Call at the top of app.py. Shows a login screen until authenticated.
    Returns immediately (does nothing) if already logged in.
    """
    if st.session_state.get("authenticated"):
        return

    st.set_page_config(page_title="BetPredict — Sign In", page_icon="🔒", layout="centered")

    st.markdown("""
    <style>
    .login-wrap { max-width: 400px; margin: 80px auto; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='login-wrap'>", unsafe_allow_html=True)
    st.markdown("## 🔒 BetPredict")
    st.markdown("Private access only.")

    with st.form("login_form"):
        email    = st.text_input("Email", placeholder="your@email.com")
        password = st.text_input("Password", type="password")
        submit   = st.form_submit_button("Sign In", use_container_width=True)

    if submit:
        ok, err = _verify(email, password)
        if ok:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error(err)

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()
