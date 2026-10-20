import base64
import io

from pypdf import PdfReader
from odf.opendocument import load as load_odf
from odf.text import P
from odf import teletype


import os
import time
import sqlite3
import secrets
import logging
from collections import defaultdict, deque
from pathlib import Path
from functools import wraps
from datetime import timedelta

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
from authlib.integrations.flask_client import OAuth
from groq import Groq

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartlearn")

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "smartlearn.db"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads" / "profiles"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

IS_DEV = os.getenv("FLASK_ENV", "production").lower() == "development"

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# ---------------------------------------------------------------
# SECRET KEY  (must come from environment, never hardcode it)
# ---------------------------------------------------------------
_secret = os.getenv("SECRET_KEY")
if not _secret:
    if IS_DEV:
        _secret = secrets.token_hex(32)
        logger.warning("SECRET_KEY missing - using a temporary key (dev only).")
    else:
        raise RuntimeError(
            "SECRET_KEY environment variable is not set. "
            "Add it in your .env file or Render environment settings."
        )
app.secret_key = _secret

app.permanent_session_lifetime = timedelta(days=7)

app.config.update(
    # Secure cookies need HTTPS; disabled automatically on localhost dev
    SESSION_COOKIE_SECURE=not IS_DEV,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_REFRESH_EACH_REQUEST=True,
    MAX_CONTENT_LENGTH=3 * 1024 * 1024,
    UPLOAD_FOLDER=str(UPLOAD_FOLDER),
)

# ---------------------------------------------------------------
# GROQ CONFIGURATION  (key comes from environment)
# --------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM_INSTRUCTION = """
You are SmartLearn AI Tutor, a clear, accurate, friendly tutor for school and college students.

LANGUAGE AND STYLE
- Reply in the same language as the user. If the user writes Hinglish, reply in natural Hinglish; if Hindi, use simple Hindi; if English, use clear English.
- Be warm and encouraging, but do not add unnecessary greetings, filler, repeated summaries, or irrelevant questions.
- Use short paragraphs and clear numbered steps when solving a problem.
- Answer the exact question first. Explain only what helps the student understand.
- Use correct spelling, grammar, and readable spacing.

FORMATTING RULES (VERY IMPORTANT)
- The website chat may display plain text rather than fully rendered Markdown/LaTeX. Therefore NEVER output raw LaTeX commands or delimiters such as square-bracket or parenthesis math delimiters, fraction or boxed commands, or backslash-based formulas.
- Do not use Markdown headings with ###, tables, or code fences unless the user specifically asks for them.
- For maths, write formulas in readable plain text, for example: 2x² − 7x + 3 = 0. Use x², x³, √, ×, ÷ and normal fractions like 1/2 where helpful.
- Put each equation on its own line. Use labels such as “Step 1”, “Step 2”, and “Final answer”.
- Never describe a factorisation as complete until you verify it by expansion or a quick check.
- Avoid showing internal planning, alternative methods that are not needed, or long explanations of obvious steps.

TEACHING RULES
- Explain step by step using simple language and a small example when useful.
- For homework, provide a correct worked solution and a brief explanation so the student can learn.
- For calculations, double-check the answer before replying.
- Support maths, science, coding, English, general knowledge, and study questions.
- Ask at most one short follow-up question, and only when it genuinely helps learning.
"""

client = None

if GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized successfully")
    except Exception:
        logger.exception("Groq configuration error")
else:
    logger.warning(
        "GROQ_API_KEY is not set - AI tutor will be disabled."
    )

# ---------------------------------------------------------------
# OAUTH (Google + Facebook)
# ---------------------------------------------------------------
oauth = OAuth(app)

google = oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

facebook = oauth.register(
    name="facebook",
    client_id=os.getenv("FACEBOOK_CLIENT_ID"),
    client_secret=os.getenv("FACEBOOK_CLIENT_SECRET"),
    api_base_url="https://graph.facebook.com/v12.0/",
    access_token_url="https://graph.facebook.com/v12.0/oauth/access_token",
    authorize_url="https://www.facebook.com/v12.0/dialog/oauth",
    client_kwargs={"scope": "email public_profile"},
)

# ---------------------------------------------------------------
# SIMPLE IN-MEMORY RATE LIMITER
# ---------------------------------------------------------------
_hits = defaultdict(deque)


def rate_limited(key, limit, window_seconds):
    """Return True if `key` exceeded `limit` hits in the last window."""
    now = time.time()
    bucket = _hits[key]
    while bucket and now - bucket[0] > window_seconds:
        bucket.popleft()
    if len(bucket) >= limit:
        return True
    bucket.append(now)
    return False


def client_ip():
    return request.remote_addr or "unknown"


# ---------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------
def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db


def init_db():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        existing = {
            row["name"] for row in
            db.execute("PRAGMA table_info(users)").fetchall()
        }

        additions = {
            "phone": "TEXT DEFAULT ''",
            "date_of_birth": "TEXT DEFAULT ''",
            "gender": "TEXT DEFAULT ''",
            "address": "TEXT DEFAULT ''",
            "city": "TEXT DEFAULT ''",
            "state": "TEXT DEFAULT ''",
            "pincode": "TEXT DEFAULT ''",
            "education_level": "TEXT DEFAULT ''",
            "institution": "TEXT DEFAULT ''",
            "course": "TEXT DEFAULT ''",
            "branch": "TEXT DEFAULT ''",
            "study_year": "TEXT DEFAULT ''",
            "profile_photo": "TEXT DEFAULT ''",
        }

        for column, definition in additions.items():
            if column not in existing:
                db.execute(
                    f"ALTER TABLE users ADD COLUMN {column} {definition}"
                )

        # Chat history so the AI tutor remembers the conversation
        db.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'model')),
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_user "
            "ON chat_history (user_id, id)"
        )


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    with get_db() as db:
        return db.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()


@app.context_processor
def inject_user():
    user = current_user()
    return {
        "logged_in": user is not None,
        "user_name": user["name"] if user else "",
        "nav_photo": user["profile_photo"] if user else "",
    }


# ---------------------------------------------------------------
# UPLOAD VALIDATION
# ---------------------------------------------------------------
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def is_real_image(file_storage):
    """Check the file's actual bytes, not just its extension."""
    head = file_storage.stream.read(12)
    file_storage.stream.seek(0)
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if head.startswith(b"\xff\xd8\xff"):
        return True
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return True
    return False


# ---------------------------------------------------------------
# PAGES + EMAIL/PASSWORD AUTH
# ---------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")



@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":

        # Prevent too many registration attempts
        if rate_limited(f"register:{client_ip()}", 10, 3600):
            flash("Too many attempts. Please try again later.")
            return render_template("register.html")

        # Read form values
        name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        # Temporary diagnostic: never log the actual password
        app.logger.info(
            "REGISTER DEBUG: keys=%s | name=%s | email=%s | password=%s | confirm=%s",
            sorted(request.form.keys()),
            bool(name),
            bool(email),
            bool(password),
            bool(confirm_password),
        )

        # Validate required fields
        if not all([name, email, password, confirm_password]):
            flash("Please complete all required fields.")
            return render_template("register.html")

        # Validate password length
        if len(password) < 8:
            flash("Password must be at least 8 characters.")
            return render_template("register.html")

        # Validate password confirmation
        if password != confirm_password:
            flash("Passwords do not match.")
            return render_template("register.html")

        # Create the account
        try:
            with get_db() as db:
                cursor = db.execute(
                    """
                    INSERT INTO users (name, email, password_hash)
                    VALUES (?, ?, ?)
                    """,
                    (
                        name,
                        email,
                        generate_password_hash(password),
                    ),
                )
                new_user_id = cursor.lastrowid

            # Log the user in after successful registration
            session.clear()
            session.permanent = True
            session["user_id"] = new_user_id

            flash("Registration successful!")
            return redirect(url_for("dashboard"))

        except sqlite3.IntegrityError:
            # Email already exists in the database
            flash("This email already has an account. Please log in.")
            return render_template("register.html")

        except sqlite3.Error:
            app.logger.exception("Registration database error")
            flash("Registration could not be completed. Please try again.")
            return render_template("register.html")

    return render_template("register.html")



@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        # Brute-force protection: 10 attempts per 15 minutes per IP
        if rate_limited(f"login:{client_ip()}", 10, 900):
            flash("Too many login attempts. Please wait a few minutes.")
            return render_template("login.html")

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        with get_db() as db:
            user = db.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session.permanent = True
            session["user_id"] = user["id"]
            return redirect(url_for("dashboard"))

        flash("Incorrect email or password. Please try again.")

    return render_template("login.html")


# ---------------------------------------------------------------
# GOOGLE OAUTH
# ---------------------------------------------------------------
@app.route("/google-login")
def google_login():
    redirect_uri = url_for("google_authorize", _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route("/login/google/authorized")
def google_authorize():
    try:
        token = google.authorize_access_token()
        user_info = token.get("userinfo")

        if not user_info:
            flash("Google authentication failed.")
            return redirect(url_for("login"))

        # Only trust emails that Google has verified
        if not user_info.get("email_verified", False):
            flash("Your Google email is not verified.")
            return redirect(url_for("login"))

        email = (user_info.get("email") or "").strip().lower()
        name = user_info.get("name", "Google User")

        if not email:
            flash("Google did not provide an email address.")
            return redirect(url_for("login"))

        with get_db() as db:
            user = db.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()

            if not user:
                dummy_password = generate_password_hash(secrets.token_hex(16))
                cursor = db.execute(
                    "INSERT INTO users (name, email, password_hash) "
                    "VALUES (?, ?, ?)",
                    (name, email, dummy_password),
                )
                user_id = cursor.lastrowid
            else:
                user_id = user["id"]

        session.clear()
        session.permanent = True
        session["user_id"] = user_id

        flash("Successfully logged in with Google!")
        return redirect(url_for("dashboard"))

    except Exception:
        logger.exception("Google login error")
        flash("Google login failed. Please try again.")
        return redirect(url_for("login"))


# ---------------------------------------------------------------
# FACEBOOK OAUTH
# ---------------------------------------------------------------
@app.route("/facebook-login")
def facebook_login():
    redirect_uri = url_for("facebook_authorize", _external=True)
    return facebook.authorize_redirect(redirect_uri)


@app.route("/login/facebook/authorized")
def facebook_authorize():
    try:
        facebook.authorize_access_token()
        resp = facebook.get("me?fields=id,name,email")
        user_info = resp.json()

        email = (user_info.get("email") or "").strip().lower()
        name = user_info.get("name", "Facebook User")

        if not email:
            flash("Facebook account does not provide an email address.")
            return redirect(url_for("login"))

        with get_db() as db:
            user = db.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()

            if user:
                # Facebook does not guarantee a verified email, so we never
                # auto-login into an existing account (prevents takeover).
                flash(
                    "An account with this email already exists. "
                    "Please log in with your password or Google."
                )
                return redirect(url_for("login"))

            dummy_password = generate_password_hash(secrets.token_hex(16))
            cursor = db.execute(
                "INSERT INTO users (name, email, password_hash) "
                "VALUES (?, ?, ?)",
                (name, email, dummy_password),
            )
            user_id = cursor.lastrowid

        session.clear()
        session.permanent = True
        session["user_id"] = user_id

        flash("Successfully logged in with Facebook!")
        return redirect(url_for("dashboard"))

    except Exception:
        logger.exception("Facebook login error")
        flash("Facebook login failed. Please try again.")
        return redirect(url_for("login"))


# ---------------------------------------------------------------
# PROFILE
# ---------------------------------------------------------------
@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = current_user()
    if user is None:
        session.clear()
        return redirect(url_for("login"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        phone = request.form.get("phone", "").strip()
        date_of_birth = request.form.get("date_of_birth", "")
        gender = request.form.get("gender", "")
        address = request.form.get("address", "").strip()
        city = request.form.get("city", "").strip()
        state = request.form.get("state", "").strip()
        pincode = request.form.get("pincode", "").strip()
        education_level = request.form.get("education_level", "")
        institution = request.form.get("institution", "").strip()
        course = request.form.get("course", "").strip()
        branch = request.form.get("branch", "").strip()
        study_year = request.form.get("study_year", "").strip()

        if not name:
            flash("Full name is required.")
            return render_template("profile.html", user=user)

        photo = request.files.get("profile_photo")
        photo_path = user["profile_photo"] or ""

        if photo and photo.filename:
            if not allowed_file(photo.filename):
                flash("Use a PNG, JPG, JPEG or WEBP image.")
                return render_template("profile.html", user=user)

            if not is_real_image(photo):
                flash("That file is not a valid image.")
                return render_template("profile.html", user=user)

            extension = secure_filename(photo.filename).rsplit(".", 1)[1].lower()
            filename = f"{user['id']}_{secrets.token_hex(12)}.{extension}"
            destination = UPLOAD_FOLDER / filename
            photo.save(destination)
            photo_path = f"uploads/profiles/{filename}"

        with get_db() as db:
            db.execute("""
                UPDATE users SET
                    name = ?, phone = ?, date_of_birth = ?, gender = ?,
                    address = ?, city = ?, state = ?, pincode = ?,
                    education_level = ?, institution = ?, course = ?,
                    branch = ?, study_year = ?, profile_photo = ?
                WHERE id = ?
            """, (
                name, phone, date_of_birth, gender,
                address, city, state, pincode,
                education_level, institution, course,
                branch, study_year, photo_path, user["id"],
            ))

        flash("Your profile has been saved successfully.")
        return redirect(url_for("profile"))

    return render_template("profile.html", user=user)


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=current_user())


# ---------------------------------------------------------------
# AI TUTOR
# ---------------------------------------------------------------
MAX_MESSAGE_CHARS = 2000
HISTORY_TURNS = 20  # number of recent chat messages included as context


@app.route("/tutor")
@login_required
def ai_tutor():
    return render_template("ai_tutor.html", user=current_user())


def load_history(user_id, limit=HISTORY_TURNS):
    """Load recent chat messages in OpenAI-compatible chat format."""
    with get_db() as db:
        rows = db.execute(
            "SELECT role, content FROM chat_history "
            "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()

    rows = list(reversed(rows))

    # Ensure conversation context starts with a user message.
    while rows and rows[0]["role"] not in ("user",):
        rows.pop(0)

    history = []
    for row in rows:
        role = "assistant" if row["role"] == "model" else row["role"]
        if role in ("user", "assistant"):
            history.append({"role": role, "content": row["content"]})
    return history

def save_message(user_id, role, content):
    with get_db() as db:
        db.execute(
            "INSERT INTO chat_history (user_id, role, content) "
            "VALUES (?, ?, ?)",
            (user_id, role, content),
        )



@app.route("/api/chat", methods=["POST"])
@login_required
def chat_with_tutor():
    if client is None:
        return jsonify({
            "error": "AI tutor is not configured. Add GROQ_API_KEY in Render Environment."
        }), 503

    user_id = session["user_id"]

    if rate_limited(f"chat:{user_id}", 20, 60):
        return jsonify({
            "error": "You're sending messages too fast. Please wait a moment."
        }), 429

    attachment = request.files.get("attachment")

    # Accept both normal JSON chats and uploaded files.
    if request.is_json:
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()
    else:
        user_message = (request.form.get("message") or "").strip()

    if not user_message and not attachment:
        return jsonify({"error": "Enter a message or attach a file."}), 400

    if len(user_message) > MAX_MESSAGE_CHARS:
        return jsonify({
            "error": f"Message too long (max {MAX_MESSAGE_CHARS} characters)."
        }), 400

    image_content = None

    try:
        if attachment and attachment.filename:
            filename = secure_filename(attachment.filename)
            extension = filename.rsplit(".", 1)[-1].lower()
            file_bytes = attachment.read()

            if not file_bytes:
                return jsonify({"error": "The uploaded file is empty."}), 400

            # Images: send the actual image to a vision model.
            if extension in {"png", "jpg", "jpeg", "webp"}:
                import base64

                mime_type = {
                    "png": "image/png",
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "webp": "image/webp",
                }[extension]

                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{mime_type};base64,"
                            f"{base64.b64encode(file_bytes).decode('utf-8')}"
                        )
                    },
                }

                if not user_message:
                    user_message = (
                        f"Analyze the uploaded image {filename}. "
                        "Describe its contents and answer any visible questions."
                    )

            # PDF: extract readable text.
            elif extension == "pdf":
                try:
                    pdf = PdfReader(io.BytesIO(file_bytes))
                    extracted = "\n".join(
                        page.extract_text() or "" for page in pdf.pages
                    ).strip()
                except Exception:
                    return jsonify({
                        "error": "Could not read this PDF. Please upload a valid PDF."
                    }), 400

                if not extracted:
                    return jsonify({
                        "error": "No selectable text found in this PDF. Scanned PDFs need OCR."
                    }), 400

                user_message = (
                    f"{user_message}\n\n"
                    f"Document: {filename}\n"
                    f"Extracted document text:\n{extracted[:30000]}"
                ).strip()

            # OpenDocument text files: ODT, ODS and ODP.
            elif extension in {"odt", "ods", "odp"}:
                try:
                    document = load_odf(io.BytesIO(file_bytes))
                    paragraphs = document.getElementsByType(P)
                    extracted = "\n".join(
                        teletype.extractText(p) for p in paragraphs
                    ).strip()
                except Exception:
                    return jsonify({
                        "error": "Could not read this OpenDocument file."
                    }), 400

                if not extracted:
                    return jsonify({
                        "error": "No readable text found in this document."
                    }), 400

                user_message = (
                    f"{user_message}\n\n"
                    f"Document: {filename}\n"
                    f"Extracted document text:\n{extracted[:30000]}"
                ).strip()

            else:
                return jsonify({
                    "error": "Supported formats: PNG, JPG, JPEG, WEBP, PDF, ODT, ODS and ODP."
                }), 400

        messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
        messages.extend(load_history(user_id))

        if image_content:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    image_content,
                ],
            })
            model = os.getenv(
    "GROQ_VISION_MODEL",
    "qwen/qwen3.8-27b"
)
        else:
            messages.append({"role": "user", "content": user_message})
            model = GROQ_MODEL

     response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=1.0 if image_content else 0.35,
    max_tokens=1800,
)

        reply = (response.choices[0].message.content or "").strip()
        if not reply:
            reply = "I couldn't generate a reply. Please try again."

        # Store text history, not the raw uploaded file.
        save_message(user_id, "user", user_message)
        save_message(user_id, "model", reply)

        return jsonify({"response": reply, "status": "success"})

    except Exception:
        logger.exception("Groq chat or file analysis error")
        return jsonify({
            "error": (
                "Analysis failed. Check the Render logs and confirm that "
                "the configured Groq model supports the uploaded content."
            )
        }), 500


@app.route("/api/chat/history")
@login_required
def chat_history():
    with get_db() as db:
        rows = db.execute(
            "SELECT role, content, created_at FROM chat_history "
            "WHERE user_id = ? ORDER BY id DESC LIMIT 50",
            (session["user_id"],),
        ).fetchall()
    messages = [dict(r) for r in reversed(rows)]
    return jsonify({"messages": messages})


@app.route("/api/chat/clear", methods=["POST"])
@login_required
def clear_chat():
    with get_db() as db:
        db.execute(
            "DELETE FROM chat_history WHERE user_id = ?",
            (session["user_id"],),
        )
    return jsonify({"status": "cleared"})


# ---------------------------------------------------------------
# LOGOUT + ERRORS
# ---------------------------------------------------------------
@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("You have logged out successfully.")
    return redirect(url_for("login"))


@app.errorhandler(413)
def file_too_large(error):
    flash("Photo is too large. Please choose an image under 3 MB.")
    return redirect(url_for("profile"))


init_db()

if __name__ == "__main__":
    # Debug only when FLASK_ENV=development (never in production)
    app.run(debug=IS_DEV)
