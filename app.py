import os
import sqlite3
import secrets
from pathlib import Path
from functools import wraps
from datetime import timedelta  # <-- Session lifetime ke liye import

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from authlib.integrations.flask_client import OAuth

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "smartlearn.db"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads" / "profiles"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
from werkzeug.middleware.proxy_fix import ProxyFix

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.secret_key = os.getenv("SECRET_KEY") or secrets.token_hex(32)

# --- Session Lifetime Setting Added ---
app.permanent_session_lifetime = timedelta(days=7) 

app.config["MAX_CONTENT_LENGTH"] = 3 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

# OAuth Setup for Google and Facebook
oauth = OAuth(app)

google = oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID', 'your_google_client_id'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET', 'your_google_client_secret'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

facebook = oauth.register(
    name='facebook',
    client_id=os.getenv('FACEBOOK_CLIENT_ID', 'your_facebook_client_id'),
    client_secret=os.getenv('FACEBOOK_CLIENT_SECRET', 'your_facebook_client_secret'),
    api_base_url='https://graph.facebook.com/v12.0/',
    access_token_url='https://graph.facebook.com/v12.0/oauth/access_token',
    authorize_url='https://www.facebook.com/v12.0/dialog/oauth',
    client_kwargs={'scope': 'email public_profile'}
)


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
            "profile_photo": "TEXT DEFAULT ''"
        }

        for column, definition in additions.items():
            if column not in existing:
                db.execute(
                    f"ALTER TABLE users ADD COLUMN {column} {definition}"
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
        "nav_photo": user["profile_photo"] if user else ""
    }


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("home"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not name or not email or not password:
            flash("Please complete all required fields.")
        elif len(password) < 8:
            flash("Password must be at least 8 characters.")
        elif password != confirm:
            flash("Passwords do not match.")
        else:
            try:
                with get_db() as db:
                    cursor = db.execute(
                        """INSERT INTO users (name, email, password_hash)
                            VALUES (?, ?, ?)""",
                        (name, email, generate_password_hash(password))
                    )
                    new_id = cursor.lastrowid

                session.clear()
                session.permanent = True  # <-- Permanent session
                session["user_id"] = new_id
                return redirect(url_for("home"))

            except sqlite3.IntegrityError:
                flash("This email already has an account. Please log in.")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("home"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        with get_db() as db:
            user = db.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session.permanent = True  # <-- Permanent session
            session["user_id"] = user["id"]
            return redirect(url_for("home"))

        flash("Incorrect email or password. Please try again.")

    return render_template("login.html")


# --- GOOGLE OAUTH ROUTES ---
@app.route('/google-login')
def google_login():
    redirect_uri = url_for('google_authorize', _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route('/login/google/authorized')
def google_authorize():
    token = google.authorize_access_token()
    user_info = token.get('userinfo')
    
    if not user_info:
        flash("Google authentication failed.")
        return redirect(url_for('login'))
    
    email = user_info.get('email')
    name = user_info.get('name', 'Google User')
    
    with get_db() as db:
        user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            dummy_password = generate_password_hash(secrets.token_hex(16))
            cursor = db.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, dummy_password)
            )
            user_id = cursor.lastrowid
        else:
            user_id = user["id"]
            
    session.clear()
    session.permanent = True  # <-- Permanent session for Google login
    session["user_id"] = user_id
    flash("Successfully logged in with Google!")
    return redirect(url_for("home"))


# --- FACEBOOK OAUTH ROUTES ---
@app.route('/facebook-login')
def facebook_login():
    redirect_uri = url_for('facebook_authorize', _external=True)
    return facebook.authorize_redirect(redirect_uri)


@app.route('/login/facebook/authorized')
def facebook_authorize():
    token = facebook.authorize_access_token()
    resp = facebook.get('me?fields=id,name,email')
    user_info = resp.json()
    
    email = user_info.get('email')
    name = user_info.get('name', 'Facebook User')
    
    if not email:
        flash("Facebook account does not provide an email address.")
        return redirect(url_for('login'))
        
    with get_db() as db:
        user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            dummy_password = generate_password_hash(secrets.token_hex(16))
            cursor = db.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, dummy_password)
            )
            user_id = cursor.lastrowid
        else:
            user_id = user["id"]
            
    session.clear()
    session.permanent = True  # <-- Permanent session for Facebook login
    session["user_id"] = user_id
    flash("Successfully logged in with Facebook!")
    return redirect(url_for("home"))


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
                branch, study_year, photo_path, user["id"]
            ))

        flash("Your profile has been saved successfully.")
        return redirect(url_for("profile"))

    return render_template("profile.html", user=user)


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=current_user())


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
    app.run(debug=True)
