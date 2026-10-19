

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
    if not client:
        return jsonify({"error": "AI tutor is not configured right now."}), 503

    user_id = session["user_id"]

    # 20 messages per minute per user
    if rate_limited(f"chat:{user_id}", 20, 60):
        return jsonify({"error": "You're sending messages too fast. "
                                 "Please wait a moment."}), 429

    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400
    if len(user_message) > MAX_MESSAGE_CHARS:
        return jsonify({
            "error": f"Message too long (max {MAX_MESSAGE_CHARS} characters)."
        }), 400

    try:
        chat = client.chats.create(
            model=GEMINI_MODEL,
            history=load_history(user_id),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION
            ),
        )

        response = chat.send_message(user_message)
        reply = response.text

        if not reply:
            reply = (
                "I couldn't generate a reply just now. "
                "Please try asking in a different way."
            )

        save_message(user_id, "user", user_message)
        save_message(user_id, "model", reply)


        return jsonify({"response": reply, "status": "success"})

    except Exception:
        logger.exception("Gemini chat error")
        return jsonify({
            "error": "The AI tutor is unavailable right now. "
                     "Please try again shortly."
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
