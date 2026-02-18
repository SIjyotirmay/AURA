from flask import Flask, render_template, request
from graph import workflow  # compiled LangGraph pipeline

app = Flask(__name__)


# -----------------------------
# Home Page (UI Form)
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# -----------------------------
# Search Route (Runs Agent Pipeline)
# -----------------------------
@app.route("/search", methods=["POST"])
def search():
    # Get data safely from form
    name = request.form.get("name", "")
    skills_input = request.form.get("skills", "")
    preferred_role = request.form.get("preferred_role", "")
    location = request.form.get("location", "India")

    # Convert skills string → list (important for Node pipeline)
    skills = [s.strip() for s in skills_input.split(",") if s.strip()]

    # Initial state (LangGraph compatible)
    initial_state = {
        "user_profile": {
            "name": name,
            "skills": skills,
            "preferred_role": preferred_role,
            "location": location
        },
        "queries": [],
        "raw_jobs": [],
        "scored_jobs": []
    }

    try:
        # Run your Agentic AI pipeline (Node 1 → Node 5)
        final_state = workflow.invoke(initial_state)

        # Get scored jobs safely
        jobs = final_state.get("scored_jobs", [])

    except Exception as e:
        print("Pipeline Error:", e)
        jobs = []

    # Render results UI
    return render_template(
        "results.html",
        jobs=jobs,
        name=name,
        total_jobs=len(jobs)
    )


# -----------------------------
# Run Flask App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)

