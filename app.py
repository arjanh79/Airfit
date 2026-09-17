from flask import Flask, render_template_string, request, jsonify

from PUSHUPS.pushups_workout import Workout

app = Flask(__name__)


def create_workout():
    reps = Workout().optimize()[0].tolist()
    reps = [f'{i:02d}' for i in reps]


    EXERCISES = [
        "Step Ups",
        "Push Ups",
        "Bent Over Rows",
        "Ab Crunches",
        "Clean and Press",
        "Step Ups",
        "Push Ups",
        "Bent Over Rows",
        "Ab Crunches",
        "Clean and Press",
        "Step Ups",
    ]

    WEIGHTS = [12, 0, 24, 6, 24, 12, 0, 24, 6, 24, 12]

    return [
        {
            "name": name,
            "reps": rep,
            "weight": weight
        }
        for name, rep, weight in zip(EXERCISES, reps, WEIGHTS)
    ]


HTML = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <title>Workout of the Day</title>

    <style>

        body {
            font-family: Arial, sans-serif;
            background: #0F2027;
            color: #A1B5BD;
            margin: 0;
            padding: 40px;
        }

        .container {
            max-width: 700px;
            margin: auto;
        }

        h1 {
            text-align: center;
            margin-bottom: 40px;
        }

        .exercise {
            background: #182A30;
            border: 1px solid #788083;
            border-radius: 12px;
            padding: 16px 50px;
            margin-bottom: 12px;
            cursor: pointer;
            transition: 0.2s;
        }

        .exercise:hover {
            transform: scale(1.02);
        }

        .exercise[data-state="completed"] {
            background: #157529;
        }

        .exercise[data-state="failed"] {
            background: #751B1B;
        }

        .exercise-name {
            display: grid;
            grid-template-columns: 100px 1fr 150px;
            align-items: center;
    
            font-size: 22px;
            font-weight: bold;
        }
        
        .reps {
            text-align: left;
        }
        
        .name {
            text-align: left;
        }
        
        .weight {
            text-align: left;
        }

        .details {
            color: #D0E0E5;
        }

        .finish-button {
            width: 100%;
            padding: 18px;
            margin-top: 30px;
            margin-bottom: 50px;
            border: none;
            border-radius: 12px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
        }

        .finish-button:hover {
            opacity: 0.9;
        }

    </style>
</head>

<body>

<div class="container">

    <h1>Workout of the Day</h1>

    {% for exercise in exercises %}

    <div
        class="exercise"
        data-state="todo"
        data-reps="{{ exercise.reps }}"
        data-exercise="{{ exercise.name }}"
        data-seq="{{ loop.index }}"
        onclick="cycleState(this)"
    >

        <div class="exercise-name">
            <span class="reps">{{ exercise.reps }}</span>
            <span class="name">{{ exercise.name }}</span>
            <span class="weight">
            {% if exercise.weight %}
                {{ exercise.weight }} kg
        {% endif %}
    </span>
</div>

    </div>

    {% endfor %}

    <button
        class="finish-button"
        onclick="finishWorkout()"
    >
        Finish Workout
    </button>

</div>


<script>

function cycleState(element) {

    const currentState = element.dataset.state;

    if (currentState === "todo") {
        element.dataset.state = "completed";
    }
    else if (currentState === "completed") {
        element.dataset.state = "failed";
    }
    else {
        element.dataset.state = "todo";
    }

}


function finishWorkout() {

    const elements = document.querySelectorAll(".exercise");
    const workout = [];

    elements.forEach(element => {

        workout.push({
            seq: Number(element.dataset.seq),
            reps: Number(element.dataset.reps),
            state: element.dataset.state
        });

    });


    fetch("/finish-workout", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify(workout)

    })
    .then(response => response.json())
    .then(data => {

        if (data.success) {
            alert("Workout saved!");
        }

    });

}

</script>

</body>
</html>
"""


@app.route("/")
def home():
    exercises = create_workout()

    return render_template_string(
        HTML,
        exercises=exercises
    )


@app.route("/finish-workout", methods=["POST"])
def finish_workout():

    workout = request.get_json()
    result = [[exercise['reps'], 1 if exercise['state'] == 'completed' else 0] for exercise in workout]
    reps, completed = zip(*result)
    result = reps + completed


    with open("PUSHUPS/progress.txt", "a") as f:
        f.write(", ".join(map(str, result)) + "\n")

    return jsonify(success=True)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")