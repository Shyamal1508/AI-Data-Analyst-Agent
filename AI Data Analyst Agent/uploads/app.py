from flask import Flask, render_template, request, session
import pandas as pd
import searche  # Your custom module
from searche import search
import utils  # Your custom module
from utils import preprocess_and_save

# --- MATPLOTLIB FIX ---
# This is the crucial part.
# 1. Import matplotlib
import matplotlib
# 2. Set the backend to 'Agg' (non-interactive)
matplotlib.use('Agg')
# 3. NOW import pyplot and seaborn
import matplotlib.pyplot as plt
# ----------------------

import seaborn as sns
import numpy as np
import io
import os
from groq import Groq

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key-here")  # Change in production!

# Ensure the static directory exists for saving plots
os.makedirs("static", exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    df = None
    df_html = ""  # This is returned by your util, so we keep it
    df_preview_html = ""
    result_html = ""
    code_generated = ""
    cols = []  # Default empty list to avoid undefined vars

    # Initialize session variables
    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == "POST":
        file = request.files.get("file")
        query = request.form.get("query", "").strip()  # Strip whitespace
        groq_key = request.form.get("api_key")

        if not groq_key:
            message = "Please enter your Groq API key."
        elif not file or file.filename == '':
            message = "Please upload a file."
        else:
            # Process uploaded file
            df, cols, df_html, err = preprocess_and_save(file)
            if err:
                message = err
            else:
                # Show full DataFrame and preview
                df_preview_html = df.head().to_html(classes="table table-auto w-full", index=False)

                if query:  # Now safely checks for non-empty query
                    try:
                        # Updated prompt: Tells LLM *not* to save the plot,
                        # as our server logic will handle that.
                        prompt = f"""
You are a Python data analyst. Given a pandas DataFrame named `df` with columns: {list(cols)},

Write **only** the Python code (no explanations, no markdown) to answer this question:

Question: {query}

- Use only `pandas`, `numpy`, `matplotlib`, `seaborn` if needed.
- Store the final *data* answer (like a number, string, or DataFrame) in a variable named `result`.
- If plotting, just create the plot (e.g., `sns.histplot(df['column'])`). **DO NOT** save the plot to a file or assign the plot object to the `result` variable. The server will save the plot automatically.
"""

                        client = Groq(api_key=groq_key)
                        chat_completion = client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model="llama-3.3-70b-versatile", # Note: Check if this model name is correct/available
                            temperature=0.2,
                            max_tokens=1024
                        )
                        raw_response = chat_completion.choices[0].message.content.strip()

                        # Clean code block
                        code_generated = raw_response
                        if code_generated.startswith("```python"):
                            code_generated = code_generated[10:]
                        if code_generated.startswith("```"):
                            code_generated = code_generated[3:]
                        if code_generated.endswith("```"):
                            code_generated = code_generated[:-3]
                        code_generated = code_generated.strip()

                        # Save to session
                        session['chat_history'].append({"role": "user", "content": query}) # Save user query
                        session['chat_history'].append({"role": "assistant", "content": code_generated})
                        session.modified = True

                        # Safe execution environment
                        local_vars = {
                            "df": df.copy(),  # Prevent modification of original
                            "pd": pd,
                            "plt": plt,
                            "sns": sns,
                            "np": np,
                            "io": io
                        }

                        # Clear any previous plot from matplotlib's global state
                        plt.clf()

                        # Execute generated code
                        exec(code_generated, {}, local_vars)

                        result = local_vars.get("result")
                        plot_path = "static/result_plot.png"
                        # Add a cache-busting query param to the URL
                        plot_url = f"/{plot_path}?v={os.times().system}" 

                        # Clear old plot file if it exists
                        if os.path.exists(plot_path):
                            os.remove(plot_path)

                        # Handle the 'result' variable (data)
                        if result is not None:
                            if isinstance(result, pd.DataFrame):
                                result_html = result.to_html(classes="table table-auto w-full", index=False)
                            elif isinstance(result, (pd.Series, list, dict, str, int, float)):
                                result_html = f'<pre class="bg-gray-100 p-2 rounded">{str(result)}</pre>'
                            else:
                                result_html = "Result generated (type not directly displayable)."
                        else:
                            # Don't say "no result" if a plot was the intended output
                            if not plt.gcf().get_axes():
                                result_html = "Code executed but no `result` variable or plot was found."

                        # Handle plot generation
                        # Check if the code *actually* created a plot by seeing if there are axes
                        if 'plt' in local_vars and plt.gcf().get_axes():
                            plt.tight_layout()
                            plt.savefig(plot_path, bbox_inches='tight', dpi=150)
                            plt.close() # Close figure to free memory
                            # Add image tag to the result
                            result_html += f'<br><img src="{plot_url}" alt="Generated Plot" class="mt-4 rounded border">'
                        
                        # Clear figure again just in case
                        plt.clf()
                        plt.close()

                    except Exception as e:
                        message = f"Error executing code: {str(e)}"
                        import traceback
                        print(traceback.format_exc())
                elif file: # If file was uploaded but no query
                     message = "File loaded successfully. Enter a query to analyze the data."

    return render_template(
        "index.html",
        message=message,
        df_html=df_html,  # Full data (if your template uses it)
        df_preview_html=df_preview_html, # Head() preview
        code_generated=code_generated,
        result_html=result_html,
        chat_history=session['chat_history']
    )


@app.route("/search", methods=["GET", "POST"])
def search_route():
    message = ""
    response = ""
    
    if request.method == "POST":
        question = request.form.get("question")
        groq_key = request.form.get("api_key")
        
        if not groq_key:
            message = "Please enter your Groq API key."
        elif not question:
            message = "Please enter a question."
        else:
            try:
                response = search(question, groq_key)
            except Exception as e:
                message = f"Error processing search: {str(e)}"
                import traceback
                print(traceback.format_exc())
    
    return render_template("search.html", response=response, message=message)


if __name__ == "__main__":
    app.run(debug=True)


    