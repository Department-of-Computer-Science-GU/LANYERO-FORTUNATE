from flask import Flask, request, render_template_string
import re
from model import predict_phishing_percentage

app = Flask(__name__)

# Replace this with your trained model logic
# Expected return: probability between 0 and 100

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phishing Detection Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }

        body {
            min-height: 100vh;
            background: linear-gradient(135deg, #0f172a, #1d4ed8);
            padding: 30px;
        }

        .wrapper {
            max-width: 1150px;
            margin: 0 auto;
        }

        .header {
            color: white;
            text-align: center;
            margin-bottom: 25px;
        }

        .header h1 {
            font-size: 38px;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 16px;
            opacity: 0.95;
        }

        .dashboard {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 24px;
        }

        .card {
            background: white;
            border-radius: 20px;
            padding: 28px;
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.18);
        }

        .card h2 {
            color: #0f172a;
            margin-bottom: 16px;
            font-size: 24px;
        }

        .description {
            color: #475569;
            line-height: 1.7;
            margin-bottom: 20px;
        }

        .label {
            display: block;
            margin-bottom: 10px;
            font-weight: bold;
            color: #1e293b;
        }

        .input-box {
            width: 100%;
            padding: 15px;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            font-size: 16px;
            margin-bottom: 16px;
            outline: none;
        }

        .input-box:focus {
            border-color: #2563eb;
            box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.12);
        }

        .btn {
            width: 100%;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 12px;
            padding: 15px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s ease;
        }

        .btn:hover {
            background: #1d4ed8;
        }

        .result-box {
            margin-top: 22px;
            padding: 18px;
            border-radius: 14px;
            font-size: 17px;
            font-weight: bold;
            text-align: center;
        }

        .danger {
            background: #fee2e2;
            color: #b91c1c;
            border: 1px solid #fecaca;
        }

        .safe {
            background: #dcfce7;
            color: #166534;
            border: 1px solid #bbf7d0;
        }

        .metrics {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 14px;
            margin-top: 20px;
        }

        .metric {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 16px;
            text-align: center;
        }

        .metric .value {
            font-size: 26px;
            font-weight: bold;
            color: #0f172a;
        }

        .metric .name {
            margin-top: 6px;
            color: #64748b;
            font-size: 14px;
        }

        .tips {
            margin-top: 24px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 18px;
        }

        .tips h3 {
            margin-bottom: 10px;
            color: #0f172a;
        }

        .tips ul {
            padding-left: 20px;
            color: #334155;
            line-height: 1.8;
        }

        .chart-wrap {
            position: relative;
            height: 340px;
            margin-top: 10px;
        }

        .small-note {
            text-align: center;
            color: #64748b;
            font-size: 13px;
            margin-top: 16px;
        }

        @media (max-width: 900px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="wrapper">
        <div class="header">
            <h1>Phishing URL Detection Dashboard</h1>
            <p>Analyze a web link and view phishing probability with a clear percentage chart.</p>
        </div>

        <div class="dashboard">
            <div class="card">
                <h2>URL Scanner</h2>
                <p class="description">
                    Paste a URL below and click the button to predict whether the link is likely to be phishing or legitimate.
                </p>

                <form method="POST">
                    <label class="label" for="url">Enter URL</label>
                    <input
                        class="input-box"
                        type="text"
                        id="url"
                        name="url"
                        placeholder="Example: http://secure-login-update.com/account"
                        value="{{ url if url else '' }}"
                        required
                    >
                    <button class="btn" type="submit">Analyze URL</button>
                </form>

                {% if checked %}
                    <div class="result-box {{ result_class }}">
                        {{ result_text }}
                    </div>

                    <div class="metrics">
                        <div class="metric">
                            <div class="value">{{ phishing_percent }}%</div>
                            <div class="name">Phishing Probability</div>
                        </div>
                        <div class="metric">
                            <div class="value">{{ legit_percent }}%</div>
                            <div class="name">Legitimate Probability</div>
                        </div>
                    </div>
                {% endif %}

                <div class="tips">
                    <h3>Common phishing signs</h3>
                    <ul>
                        <li>Unexpected login or verification requests</li>
                        <li>Strange characters, many dashes, or very long URLs</li>
                        <li>Domains pretending to be trusted banks or services</li>
                        <li>Links using urgent words like update, secure, or confirm</li>
                    </ul>
                </div>
            </div>

            <div class="card">
                <h2>Detection Percentage Graph</h2>
                <div class="chart-wrap">
                    <canvas id="resultChart"></canvas>
                </div>
                <div class="small-note">
                    The chart compares phishing probability against legitimate probability.
                </div>
            </div>
        </div>
    </div>

    <script>
        const phishingPercent = {{ phishing_percent|default(0) }};
        const legitPercent = {{ legit_percent|default(0) }};

        const ctx = document.getElementById('resultChart').getContext('2d');
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Phishing %', 'Legitimate %'],
                datasets: [{
                    data: [phishingPercent, legitPercent],
                    backgroundColor: ['#dc2626', '#16a34a'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.raw + '%';
                            }
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    url = ''
    checked = False
    phishing_percent = 0
    legit_percent = 0
    accuracy = 0
    prediction = ''
    result_text = ''
    result_class = 'safe'

    if request.method == 'POST':
        url = request.form.get('url', '').strip()
        checked = True

        phishing_percent, legit_percent, prediction, accuracy = predict_phishing_percentage(url)

        phishing_percent = round(phishing_percent, 2)
        legit_percent = round(legit_percent, 2)
        accuracy_percent = round(accuracy * 100, 2)

        if prediction == 'spam':
            result_text = f'Warning: This URL is likely phishing with {phishing_percent}% probability.'
            result_class = 'danger'
        else:
            result_text = f'This URL appears legitimate with {legit_percent}% confidence.'
            result_class = 'safe'
    else:
        accuracy_percent = 0

    return render_template_string(
        TEMPLATE,
        url=url,
        checked=checked,
        phishing_percent=phishing_percent,
        legit_percent=legit_percent,
        result_text=result_text,
        result_class=result_class,
        accuracy_percent=accuracy_percent,
    )

if __name__ == '__main__':
    app.run(debug=True)