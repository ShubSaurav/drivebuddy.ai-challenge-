"""
Vercel Serverless Application: Shubham Saurav — DrivebuddyAI ML Data Pre-Processing Challenge
Flask web application and API endpoint for Vercel deployment.
Exports top-level `app` variable for Vercel WSGI runtime.
"""

import os
import sys

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from flask import Flask, request, jsonify, render_template_string
from src.predict import predict_post
from src.preprocessing import preprocess_text
from src.features import PAV_BHAJI_LEXICON, OTHER_FOODS_LEXICON

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shubham Saurav — DrivebuddyAI ML Challenge</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background: #0B132B; color: #E2E8F0; min-height: 100vh; padding: 2rem 1rem; }
        .container { max-width: 1100px; margin: 0 auto; }
        .hero {
            background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
            padding: 2.5rem; border-radius: 18px; color: white;
            box-shadow: 0 12px 30px rgba(0,0,0,0.3); border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 2rem;
        }
        .hero-tag {
            background: rgba(255, 255, 255, 0.15); color: #FFD166; padding: 0.35rem 0.95rem;
            border-radius: 20px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.6px; display: inline-block; margin-bottom: 0.8rem; border: 1px solid rgba(255, 209, 102, 0.35);
        }
        .hero h1 { font-size: 2.2rem; font-weight: 800; line-height: 1.2; margin-bottom: 0.6rem; }
        .hero p { color: #CBD5E1; font-size: 1rem; line-height: 1.5; max-width: 900px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
        .stat-card {
            background: #1C2541; padding: 1.2rem; border-radius: 14px; border: 1px solid #3A506B; text-align: center;
        }
        .stat-val { font-size: 1.8rem; font-weight: 800; color: #6FFFE9; }
        .stat-lbl { font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; margin-top: 0.3rem; }
        
        .card { background: #1C2541; border: 1px solid #3A506B; border-radius: 16px; padding: 2rem; margin-bottom: 2rem; }
        .presets { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.5rem; }
        .preset-btn {
            background: #2D3748; color: #E2E8F0; border: 1px solid #4A5568; padding: 0.45rem 0.9rem;
            border-radius: 20px; font-size: 0.82rem; font-weight: 600; cursor: pointer; transition: all 0.2s;
        }
        .preset-btn:hover { background: #4A5568; border-color: #6FFFE9; color: white; }
        label { display: block; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.4rem; color: #CBD5E1; }
        textarea, input[type="text"] {
            width: 100%; background: #0B132B; border: 1px solid #3A506B; border-radius: 10px; padding: 0.8rem;
            color: white; font-size: 0.95rem; margin-bottom: 1.2rem; outline: none; transition: border 0.2s;
        }
        textarea:focus, input[type="text"]:focus { border-color: #6FFFE9; }
        .classify-btn {
            background: linear-gradient(135deg, #059669 0%, #10B981 100%); color: white; border: none;
            padding: 0.9rem 2rem; border-radius: 12px; font-size: 1.05rem; font-weight: 700; cursor: pointer;
            width: 100%; transition: all 0.2s; box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4);
        }
        .classify-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(16, 185, 129, 0.5); }
        
        .result-box { display: none; background: #0B132B; border: 1px solid #3A506B; border-radius: 14px; padding: 1.5rem; margin-top: 1.5rem; }
        .badge { display: inline-block; padding: 0.6rem 1.4rem; border-radius: 30px; font-weight: 800; font-size: 1.25rem; margin-bottom: 1rem; }
        .badge-pb { background: linear-gradient(135deg, #10B981, #059669); color: white; }
        .badge-not-pb { background: linear-gradient(135deg, #3B82F6, #1D4ED8); color: white; }
        
        .progress-bar { height: 12px; background: #2D3748; border-radius: 6px; overflow: hidden; margin: 0.8rem 0; }
        .progress-fill { height: 100%; background: #10B981; transition: width 0.4s ease; }
        
        .chip { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 6px; font-size: 0.8rem; font-weight: 600; margin: 0.2rem; }
        .chip-pos { background: #064E3B; color: #6EE7B7; border: 1px solid #059669; }
        .chip-neg { background: #1E3A8A; color: #93C5FD; border: 1px solid #3B82F6; }
        
        .footer { text-align: center; color: #64748B; font-size: 0.85rem; margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #1E293B; }
    </style>
</head>
<body>
    <div class="container">
        <div class="hero">
            <div class="hero-tag">Vercel Deployment • DrivebuddyAI Submission</div>
            <h1>Shubham Saurav — DrivebuddyAI ML Challenge</h1>
            <p>Production-Grade Zero-Vision NLP Classification System detecting whether Instagram posts represent <b>Pav Bhaji</b> using captions and hashtags without looking at images.</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card"><div class="stat-val">1,500</div><div class="stat-lbl">Scraped Posts</div></div>
            <div class="stat-card"><div class="stat-val">452</div><div class="stat-lbl">Ground-Truth Labeled</div></div>
            <div class="stat-card"><div class="stat-val">99.5%</div><div class="stat-lbl">#pavbhaji Leakage Rate</div></div>
            <div class="stat-card"><div class="stat-val">70.3%</div><div class="stat-lbl">Holdout Accuracy</div></div>
            <div class="stat-card"><div class="stat-val">78.4%</div><div class="stat-lbl">Positive Recall</div></div>
        </div>

        <div class="card">
            <h2 style="font-size: 1.3rem; margin-bottom: 1rem; color: #F8FAFC;">⚡ Live Text Classifier</h2>
            <div class="presets">
                <span style="font-size: 0.85rem; color: #94A3B8; align-self: center; margin-right: 0.4rem;">Quick Presets:</span>
                <button class="preset-btn" onclick="setPreset(1)">Authentic Pav Bhaji</button>
                <button class="preset-btn" onclick="setPreset(2)">Mumbai Khaugalli</button>
                <button class="preset-btn" onclick="setPreset(3)">Pani Puri (Spam Tag)</button>
                <button class="preset-btn" onclick="setPreset(4)">Chicken Tikka (Spam Tag)</button>
                <button class="preset-btn" onclick="setPreset(5)">Vada Pav</button>
            </div>

            <label for="desc">📝 Instagram Caption / Description:</label>
            <textarea id="desc" rows="4" placeholder="Enter post caption..."></textarea>

            <label for="tags">🏷️ Hashtags (Tags):</label>
            <input type="text" id="tags" placeholder="#foodie #streetfood #pavbhaji ...">

            <button class="classify-btn" onclick="classifyPost()">⚡ Classify Post Now (Text-Only)</button>

            <div id="resultBox" class="result-box">
                <div id="badgeContainer"></div>
                <div style="display: flex; justify-content: space-between; font-weight: 600; font-size: 0.9rem;">
                    <span>Pav Bhaji Probability: <span id="probVal">0%</span></span>
                    <span>Confidence: <span id="confVal">0%</span></span>
                </div>
                <div class="progress-bar">
                    <div id="progressFill" class="progress-fill" style="width: 0%;"></div>
                </div>
                <div style="margin-top: 1rem;">
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.3rem;">Sanitized Cleaned Text:</div>
                    <code id="cleanText" style="display: block; background: #111827; padding: 0.6rem; border-radius: 8px; font-size: 0.85rem; color: #F3F4F6;"></code>
                </div>
                <div id="signalsContainer" style="margin-top: 1rem;"></div>
            </div>
        </div>

        <div class="footer">
            Shubham Saurav &bull; DrivebuddyAI ML Data Pre-Processing Challenge &bull; Deployed on Vercel
        </div>
    </div>

    <script>
        const presets = {
            1: {
                desc: "Hello frandz, pav bhaji khaalo! Garam hai, ye achhi thi. At Vega pure vegetarian, CP. Tag your food fanatic friend!",
                tags: "#pavbhaji #foodgram #foodphotography #delhifoodie #mumbai #foodie #butterpavbhaji"
            },
            2: {
                desc: "Pav Bhaji at Mazgaon Khaugalli! Devoured this piping hot Pav Bhaji at the little known street lane. Lots of stalls with spicy buttery food.",
                tags: "#mumbai #mumbaifoodie #pavbhaji #butterpavbhaji #khaugalli #streetfoods #desifood"
            },
            3: {
                desc: "Such a colourful Sprouts Chaat and crispy golgappa with tangy mint water! Must have evening snacks in Delhi street stall.",
                tags: "#samosa #thali #panipuri #golgappa #chaat #streetfood #pavbhaji #kolkata"
            },
            4: {
                desc: "Chicken Tikka pieces grilled over charcoal! Tender juicy kebabs served with spicy mint chutney.",
                tags: "#chickentikka #chicken #delhifoodie #streetfood #pavbhaji #nonveg #delhieater"
            },
            5: {
                desc: "Hot fried potato dumpling inside soft pav bread served with fried green chilies and dry garlic chutney!",
                tags: "#vadapav #mumbaistreetfood #streetfood #pavbhaji #delhifoodie #desifood"
            }
        };

        function setPreset(id) {
            document.getElementById('desc').value = presets[id].desc;
            document.getElementById('tags').value = presets[id].tags;
            classifyPost();
        }

        async function classifyPost() {
            const desc = document.getElementById('desc').value;
            const tags = document.getElementById('tags').value;
            if (!desc && !tags) return;

            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ description: desc, hashtags: tags })
            });
            const data = await res.json();

            document.getElementById('resultBox').style.display = 'block';
            const badgeContainer = document.getElementById('badgeContainer');
            if (data.class_label === 1) {
                badgeContainer.innerHTML = '<span class="badge badge-pb">🍛 PAV BHAJI (Class 1)</span>';
            } else {
                badgeContainer.innerHTML = '<span class="badge badge-not-pb">❌ NOT PAV BHAJI (Class 0)</span>';
            }

            document.getElementById('probVal').innerText = (data.probability_pavbhaji * 100).toFixed(1) + '%';
            document.getElementById('confVal').innerText = data.confidence.toFixed(1) + '%';
            document.getElementById('progressFill').style.width = (data.probability_pavbhaji * 100) + '%';
            document.getElementById('cleanText').innerText = data.cleaned_text || '(Empty)';

            let sigHtml = '';
            if (data.positive_signals && data.positive_signals.length) {
                sigHtml += '<div style="margin-bottom: 0.5rem;"><b style="color: #6EE7B7;">🟢 Supporting Signals:</b> ' +
                    data.positive_signals.map(s => `<span class="chip chip-pos">+${s}</span>`).join(' ') + '</div>';
            }
            if (data.negative_signals && data.negative_signals.length) {
                sigHtml += '<div><b style="color: #93C5FD;">🔵 Competing Dish Signals:</b> ' +
                    data.negative_signals.map(s => `<span class="chip chip-neg">-${s}</span>`).join(' ') + '</div>';
            }
            document.getElementById('signalsContainer').innerHTML = sigHtml;
        }

        // Auto run default preset
        setPreset(1);
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json() or {}
    desc = data.get("description", "")
    tags = data.get("hashtags", "")
    comments = data.get("comments", "")
    remove_leakage = data.get("remove_leakage", True)

    result = predict_post(
        description=desc,
        hashtags=tags,
        comments=comments,
        remove_leakage=remove_leakage
    )

    cleaned = result.get("cleaned_text", "")
    words = set(cleaned.split())
    pos_hits = [w for w in sorted(words) if any(kw in w for kw in PAV_BHAJI_LEXICON)]
    neg_hits = [w for w in sorted(words) if any(kw in w for kw in OTHER_FOODS_LEXICON)]

    result["positive_signals"] = pos_hits[:8]
    result["negative_signals"] = neg_hits[:8]
    return jsonify(result)

# Top-level exports for Vercel WSGI / ASGI
handler = app
application = app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
