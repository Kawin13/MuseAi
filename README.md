# MuseAI — AI Music Generator

An end-to-end LSTM-based music generation system.  
Train on MIDI datasets → generate new melodies → download as `.mid`.

---

## Folder Structure

```
music-generator/
├── backend/
│   ├── app.py            Flask REST API
│   ├── train.py          LSTM training script
│   ├── generate.py       Music generation script
│   ├── preprocess.py     MIDI parsing & vocabulary
│   ├── load_dataset.py   Dataset downloader / synthetic fallback
│   ├── config.py         Central configuration
│   ├── requirements.txt
│   ├── Procfile          gunicorn entry-point (Render)
│   ├── runtime.txt       Python 3.11 pin (Render)
│   ├── models/           Saved model & pickle files
│   ├── generated/        Output MIDI files
│   └── dataset/          MIDI training data
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── config.js         ← change BACKEND_URL here for deployment
├── render.yaml
├── .gitignore
└── README.md
```

---

## 1 · Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11.x |
| pip | ≥ 23 |
| Git | any |

> **Windows users:** Python 3.11 from python.org works fine.  
> A GPU is **not** required — training runs on CPU.

---

## 2 · Installation

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/music-generator.git
cd music-generator/backend

# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 3 · Dataset Setup

The loader will first try to download a public-domain MIDI pack.
If the network is unavailable it automatically generates a synthetic dataset.

```bash
cd backend
python load_dataset.py
```

Expected output:
```
✓ Dataset ready: 40 MIDI file(s) in backend/dataset/
```

You can also drop your own `.mid` files directly into `backend/dataset/`.

**Recommended free datasets (manual download):**

| Dataset | URL |
|---------|-----|
| MAESTRO v3 | https://magenta.tensorflow.org/datasets/maestro |
| Lakh MIDI | https://colinraffel.com/projects/lmd/ |
| Classical Piano MIDI | http://www.piano-midi.de/ |

---

## 4 · Preprocessing

Parses MIDI files → extracts note/chord sequences → saves `notes.pkl` and `mapping.pkl`.

```bash
python preprocess.py
```

This runs automatically when you call `train.py` if the pickle files are missing.

---

## 5 · Training

```bash
# Fresh training (50 epochs by default)
python train.py

# Custom epoch count
python train.py --epochs 30

# Resume from last checkpoint
python train.py --resume
```

Model saved to `backend/models/music_model.keras`.

**Expected duration:**
- Synthetic dataset (40 files): ~5 min on a modern laptop CPU
- MAESTRO full dataset: 30–90 min

---

## 6 · Generating Music

```bash
# Default: 500 notes, temperature 1.0
python generate.py

# Custom parameters
python generate.py --length 300 --temperature 0.8 --seed 42

# Specify output filename
python generate.py --output my_song.mid
```

Output saved to `backend/generated/`.

**Temperature guide:**
- `0.1 – 0.5` → conservative, repetitive
- `0.8 – 1.2` → balanced (recommended)
- `1.5 – 2.0` → wild, experimental

---

## 7 · Running the Flask Server

```bash
cd backend
python app.py
```

The API listens on `http://localhost:5000`.

Open `frontend/index.html` in your browser (double-click or use Live Server in VS Code).

---

## 8 · API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check + model status |
| POST | `/train` | Start training `{"epochs": 50}` |
| GET | `/train/status` | Polling endpoint for training progress |
| POST | `/generate` | Generate music `{"length": 500, "temperature": 1.0, "seed": null}` |
| GET | `/download?filename=x.mid` | Download a MIDI file |
| GET | `/songs` | List all generated songs |

---

## 9 · Deploying on Render (Free Tier)

> **Important:** Do NOT train on Render — free instances have 512 MB RAM.  
> Train locally, commit the model file, then deploy.

### Step-by-step

**9.1 Train locally and commit the model:**

```bash
# After training, un-ignore the model files:
# Edit .gitignore and remove these two lines:
#   backend/models/*.pkl
#   backend/models/*.keras
git add backend/models/
git commit -m "Add pre-trained model"
git push
```

**9.2 Create a Render account**

https://render.com → sign up free

**9.3 Connect your GitHub repository**

- New → Web Service → connect repo
- Render auto-detects `render.yaml` — confirm settings

**9.4 Set environment variables (optional)**

In the Render dashboard → Environment:
```
SECRET_KEY=your-random-secret
```

**9.5 Update frontend config**

Edit `frontend/config.js`:
```js
const CONFIG = {
  BACKEND_URL: "https://museai-music-generator.onrender.com",
};
```

Push the change; Render redeploys automatically.

**9.6 Host the frontend**

- Render Static Site → connect same repo → root `frontend/` → publish.
- Or use GitHub Pages / Netlify (drag-drop the `frontend/` folder).

---

## 10 · Troubleshooting

| Problem | Fix |
|---------|-----|
| `No MIDI files found` | Run `python load_dataset.py` |
| `No trained model found` | Run `python train.py` first |
| `music21` import error | `pip install music21` in your venv |
| TensorFlow not installing | Ensure Python 3.11; try `pip install tensorflow-cpu==2.16.1` |
| Render deploy OOM | Make sure you committed a pre-trained model and `/generate` uses it |
| CORS error in browser | Check `BACKEND_URL` in `frontend/config.js` matches your server |
| Port already in use | `set PORT=5001` (Windows) or `export PORT=5001` then re-run |

---

## 11 · Architecture

```
MIDI Files
    │
    ▼
preprocess.py  ──→  notes.pkl
                    mapping.pkl
    │
    ▼
train.py
  LSTM(256) → Dropout → LSTM(256) → Dense(256) → Softmax
    │
    ▼
music_model.keras
    │
    ▼
generate.py  (temperature sampling)
    │
    ▼
generated/*.mid  ──→  Flask API  ──→  Browser UI
```

---

## License

MIT — free to use, modify, and distribute.
