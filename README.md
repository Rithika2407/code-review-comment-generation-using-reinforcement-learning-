# RL Code Review System — Setup Guide

## Project Structure (after setup)

```
your-project/
├── app.py                        ← Flask backend (VSCode)
├── requirements.txt              ← Python dependencies
├── code_review_system_local.html ← Frontend (open in browser)
├── reward_model.pt               ← ★ Downloaded from Colab
├── model_config.json             ← ★ Downloaded from Colab
└── tokenizer_config/             ← ★ Downloaded from Colab
    ├── tokenizer_config.json
    ├── vocab.json
    └── merges.txt
```

---

## STEP 1 — Train the model in Google Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook and paste the contents of `train_reward_model.py`
   - Or upload the file directly: File → Upload notebook
3. Set Runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
4. Run all cells top-to-bottom (~5–10 min on GPU)
5. After training completes, download these 3 items to your VSCode project folder:

```python
# Run this in a Colab cell to zip everything for easy download
import shutil
shutil.make_archive('model_files', 'zip', '.', 
    base_dir=None, 
    verbose=True)
# Then Files panel → right-click model_files.zip → Download
```

Or download individually from the Colab Files panel (📁 icon):
- `reward_model.pt`
- `model_config.json`
- `tokenizer_config/` folder (download as zip)

---

## STEP 2 — Set up VSCode backend

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your downloaded files in the same folder as app.py:
#    reward_model.pt
#    model_config.json
#    tokenizer_config/

# 4. Start the server
python app.py
```

You should see:
```
[Server] Loading model on cpu...
[Server] Model loaded ✅
[Server] Starting on http://localhost:5000
```

---

## STEP 3 — Run the frontend

1. Open `code_review_system_local.html` directly in your browser
   (double-click the file, or drag it into Chrome/Firefox)
2. The frontend will POST to `http://localhost:5000/analyze`
3. Paste any Python/JavaScript/Java/C++ code and click **Analyze**

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `reward_model.pt not found` | Make sure the .pt file is in the same folder as app.py |
| `CORS error` in browser | Make sure flask-cors is installed and app.py is running |
| `Connection refused` | Flask server isn't running — run `python app.py` |
| Slow inference | Normal on CPU; use GPU machine for faster results |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |

---

## How the RL pipeline works

```
Code Input
    ↓
Tokenize  →  [CODE] ... [REVIEW] ...
    ↓
CodeBERT Encoder  →  768-dim [CLS] embedding
    ↓
Generate 13–18 candidate review comments (pattern-matched + generic)
    ↓
Reward Model Head  →  scalar reward ∈ [0, 1] per candidate
    ↓
RL Selection  →  top-K candidates by reward score
    ↓
Category + Severity classifiers (auxiliary heads)
    ↓
Return structured JSON → Frontend renders results
```

The model is trained with:
- **MSE loss** on reward scores (60% weight)
- **Cross-entropy** on category classification (25% weight)  
- **Cross-entropy** on severity classification (15% weight)
- **AdamW** optimizer with linear warmup over 5 epochs on CodeBERT-base
