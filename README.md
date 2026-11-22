# GenAI & LLM Engineering — SCFP125030

**Repository:** `GenAI_LLM_Engineering_SCFP125030`  
**GitHub owner:** [Virajkandikatla](https://github.com/Virajkandikatla)  
**Project ID:** `SCFP125030`

This is a personalized, cleaned, runnable-oriented package of the uploaded course notebooks. It retains all **24 original notebooks**, includes equivalent **24 Python scripts** in `scripts/`, adds dependency files, a simulated IMDb CSV for demos, and a clearly labeled PDF example for the research assistant.

## Quick start — Windows PowerShell

Use **Python 3.11** if possible. Open PowerShell in the extracted project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
python validate_project.py
jupyter lab
```

If `py -3.11` is unavailable, install Python 3.11, or use `py -3 -m venv .venv` with a supported Python version. If PowerShell blocks activation, execute `.\.venv\Scripts\python.exe -m pip ...` and `.\.venv\Scripts\python.exe -m jupyter lab` instead. Do not commit your `.env` to GitHub.

### Run directly as Python

```powershell
python scripts/Lab3/exercise2.py
python scripts/Lab5/exercise1.py
python scripts/Lab6/exercise1.py
```

For activities (training), open notebooks in JupyterLab; or use:

```powershell
$env:EPOCHS="1"   # fast demo check; default 10
python scripts/Activity_01/rnn_text_classification.py
python "scripts/Activity_02/Text Classification with LSTM.py"
```

The small `data/IMDB_sample.csv` consists of synthetic examples for verifying code execution, **not** the real IMDb dataset; accuracy from it is not meaningful. For real results, download a properly licensed full IMDb reviews CSV and place it at `data/IMDB Dataset.csv`. Notebook code prioritizes that file. The training code will use up to 3,000 reviews.

## API setup

1. Copy `.env.example` to `.env`.
2. Put your own `GROQ_API_KEY` in `.env` to run **Lab3, Lab5, Lab6, and the RAG assistant**.
3. `Lab5/exercise4.ipynb` additionally needs `WEATHER_API_KEY` from WeatherAPI.com. API forecast date restrictions apply.
4. Use synthetic input when experimenting with medical, support, hiring, or resume examples. Do not send real private records without permission.

The notebook models are `openai/gpt-oss-20b` and `openai/gpt-oss-120b` on Groq. Internet access, available API quotas, valid keys, and model access are required. They cannot be fully tested offline.

## Research Paper Assistant

For RAG install the extra dependencies:

```powershell
pip install -r requirements-rag.txt
jupyter lab
```

Open `RAG-2/Research Paper Assistant.ipynb`. It discovers PDFs in `RAG-2/ResearchPapers/`, including the original *demo* handout. Add your own PDFs there for actual research. `USE_DOCLING=0` (default) uses lightweight PyMuPDF extraction; use `pip install -r requirements-docling.txt` and set `USE_DOCLING=1` if you specifically want the original Docling-based parser (larger downloads).

First execution of embeddings downloads a sentence-transformers model from Hugging Face. Subsequent runs cache it locally. The `rag_db` vector store is automatically generated and is ignored by Git.

## Included folders

- `Activity_01/` — RNN sentiment classification with PyTorch
- `Activity_02/` — LSTM sentiment classification with PyTorch
- `Lab3/` — 6 direct Groq prompting exercises
- `Lab5/` — 6 LangChain chains and structured-output exercises
- `Lab6/` — 9 tool-calling exercises with a full multi-round tool loop
- `RAG-2/` — 1 research PDF assistant
- `scripts/` — stand-alone `.py` exports for all notebooks
- `data/` — synthetic sample training dataset
- `validate_project.py` — local/offline structural checks

## Verification scope

All notebook Python sources and script exports are syntax-checked by `validate_project.py`; the synthetic CSV and demo PDF are validated as required assets. This does **not** guarantee live API execution, package compatibility on every computer, learned model quality, or future third-party service uptime. Run the notebooks with your own credentials to verify live Groq/WeatherAPI responses.

## Upload to GitHub

To keep the repository you already own, extract the ZIP, then use Git from inside the extracted folder. See `GITHUB_UPLOAD.md` for exact steps. The ZIP itself is not a Git repository and does not rename your remote GitHub repository automatically.
