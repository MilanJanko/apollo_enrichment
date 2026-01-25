Apollo Company Enrichment

A Streamlit-based internal tool for enriching companies with one high-quality decision-maker contact using the Apollo API.
The app is designed for safe, resumable, credit-aware enrichment and outputs a CSV where enriched contact data is appended back into the original file.

What it does

Input

CSV or Excel file
Column containing company websites

Output

Same file structure as input
New columns appended with Apollo enrichment data:

Contact name
Title
Email
LinkedIn
Company name
Employee count
Enrichment status & notes
Enrichment flow

Organization Search (paid, cached)
POST /api/v1/mixed_companies/search

Validates company exists in Apollo

Cached by domain to reduce credit usage

People API Search (free)
POST /api/v1/mixed_people/api_search

Finds potential decision-makers
Filtered by seniority
Deterministic scoring to select the best candidate

People Enrichment (paid)
POST /api/v1/people/match

Enriches the selected person
Returns work email and LinkedIn (when available)

Key features

Streamlit UI (no CLI required)
Upload CSV or Excel files
Select website column
Select row range (e.g. 10–50)
Progress tracking
Automatic checkpointing (resume after crash)
Retry failed rows only
Organization-search cache (credit-saving)
Enrichment appended to original data (safe for re-import)
Auto-save to server-side storage
Manual download with custom filename

Project structure
.
├── app.py                  # Streamlit application
├── src/
│   ├── pipeline.py         # Enrichment orchestration
│   ├── apollo_client.py
│   ├── apollo_org_search.py
│   ├── apollo_search.py
│   ├── apollo_enrich.py
│   ├── decision_maker.py
│   ├── domain_utils.py
│   └── org_cache.py
├── data/
│   ├── uploads/            # Uploaded input files (gitignored)
│   ├── outputs/            # Enriched outputs + checkpoints (gitignored)
│   └── cache/              # Org search cache (gitignored)
├── requirements.txt
└── README.md

Setup
1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

2. Install dependencies
pip install -r requirements.txt

3. Environment variables

Create a .env file:

APOLLO_API_KEY=your_master_api_key_here

The People API Search endpoint requires a master API key.

Run the app
streamlit run app.py

Then open the URL shown in the terminal (usually http://localhost:8501).

Output & storage notes

Uploaded files are stored server-side in data/uploads/
Enriched outputs are stored server-side in data/outputs/
Checkpoints are written after each row to allow resume
Files are not written to the user’s computer automatically
Use the Download button to save locally
If running locally, these folders exist in your project directory.
If running remotely, they exist on the server/container.