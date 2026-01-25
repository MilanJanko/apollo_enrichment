# Apollo Company Enrichment (Phase 1)

## What it does
Input: CSV with `company_website`
Output: CSV with one enriched contact per company (full name, title, email, linkedin)

Flow:
1) People API Search (free): POST /api/v1/mixed_people/api_search
2) Pick best decision maker (deterministic scoring)
3) People Enrichment (paid): POST /api/v1/people/match

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


## Run
python -m src.pipeline --input data/input_companies.csv --output data/enriched_contacts.csv
