# Visit With Us — Wellness Tourism Purchase Prediction

## Run locally
```bash
pip install -r requirements-ci.txt
python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
python tourism_project/model_building/train.py
```

## Deploy the Streamlit app
Push this repository to GitHub, wait for the Actions workflow to complete, then create a Streamlit Community Cloud app using branch `main` and main file path `tourism_project/deployment/app.py`. Select Python 3.11.

The model uses customer profile attributes available before contact. Sales-interaction fields and identifier columns are excluded to avoid target leakage.
