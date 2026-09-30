# Exposome explorer: Hai District, Tanzania

Interactive map of the exposome layers available for **Hai District, Tanzania** (17 wards of Hai District, Kilimanjaro Region, northern Tanzania). Toggle each layer and click
the map to read the value of every active layer at that point.

Part of BrainLat / GEMMA (Global Exposome Modeling, Mapping & Analytics).

- Rasters are drawn pixel by pixel (no resampling to a finer grid); values on click are read from the native raster.
- Administrative layers are one ward = one value; 1 km analysis grids are shown as cells.
- Sources, methodology and limitations are in the *Legend and sources* panel.
- Ward boundaries: Tanzania National Bureau of Statistics (NBS), 2022 Population and Housing Census. Please cite NBS when using them.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy (Streamlit Community Cloud)

Repository: this one, branch `main`, main file `streamlit_app.py`, Python 3.12 or 3.13.
