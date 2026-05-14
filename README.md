# AGS Geotechnical Viewer

Small first iteration of a Python app for uploading AGS data and plotting Standard Penetration Test results.

## What it does now

- Upload `.ags`, `.csv`, `.txt`, or AGS-style `.xlsx` files.
- Parse AGS groups into tables.
- Read `ISPT` data using:
  - `LOCA_ID` as the investigation ID.
  - `ISPT_TOP` as SPT depth.
  - `ISPT_MAIN` as the SPT blow count.
- Match each SPT to `GEOL` using:
  - Same `LOCA_ID`.
  - `GEOL_TOP <= ISPT_TOP < GEOL_BASE`.
  - Final-base inclusive fallback where a test is exactly on the deepest logged base.
- Plot:
  - SPTs against depth for all selected investigations.
  - SPTs against depth filtered by geological unit (`GEOL_GEOL`).

## Run

```powershell
streamlit run app.py
```

Then upload an AGS file or the provided AGS-style Excel export.
