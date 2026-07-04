import os
import re
import urllib.request
import pandas as pd

def download_and_clean_data():
    # 1. Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    excel_path = "data/GSAF5.xls"
    csv_output_path = "data/gsaf.csv"
    
    # 2. Download official GSAF Excel File if not already present
    url = "https://www.sharkattackfile.net/spreadsheets/GSAF5.xls"
    if not os.path.exists(excel_path):
        print(f"📥 Downloading GSAF Excel file from {url}...")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            with open(excel_path, "wb") as f:
                f.write(response.read())
        print("✅ Download completed.")
    else:
        print("ℹ️ Excel log already exists locally. Skipping download.")

    # 3. Read Excel file
    print("📖 Reading GSAF Excel spreadsheet...")
    # Excel files in GSAF are often large and have empty rows at the bottom.
    df = pd.read_excel(excel_path, engine="xlrd")
    print(f"Loaded raw dataset with {len(df)} rows.")

    # Clean headers (strip spaces)
    df.columns = df.columns.str.strip()
    
    # 4. Filter for Hawaii incidents
    # GSAF uses "State" in this version
    state_col = None
    for c in ["State", "Area"]:
        if c in df.columns:
            state_col = c
            break
    if state_col is None:
        raise KeyError("Could not find 'State' or 'Area' column in the dataset to filter by Hawaii.")
        
    print(f"🌴 Filtering for incidents around Hawai'i using column '{state_col}'...")
    df_hawaii = df[df[state_col].str.contains(r"Hawaii|Hawai'i", case=False, na=False)].copy()
    print(f"Found {len(df_hawaii)} raw rows for Hawaii.")

    # 5. Clean columns & Extract features
    # Standardize Fatal target
    fatal_col = None
    for c in ["Fatal Y/N", "Fatal (Y/N)", "Fatal"]:
        if c in df_hawaii.columns:
            fatal_col = c
            break
    if fatal_col is None:
        raise KeyError("Could not find 'Fatal Y/N' or 'Fatal (Y/N)' column in GSAF dataset.")
        
    df_hawaii["Fatal (Y/N)"] = df_hawaii[fatal_col].astype(str).str.strip().str.upper()
    # Keep only strict Y and N cases
    df_hawaii = df_hawaii[df_hawaii["Fatal (Y/N)"].isin(["Y", "N"])]
    print(f"Rows with clean binary target (Fatal Y/N): {len(df_hawaii)}")

    # Map Species to Cleaned Species
    # Find species column (GSAF often has 'Species' or 'Species ')
    species_col = None
    for c in ["Species", "Species "]:
        if c in df_hawaii.columns:
            species_col = c
            break
    if species_col is None:
        species_col = [c for c in df_hawaii.columns if "species" in c.lower()][0]
    
    # Map Time to Time of Day
    time_col = "Time" if "Time" in df_hawaii.columns else "Time of Day"

    # Rename columns
    rename_dict = {
        time_col: "Time of Day"
    }
    df_hawaii = df_hawaii.rename(columns=rename_dict)

    # Clean Species to remove leakage and map to clean categories
    def clean_species(text):
        if not isinstance(text, str):
            return "Unknown"
        text = text.lower()
        if "tiger" in text:
            return "Tiger Shark"
        elif "reef" in text or "blacktip" in text:
            return "Reef/Blacktip Shark"
        elif "galapagos" in text:
            return "Galapagos Shark"
        elif "cookiecutter" in text:
            return "Cookiecutter Shark"
        elif "white" in text:
            return "White Shark"
        elif "hammerhead" in text:
            return "Hammerhead Shark"
        elif "shark" in text:
            return "Unknown Shark"
        else:
            return "Unknown"

    df_hawaii["Cleaned Species"] = df_hawaii[species_col].apply(clean_species)

    # Parse Month from Date column
    months_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    def extract_month(date_val):
        if not isinstance(date_val, str):
            if pd.notna(date_val):
                date_str = str(date_val)
            else:
                return "Unknown"
        else:
            date_str = date_val
            
        for m in months_list:
            if re.search(m, date_str, re.IGNORECASE):
                return m
        return "Unknown"

    df_hawaii["Month"] = df_hawaii["Date"].apply(extract_month)

    # Fill NaNs for our features
    features = ["Activity", "Time of Day", "Month", "Cleaned Species"]
    for feat in features:
        if feat in df_hawaii.columns:
            df_hawaii[feat] = df_hawaii[feat].fillna("Unknown").astype(str).str.strip()
        else:
            df_hawaii[feat] = "Unknown"

    # Select final columns to save
    final_cols = features + ["Fatal (Y/N)"]
    df_clean = df_hawaii[final_cols].copy()

    # Save to CSV
    df_clean.to_csv(csv_output_path, index=False)
    print(f"✅ Cleaned data exported successfully to '{csv_output_path}' ({len(df_clean)} rows).")

if __name__ == "__main__":
    download_and_clean_data()
