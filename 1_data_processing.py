from pathlib import Path
import re
import pandas as pd
import numpy as np
from io import StringIO


def to_float(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().replace(",", "")
    if s in {"", "np", "*", "**", "***", "-"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_nhs_export(csv_path):
    """Parse ABS matrix-style export (gender/variable columns)"""
    lines = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    
    # Find header rows
    gender_idx = next(i for i, line in enumerate(lines) if '"Gender"' in line)
    var_idx = gender_idx + 1
    age_header_idx = next(i for i, line in enumerate(lines) if '"Age of person in single years",' in line)
    
    # Parse gender and variable rows
    gender_cols = pd.read_csv(StringIO(lines[gender_idx]), header=None).iloc[0].tolist()
    if lines[var_idx].strip().startswith('"Age of person in single years",'):
        var_cols = gender_cols
    else:
        var_cols = pd.read_csv(StringIO(lines[var_idx]), header=None).iloc[0].tolist()
    
    # Build column mapping: (column_index, gender, level)
    col_map = []
    current_gender = None
    
    for idx, (g, v) in enumerate(zip(gender_cols, var_cols)):
        g = "" if pd.isna(g) else str(g).strip()
        v = "" if pd.isna(v) else str(v).strip()
        
        if g and g != "Gender":
            current_gender = g
        
        if idx == 0:
            col_map.append((idx, "age", "age"))
            continue
        
        if not v or "RSE" in v or "Annotations" in v:
            continue
        
        level = re.sub(r"\s+", " ", v).strip()
        col_map.append((idx, current_gender or "Unknown", level))
    
    # Parse data rows
    rows = []
    for raw in lines[age_header_idx + 1:]:
        if not raw.strip():
            continue
        
        vals = pd.read_csv(StringIO(raw), header=None).iloc[0].tolist()
        age_raw = str(vals[0]).strip().strip('"')
        
        if age_raw.lower() == "total":
            continue
        
        try:
            age = int(float(age_raw))
        except ValueError:
            continue
        
        for idx, gender, level in col_map[1:]:
            if idx >= len(vals):
                continue
            value = to_float(vals[idx])
            if pd.isna(value):
                continue
            rows.append({
                "age": age,
                "gender": gender,
                "level": level,
                "value": value,
            })
    
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No rows parsed from matrix export: {csv_path.name}")
    return out


def parse_export(csv_path):
    """Parse ABS long-style export (SEIFA, states, comorbidity)"""
    lines = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header_idx = next(i for i, line in enumerate(lines) if '"Counting","Age of person in single years"' in line)
    df = pd.read_csv(csv_path, skiprows=header_idx)
    
    rename_map = {
        "Age of person in single years": "age",
        "Gender": "gender",
        "Count": "value",
    }
    
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    
    if "age" not in df.columns or "value" not in df.columns:
        raise ValueError(f"Unexpected long export columns in {csv_path.name}: {list(df.columns)}")
    
    if "SEIFA - Index of Relative Socio-economic Disadvantage - 2021 - SA2 - Deciles - National" in df.columns:
        df = df.rename(columns={"SEIFA - Index of Relative Socio-economic Disadvantage - 2021 - SA2 - Deciles - National": "level"})
    elif "State or territory - ASGS 2021" in df.columns:
        df = df.rename(columns={"State or territory - ASGS 2021": "level"})
    
    if "level" not in df.columns:
        df["level"] = "All"
    
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["value"] = df["value"].apply(to_float)
    
    out = df[["age", "gender", "level", "value"]].copy()
    out = out.dropna(subset=["age", "value"]).copy()
    out["age"] = out["age"].astype(int)
    
    if out.empty:
        raise ValueError(f"No rows parsed from long export: {csv_path.name}")
    return out


def parse_csv(csv_path):
    """Auto-detect CSV format and parse accordingly"""
    lines = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    
    # Check if it's a long export format
    if any('"Counting","Age of person in single years"' in line for line in lines[:30]):
        return parse_export(csv_path)
    else:
        return parse_nhs_export(csv_path)

def main():

    root = Path(__file__).parent
    raw_data_dir = root / "raw_data"
    output_data_dir = root / "outputs_transformed_data"
    output_data_dir.mkdir(exist_ok=True)

    csv_files = sorted(raw_data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")
    
    for csv_path in csv_files:
        try:
            df = parse_csv(csv_path)
            df.to_csv(output_data_dir / f"{csv_path.stem}_parsed.csv", index=False)
            print(f"Parsed {csv_path.name} with {len(df)} rows")
        except Exception as e:
            print(f"Error parsing {csv_path.name}: {e}")

if __name__ == "__main__":
    main()