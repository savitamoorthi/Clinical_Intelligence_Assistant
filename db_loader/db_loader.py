import sqlite3
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/Users/savitamoorthi/Desktop/GenBI/data/raw")
DB_PATH = "mimic.db"


# ============================
# Helper Functions
# ============================

def load_csv(filename):
    return pd.read_csv(DATA_DIR / filename)


def convert_datetime(df, datetime_cols):
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").astype(str)
    return df


def execute_schema(conn, sql):
    conn.execute(sql)
    conn.commit()


def load_table(conn, table_name, df):
    df.to_sql(table_name, conn, if_exists="append", index=False)
    print(f"Loaded: {table_name}")


# ============================
# Table Loader Functions
# ============================

def load_patients(conn):
    df = load_csv("patients.csv")
    df = convert_datetime(df, ["dod"])

    sql = """
    CREATE TABLE IF NOT EXISTS patients (
        subject_id INTEGER NOT NULL PRIMARY KEY,
        gender TEXT NOT NULL,
        anchor_age INTEGER NOT NULL,
        anchor_year INTEGER NOT NULL,
        anchor_year_group TEXT NOT NULL,
        dod TEXT
    );
    """
    execute_schema(conn, sql)
    load_table(conn, "patients", df)


def load_admissions(conn):
    df = load_csv("admissions.csv")
    df = convert_datetime(df, [
        "admittime", "dischtime", "deathtime",
        "edregtime", "edouttime"
    ])

    sql = """
    CREATE TABLE IF NOT EXISTS admissions (
        subject_id INTEGER NOT NULL,
        hadm_id INTEGER NOT NULL PRIMARY KEY,
        admittime TEXT NOT NULL,
        dischtime TEXT,
        deathtime TEXT,
        admission_type TEXT NOT NULL,
        admit_provider_id TEXT,
        admission_location TEXT,
        discharge_location TEXT,
        insurance TEXT,
        language TEXT,
        marital_status TEXT,
        race TEXT,
        edregtime TEXT,
        edouttime TEXT,
        hospital_expire_flag INTEGER,
        FOREIGN KEY (subject_id) REFERENCES patients(subject_id)
    );
    """
    execute_schema(conn, sql)
    load_table(conn, "admissions", df)


def load_d_icd_diagnoses(conn):
    df = load_csv("d_icd_diagnoses.csv")

    sql = """
    CREATE TABLE IF NOT EXISTS d_icd_diagnoses (
        icd_code TEXT NOT NULL,
        icd_version INTEGER NOT NULL,
        long_title TEXT,
        PRIMARY KEY (icd_code, icd_version)
    );
    """
    execute_schema(conn, sql)
    load_table(conn, "d_icd_diagnoses", df)


def load_d_icd_procedures(conn):
    df = load_csv("d_icd_procedures.csv")

    sql = """
    CREATE TABLE IF NOT EXISTS d_icd_procedures (
        icd_code TEXT NOT NULL,
        icd_version INTEGER NOT NULL,
        long_title TEXT,
        PRIMARY KEY (icd_code, icd_version)
    );
    """
    execute_schema(conn, sql)
    load_table(conn, "d_icd_procedures", df)


def load_diagnoses_icd(conn):
    df = load_csv("diagnoses_icd.csv")

    sql = """
    CREATE TABLE IF NOT EXISTS diagnoses_icd (
        subject_id INTEGER NOT NULL,
        hadm_id INTEGER NOT NULL,
        seq_num INTEGER NOT NULL,
        icd_code TEXT,
        icd_version INTEGER,
        PRIMARY KEY (subject_id, hadm_id, seq_num),
        FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
        FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id),
        FOREIGN KEY (icd_code, icd_version) REFERENCES d_icd_diagnoses(icd_code, icd_version)
    );
    """
    execute_schema(conn, sql)
    load_table(conn, "diagnoses_icd", df)


def load_procedures_icd(conn):
    df = load_csv("procedures_icd.csv")
    df = convert_datetime(df, ["chartdate"])

    sql = """
    CREATE TABLE IF NOT EXISTS procedures_icd (
        subject_id INTEGER NOT NULL,
        hadm_id INTEGER NOT NULL,
        seq_num INTEGER,
        chartdate TEXT,
        icd_code TEXT,
        icd_version INTEGER,
        FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
        FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id),
        FOREIGN KEY (icd_code, icd_version) REFERENCES d_icd_procedures(icd_code, icd_version)
    );
    """
    execute_schema(conn, sql)
    load_table(conn, "procedures_icd", df)


def load_prescriptions(conn):
    df = load_csv("prescriptions.csv")
    df = convert_datetime(df, ["starttime", "stoptime"])

    sql = """
    CREATE TABLE IF NOT EXISTS prescriptions (
        subject_id INTEGER NOT NULL,
        hadm_id INTEGER NOT NULL,
        pharmacy_id INTEGER,
        poe_id TEXT,
        poe_seq INTEGER,
        order_provider_id TEXT,
        starttime TEXT,
        stoptime TEXT,
        drug_type TEXT,
        drug TEXT,
        formulary_drug_cd TEXT,
        gsn TEXT,
        ndc TEXT,
        prod_strength TEXT,
        form_rx TEXT,
        dose_val_rx TEXT,
        dose_unit_rx TEXT,
        form_val_disp TEXT,
        form_unit_disp TEXT,
        doses_per_24_hrs TEXT,
        route TEXT,
        FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
        FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id)
    );
    """
    execute_schema(conn, sql)
    load_table(conn, "prescriptions", df)


# ============================
# Main Entry Point
# ============================

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")

    load_patients(conn)
    load_admissions(conn)
    load_d_icd_diagnoses(conn)
    load_d_icd_procedures(conn)
    load_diagnoses_icd(conn)
    load_procedures_icd(conn)
    load_prescriptions(conn)

    conn.close()
    print("\nDone: All MIMIC tables loaded.")


if __name__ == "__main__":
    main()
