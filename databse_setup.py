import pandas as pd
import sqlite3
import logging
import os

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup_schemes_db.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_sqlite_db(csv_path: str = "dataset.csv", 
                    db_path: str = "new_schemes.db"):
    """Set up SQLite database by loading CSV data, using Unnamed: 0 as the primary key."""
    logger.info(f"Setting up SQLite database at {db_path} from CSV {csv_path}")
    try:
        # Verify CSV exists
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Read CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV with {len(df)} rows")

        # Rename 'Unnamed: 0' to 'scheme_id' for clarity
        if 'Unnamed: 0' in df.columns:
            df = df.rename(columns={'Unnamed: 0': 'scheme_id'})
            logger.info("Renamed 'Unnamed: 0' to 'scheme_id'")
        else:
            logger.error("Column 'Unnamed: 0' not found in CSV")
            raise ValueError("Column 'Unnamed: 0' not found in CSV")

        # Check for required columns
        required_columns = [
            'scheme_id', 'scheme_name', 'nodal_ministry', 'implementing_agency', 'target_beneficiaries',
            'tags', 'state', 'category', 'level', 'brief_description', 'detailed_description',
            'eligibility_criteria', 'documents_required', 'application_process', 'benefits',
            'Official Website', 'Application Form', 'Order/Notice'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns in CSV: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")

        # Verify scheme_id is unique
        if df['scheme_id'].duplicated().any():
            logger.error("Duplicate scheme_id values found")
            raise ValueError("Duplicate scheme_id values found. Each scheme_id must be unique.")

        # Handle missing values (convert NaN to empty strings)
        df = df.fillna('')
        logger.info("Handled missing values by converting NaN to empty strings")

        # Normalize scheme_name for consistency (trim spaces)
        df['scheme_name'] = df['scheme_name'].str.strip()
        logger.info("Normalized scheme_name values by trimming spaces")

        # Connect to SQLite
        conn = sqlite3.connect(db_path)
        logger.info(f"Connected to SQLite database at {db_path}")

        # Create table with explicit schema
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS schemes (
                scheme_id INTEGER PRIMARY KEY,
                scheme_name TEXT NOT NULL,
                nodal_ministry TEXT,
                implementing_agency TEXT,
                target_beneficiaries TEXT,
                tags TEXT,
                state TEXT,
                category TEXT,
                level TEXT,
                brief_description TEXT,
                detailed_description TEXT,
                eligibility_criteria TEXT,
                documents_required TEXT,
                application_process TEXT,
                benefits TEXT,
                "Official Website" TEXT,
                "Application Form" TEXT,
                "Order/Notice" TEXT
            )
        ''')
        logger.info("Created schemes table with explicit schema")

        # Load DataFrame into SQLite
        df.to_sql('schemes', conn, if_exists='replace', index=False)
        logger.info("Loaded data into 'schemes' table")

        # Set scheme_id as the primary key and index scheme_name for faster lookups
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_scheme_id ON schemes(scheme_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_scheme_name ON schemes(scheme_name)")
        logger.info("Created indexes on scheme_id (unique) and scheme_name")

        # Verify the data
        cursor.execute("SELECT COUNT(*) FROM schemes")
        row_count = cursor.fetchone()[0]
        logger.info(f"Inserted {row_count} rows into the database")

        if row_count != len(df):
            logger.warning(f"Mismatch in row count: CSV has {len(df)} rows, but database has {row_count} rows")

        # Verify table structure
        cursor.execute("PRAGMA table_info(schemes)")
        columns = [info[1] for info in cursor.fetchall()]
        logger.info(f"Table columns: {columns}")

        return conn
    except Exception as e:
        logger.error(f"Failed to set up SQLite database: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()
            logger.info("Closed SQLite connection")

def test_db(db_path: str = "new_schemes.db"):
    """Test the database structure and sample data."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM schemes LIMIT 1")
        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            scheme = dict(zip(columns, row))
            logger.info(f"Sample scheme: {scheme['scheme_name']} (ID: {scheme['scheme_id']})")
            print(f"Sample scheme: {scheme}")
        else:
            logger.warning("No data found in schemes table")
            print("No data found in schemes table")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        print(f"Test failed: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    try:
        setup_sqlite_db()
        test_db()
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Main function failed: {str(e)}")
