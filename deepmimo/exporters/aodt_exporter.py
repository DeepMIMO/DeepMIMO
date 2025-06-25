"""
AODT Exporter Module.

This module provides functionality for exporting AODT data to parquet format.
Note: This functionality requires additional dependencies.
Install them using: pip install 'deepmimo[aodt]'
"""

import os
from typing import List, TYPE_CHECKING, Any

Client = 'Client' if TYPE_CHECKING else Any # from clickhouse_driver import Client

EXCEPT_TABLES = ['cfrs', 'training_result', 'world', 'csi_report',
                 'telemetry', 'dus', 'ran_config']

# Import optional dependencies - this only runs once when the module is imported
try:
    import pandas as pd
    import pyarrow  # Required for parquet support
except ImportError:
    raise ImportError(
        "AODT export functionality requires additional dependencies. "
        "Please install them using: pip install 'deepmimo[aodt]'"
    )

def get_all_tables(client: Client, database: str) -> List[str]:
    """Get list of all tables in the database."""
    query = f"SELECT name FROM system.tables WHERE database = '{database}'"
    try:
        tables = client.execute(query)
    except Exception as e:
        raise Exception(f"Failed to get table list: {str(e)}")
    
    return [table[0] for table in tables]

def export_table_to_parquet(client: Client, database: str, table_name: str, 
                            output_dir: str) -> None:
    """Export a single table to a parquet file using clickhouse-connect."""

    query = f"SELECT * FROM {database}.{table_name}"
    
    try:
        columns = [col[0] for col in client.execute(f"DESCRIBE TABLE {database}.{table_name}")]
        df = pd.DataFrame(client.execute(query), columns=columns)
    except Exception as e:
        print(f"Error exporting {table_name}: {str(e)}")
        raise
        
    output_file = os.path.join(output_dir, f"{table_name}.parquet")
    
    # Save as Parquet
    df.to_parquet(output_file, index=False)

    print(f"Exported table {table_name} ({len(df)} rows) to {output_file}")
    return

def aodt_exporter(client: Client, database: str = '', output_dir: str = '.',
                  ignore_tables: List[str] = EXCEPT_TABLES) -> str:
    """Export a database to parquet files.
    
    Args:
        client: Clickhouse client instance
        database: Database name to export. If empty, uses first available database.
        output_dir: Directory to save parquet files. Defaults to current directory.
        ignore_tables: List of tables to ignore. Defaults to EXCEPT_TABLES.
        
    Returns:
        str: Path to the directory containing the exported files.
    """
    if database == '':  # default to first database
        database = client.execute('SHOW DATABASES')[1][0]
        print(f'Default to database: {database}')
    
    tables = get_all_tables(client, database)
    
    tables_to_export = [table for table in tables if table not in ignore_tables]
    
    tables_output_dir = os.path.join(output_dir, database)
    os.makedirs(tables_output_dir, exist_ok=True)
    
    for table in tables_to_export:
        export_table_to_parquet(client, database, table, tables_output_dir)
        
    return tables_output_dir

# Make the function directly available when importing the module
__all__ = ['aodt_exporter']