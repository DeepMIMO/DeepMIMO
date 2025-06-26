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

def get_all_databases(client: Client) -> List[str]:
    query = "SHOW DATABASES"
    return [db_name[0] for db_name in client.execute(query)]

def get_all_tables(client: Client, database: str) -> List[str]:
    """Get list of all tables in the database."""
    query = f"SELECT name FROM system.tables WHERE database = '{database}'"
    try:
        tables = client.execute(query)
    except Exception as e:
        raise Exception(f"Failed to get table list: {str(e)}")
    
    return [table[0] for table in tables]

def get_table_cols(client, database, table):
    query = f"DESCRIBE TABLE {database}.{table}"
    return [col[0] for col in client.execute(query)]

def load_table_to_df(client, database, table):
    query = f"SELECT * FROM {database}.{table}"
    
    try:
        columns = get_table_cols(client, database, table)
        df = pd.DataFrame(client.execute(query), columns=columns)
    except Exception as e:
        print(f"Error exporting {table}: {str(e)}")
        raise
    
    return df

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
    if database == '':  # select the first (non-built-in) database by default
        available_databases = get_all_databases(client)
        database = available_databases[1]
        print(f'Default to database: {database}')
    
    # Get all available tables in the database
    tables = get_all_tables(client, database)

    # Filter tables to export
    tables_to_export = [table for table in tables if table not in ignore_tables]

    # Decide if Static or Dynamic scenario (respectively, 1 or more time stamps)
    time_table = load_table_to_df(client, database, 'time_info')
    n_times = len(time_table) - 1 # AODT always outputs one more time index than necessary

    target_dirs = []
    export_dir = os.path.join(output_dir, database)
    if n_times < 1: 
        raise Exception('Empty simulation')
    elif n_times == 1: # Static
        target_dirs += [export_dir]
    elif n_times > 1: # Dynamic
        target_dirs += [os.path.join(export_dir, f'scene_{t:04d}') for t in range(n_times)]

    # Current information:
    direct_tables = ['db_info', 'materials', 'panels', 'patterns', 'runs', 'scenario']
    time_idx_tables = ['cirs', 'raypaths']
    TIME_COL = 'time_idx'
    for time_idx, target_dir in enumerate(target_dirs):
        os.makedirs(target_dir, exist_ok=True)
        for table in tables_to_export:
            
            # Decide whether to export table with or without indexing time
            if table in direct_tables:
                time_indexing_needed = False
            elif table in time_idx_tables:
                time_indexing_needed = True
            else:
                table_cols = get_table_cols(client, database, table)
                time_indexing_needed = TIME_COL in table_cols
            
            table_df = load_table_to_df(client, database, table)
            if time_indexing_needed:
                table_df = table_df[table_df[TIME_COL] == time_idx]
                
            output_file = os.path.join(target_dir, f"{table}.parquet")
            table_df.to_parquet(output_file, index=False)
            print(f"Exported table {table} ({len(table_df)} rows) to {output_file}")
    
    return export_dir

# Make the function directly available when importing the module
__all__ = ['aodt_exporter']