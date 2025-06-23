from clickhouse_driver import Client
import subprocess
import os
from pathlib import Path

def get_all_tables(client, database, table_filter=None):
    """Get list of all tables in the database."""
    try:
        query = f"""
            SELECT name 
            FROM system.tables 
            WHERE database = '{database}'
        """
        if table_filter:
            query += f" AND name LIKE '{table_filter}'"
        tables = client.execute(query)
        return [table[0] for table in tables]
    except Exception as e:
        raise Exception(f"Failed to get table list: {str(e)}")

def export_table_to_parquet(client, host, database, table_name, output_dir):
    """Export a single table to parquet file."""
    try:
        # make the output directory if does not already exist
        table_output_dir = os.path.join(output_dir, database)
        os.makedirs(table_output_dir, exist_ok=True)
        
        row_count = client.execute(f"SELECT count() FROM {database}.{table_name}")[0][0]
        output_file = os.path.join(table_output_dir, f"{table_name}.parquet")
        
        # export directly to parquet using clickhouse client
        cmd = [
            'clickhouse-client',
            '--host', host,
            '--query', f"SELECT * FROM {database}.{table_name} FORMAT Parquet",
        ]
        
        with open(output_file, 'wb') as f:
            subprocess.run(cmd, stdout=f, check=True)
            
        return output_file, row_count
    
    except Exception as e:
        print(f"Error exporting {table_name}: {str(e)}")
        raise

def export_database_to_parquet(client, host, database, output_dir):
    """Export a database to parquet files."""
    tables = get_all_tables(client, database)
    for table in tables:
        export_table_to_parquet(client, host, database, table, output_dir)
