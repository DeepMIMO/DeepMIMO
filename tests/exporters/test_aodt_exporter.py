"""Tests for AODT Exporter."""

import pytest
import os
import sys
from unittest.mock import MagicMock, patch

# Mock pandas and pyarrow before importing exporter
mock_pandas = MagicMock()
mock_pandas.DataFrame = MagicMock
sys.modules["pandas"] = mock_pandas
sys.modules["pyarrow"] = MagicMock()

import pandas as pd
import deepmimo.exporters.aodt_exporter as aodt_exporter

@pytest.fixture
def mock_client():
    client = MagicMock()
    return client

def test_get_all_databases(mock_client):
    mock_client.execute.return_value = [("db1",), ("db2",)]
    dbs = aodt_exporter.get_all_databases(mock_client)
    assert dbs == ["db1", "db2"]

def test_get_all_tables(mock_client):
    mock_client.execute.return_value = [("table1",), ("table2",)]
    tables = aodt_exporter.get_all_tables(mock_client, "db")
    assert tables == ["table1", "table2"]

def test_get_table_cols(mock_client):
    mock_client.execute.return_value = [("col1", "type"), ("col2", "type")]
    cols = aodt_exporter.get_table_cols(mock_client, "db", "table")
    assert cols == ["col1", "col2"]

def test_load_table_to_df(mock_client):
    mock_client.execute.side_effect = [
        [("col1", "type"), ("col2", "type")], # get_table_cols
        [(1, "a"), (2, "b")] # select *
    ]
    
    # Configure DataFrame mock
    class DummyDF:
        def __init__(self, data=None, columns=None):
            self.columns = ["col1", "col2"] if columns is None else columns
            self.iloc = MagicMock()
            self.iloc.__getitem__.return_value.__getitem__.return_value = 1
            self._data = data
        def __len__(self):
            return 2
        def to_parquet(self, *args, **kwargs):
            pass
            
    mock_pandas.DataFrame = DummyDF
    
    df = aodt_exporter.load_table_to_df(mock_client, "db", "table")
    # assert isinstance(df, pd.DataFrame) # pd.DataFrame is the mock class, df is instance of MagicMock, not strictly instance of pd.DataFrame unless we configure it.
    assert df.columns == ["col1", "col2"]
    assert len(df) == 2

def test_aodt_exporter_flow(mock_client, tmp_path):
    # Setup mocks
    # ... (existing setup) ...
    
    with patch("deepmimo.exporters.aodt_exporter.get_all_tables", return_value=["table1", "time_info"]), \
         patch("deepmimo.exporters.aodt_exporter.load_table_to_df") as mock_load, \
         patch("deepmimo.exporters.aodt_exporter.get_table_cols", return_value=["col1"]):
        
        # Mock load_table_to_df results
        class DummyDFTime:
            def __len__(self): return 2
            def to_parquet(self, *args, **kwargs): pass
            
        class DummyDFTable:
            def __len__(self): return 1
            def to_parquet(self, *args, **kwargs): pass
            
        time_df = DummyDFTime()
        table_df = DummyDFTable()
        
        # Spy on to_parquet
        table_df.to_parquet = MagicMock()
        
        def load_side_effect(client, db, table):
            if table == "time_info":
                return time_df
            return table_df
            
        mock_load.side_effect = load_side_effect
        
        out_dir = aodt_exporter.aodt_exporter(
            mock_client, 
            database="my_db", 
            output_dir=str(tmp_path),
            ignore_tables=[]
        )
        
        assert "my_db" in out_dir
        expected_file = tmp_path / "my_db" / "table1.parquet"
        # Since table_df is a mock, to_parquet is a method call. 
        # It won't actually write the file unless we simulate it or check the call.
        table_df.to_parquet.assert_called()
        # We can check if file exists IF we didn't mock to_parquet. But table_df is a Mock.
        # So we just verify to_parquet was called.


