"""Tests for Wireless Insite Setup Parser."""

import pytest
from unittest.mock import patch, mock_open
from deepmimo.converters.wireless_insite import setup_parser

def test_tokenize_file():
    file_content = "begin_<test>\nparam 1.0\nend_<test>\n"
    with patch("builtins.open", mock_open(read_data=file_content)):
        tokens = list(setup_parser.tokenize_file("dummy"))
        assert tokens == ['begin_<test>', '\n', 'param', '1.0', '\n', 'end_<test>', '\n']

    # Test ignoring first line if special format
    file_content_ignored = "Format type:keyword version: 1.0\nbegin_<test>\nend_<test>\n"
    with patch("builtins.open", mock_open(read_data=file_content_ignored)):
        tokens = list(setup_parser.tokenize_file("dummy"))
        assert tokens == ['begin_<test>', '\n', 'end_<test>', '\n']

def test_node_class():
    node = setup_parser.Node(name="test", kind="type")
    node["key"] = "value"
    assert node["key"] == "value"
    del node["key"]
    with pytest.raises(KeyError):
        _ = node["key"]

def test_parse_line_value():
    # Helper to create iterator
    def make_iter(lst):
        return iter(lst)
    
    # Int
    tokens = setup_parser.peekable(make_iter(["123", "\n"]))
    val = setup_parser.parse_line_value(tokens)
    assert val == (123,)
    
    # Float
    tokens = setup_parser.peekable(make_iter(["1.5", "\n"]))
    val = setup_parser.parse_line_value(tokens)
    assert val == (1.5,)
    
    # Bool
    tokens = setup_parser.peekable(make_iter(["yes", "no", "\n"]))
    val = setup_parser.parse_line_value(tokens)
    assert val == (True, False)
    
    # String
    tokens = setup_parser.peekable(make_iter(["hello", "\n"]))
    val = setup_parser.parse_line_value(tokens)
    assert val == ("hello",)

def test_parse_node():
    # Node with name and params
    tokens = setup_parser.peekable(iter([
        "begin_<node>", "my_node", "\n",
        "param", "10", "\n",
        "end_<node>", "\n"
    ]))
    name, node = setup_parser.parse_node(tokens)
    assert name == "node"
    assert node.name == "my_node"
    assert node["param"] == 10

    # Node with label
    tokens = setup_parser.peekable(iter([
        "begin_<node>", "\n",
        "some_label", "\n",
        "end_<node>", "\n"
    ]))
    name, node = setup_parser.parse_node(tokens)
    assert node.labels == ["some_label"]

    # Nested node
    tokens = setup_parser.peekable(iter([
        "begin_<parent>", "\n",
        "begin_<child>", "\n",
        "param", "1", "\n",
        "end_<child>", "\n",
        "end_<parent>", "\n"
    ]))
    name, node = setup_parser.parse_node(tokens)
    # The parser puts nested nodes as tuples (name, node) into values?
    # parse_values calls parse_node which returns (name, node).
    # Then parse_node (parent) appends it to data if not matched as label/value.
    # Actually parse_values returns a list of lines. 
    # parse_node: case _: node.data.append(value)
    # BUT, parse_values logic:
    # case (str(label), value): node[label] = value.
    # A nested node returns (name, node). So it matches (str, value).
    # So it goes into values!
    assert "child" in node.values
    child_node = node["child"]
    assert child_node["param"] == 1

def test_parse_document():
    tokens = setup_parser.peekable(iter([
        "begin_<node1>", "MyNode1", "\n",
        "end_<node1>", "\n",
        "begin_<node2>", "MyNode2", "\n", "param", "val", "\n",
        "end_<node2>", "\n"
    ]))
    doc = setup_parser.parse_document(tokens)
    assert "MyNode1" in doc
    assert "MyNode2" in doc
    assert doc["MyNode2"]["param"] == "val"

