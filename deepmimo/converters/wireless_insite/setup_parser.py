"""Wireless Insite Setup File Parser.

This module provides functionality to parse Wireless Insite setup files into Python objects.

This module provides:
- File tokenization and parsing utilities
- Node representation for setup file elements
- Document-level parsing functionality
- Type conversion and validation

The module serves as the interface between Wireless Insite's file formats and DeepMIMO's
internal data structures.

The processed file looks like a list of nodes, and nodes are dictionaries with
certain fields. Print the document to see all the elements.

The pseudo-grammar for a TXRX file looks like this:

document := node* EOF
node := BEGIN_TAG TAG_NAME? values END_TAG NL
values := (node | line_value)*
line_value := (STR | "yes" | "no" | INT | FLOAT)+ NL
"""

import contextlib
import re
from dataclasses import dataclass, field
from typing import Any

RE_BOOL_TRUE = re.compile(r"yes")
RE_BOOL_FALSE = re.compile(r"no")
RE_BEGIN_NODE = re.compile(r"begin_<(?P<node_name>\S*)>")
RE_END_NODE = re.compile(r"end_<(?P<node_name>\S*)>")
RE_INT = re.compile(r"^-?\d+$")
RE_FLOAT = re.compile(r"^-?\d+[.]\d+$")
RE_LABEL = re.compile(r"\S+")

NL_TOKEN = "\n"


def tokenize_file(path: str) -> str:
    """Break a Wireless Insite file into whitespace-separated tokens.

    Args:
        path (str): Path to the file to tokenize

    Returns:
        str: Generator yielding tokens from the file

    Notes:
        Special handling is applied to the first line if it contains format information.

    """
    with open(path) as f:
        first_line = f.readline()
        if first_line.startswith("Format type:keyword version:"):
            # print(f'Ignoring first line: {first_line.lower()}')
            pass
        else:
            yield from first_line.split()
            yield NL_TOKEN
        for line in f:
            yield from line.split()
            yield NL_TOKEN


class peekable:
    """Makes it possible to peek at the next value of an iterator."""

    def __init__(self, iterator) -> None:
        self._iterator = iterator
        # Unique sentinel used as flag.
        self._sentinel = object()
        self._next = self._sentinel

    def peek(self):
        """Peeks at the next value of the iterator, if any."""
        if self._next is self._sentinel:
            self._next = next(self._iterator)
        return self._next

    def has_values(self):
        """Check if the iterator has any values left."""
        if self._next is self._sentinel:
            with contextlib.suppress(StopIteration):
                self._next = next(self._iterator)
        return self._next is not self._sentinel

    def __iter__(self):
        """Implement the iterator protocol for `peekable`."""
        return self

    def __next__(self):
        """Implement the iterator protocol for `peekable`."""
        if (next_value := self._next) is not self._sentinel:
            self._next = self._sentinel
            return next_value
        return next(self._iterator)


@dataclass
class Node:
    """Node representation for Wireless Insite setup file sections.

    This class represents a section in a Wireless Insite setup file delimited by
    begin_<...> and end_<...> tags. It provides structured access to section data
    through dictionary-like interface.

    Attributes:
        name (str): Optional name in front of the begin_<...> tag. Defaults to ''.
        kind (str): Type of node from the tag name. Defaults to ''.
        values (dict): Dictionary mapping labels to values. Defaults to empty dict.
        labels (list): List of unlabeled identifiers. Defaults to empty list.
        data (list): List of tuples with unlabeled data. Defaults to empty list.

    Example:
        >>> node = Node()
        >>> node.name = "antenna1"
        >>> node["frequency"] = 28e9
        >>> node.values["frequency"]
        28000000000.0

    """

    name: str = ""
    kind: str = ""
    values: dict = field(default_factory=dict)
    labels: list = field(default_factory=list)
    data: list = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        """Access node values using dictionary notation.

        Args:
            key (str): Key to look up in values dictionary

        Returns:
            Any: Value associated with key

        Raises:
            KeyError: If key not found in values dictionary

        """
        return self.values.__getitem__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set node values using dictionary notation.

        Args:
            key (str): Key to set in values dictionary
            value (Any): Value to associate with key

        """
        return self.values.__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        """Delete node values using dictionary notation.

        Args:
            key (str): Key to delete from values dictionary

        Raises:
            KeyError: If key not found in values dictionary

        """
        return self.values.__delitem__(key)


def eat(tokens, expected) -> None:
    """Ensures the next token is what's expected."""
    if (tok := next(tokens)) != expected:
        msg = f"Expected token {expected!r}, got {tok!r}."
        raise RuntimeError(msg)


def parse_document(tokens) -> dict[str, Node]:
    """Parse a Wireless Insite setup document into a dictionary of nodes.

    Args:
        tokens: Iterator of tokens from tokenize_file()

    Returns:
        Dict[str, Node]: Dictionary mapping node names to Node objects

    Raises:
        RuntimeError: If document structure is invalid or contains duplicate nodes

    """
    if not isinstance(tokens, peekable):
        tokens = peekable(tokens)

    document = {}
    while tokens.has_values():
        tok = tokens.peek()
        if not RE_BEGIN_NODE.match(tok):
            msg = f"Non node {tok!r} at the top-level of the document."
            raise RuntimeError(msg)

        node_name, node = parse_node(tokens)
        node.kind = node_name
        potential_name = "_".join(tok.split(" ")[1:])[:-1]
        node_name = potential_name if potential_name else node.name
        if node_name in document:
            msg = f"Node with duplicate name {node_name} found."
            raise RuntimeError(msg)
        document[node_name] = node
    return document


def parse_node(tokens) -> tuple[str, Node]:
    """Parse a node section from a Wireless Insite setup file.

    Args:
        tokens: Iterator of tokens from tokenize_file()

    Returns:
        Tuple[str, Node]: Node name and parsed Node object

    Notes:
        A node section starts with begin_<name> and ends with end_<name>.
        The node may have an optional identifier after the begin tag.

    """
    node = Node()
    begin_tag = next(tokens)
    begin_match = RE_BEGIN_NODE.match(begin_tag)
    node_name = begin_match.group("node_name")

    # Is there a name?
    while tokens.peek() != NL_TOKEN:
        node.name += next(tokens) + " "

    # Remove possible ' ' at end of name
    if node.name and node.name[-1] == " ":
        node.name = node.name[:-1]

    eat(tokens, NL_TOKEN)

    # Parse the values and put them in the node dictionary.
    for value in parse_values(tokens):
        # What does the value look like?
        match value:
            case (str(label),):  # Is it a single label?
                node.labels.append(label)
            case (str(label), value):  # Is it a label / value pair?
                node[label] = value
            case str(label), *rest:  # Is it a label followed by 2+ values?
                node[label] = rest
            case _:  # Is it data without a label?
                node.data.append(value)

    # Parse the closing tag and newline.
    eat(tokens, f"end_<{node_name}>")
    eat(tokens, NL_TOKEN)

    return node_name, node


def parse_values(tokens):
    """Parse the lines of values within a node.

    Returns a list of line values.
    """
    lines = []

    while tokens.has_values():
        tok = tokens.peek()

        if RE_END_NODE.match(tok):
            return lines
        if RE_BEGIN_NODE.match(tok):
            lines.append(parse_node(tokens))
        else:
            lines.append(parse_line_value(tokens))

    return lines


def parse_line_value(tokens) -> tuple:
    """Parse a single line value from a Wireless Insite setup file.

    Args:
        tokens: Iterator of tokens from tokenize_file()

    Returns:
        Tuple: Tuple of parsed values with appropriate types (bool, int, float, str)

    Notes:
        Values are converted to appropriate types based on their format:
        - "yes"/"no" -> bool
        - Integer strings -> int
        - Float strings -> float
        - Other strings -> str

    """
    values = []

    while tokens.has_values() and tokens.peek() != NL_TOKEN:
        tok = next(tokens)
        if RE_BOOL_TRUE.match(tok):
            values.append(True)
        elif RE_BOOL_FALSE.match(tok):
            values.append(False)
        elif RE_FLOAT.match(tok):
            values.append(float(tok))
        elif RE_INT.match(tok):
            values.append(int(tok))
        else:
            # If it doesn't match any pattern exactly, treat as string
            values.append(tok)
    eat(tokens, NL_TOKEN)
    return tuple(values)


def parse_file(file_path: str) -> dict[str, Node]:
    """Parse a Wireless Insite setup file into a dictionary of nodes.

    Args:
        file_path (str): Path to the setup file to parse

    Returns:
        Dict[str, Node]: Dictionary mapping node names to Node objects

    Raises:
        FileNotFoundError: If file_path does not exist
        RuntimeError: If file structure is invalid

    """
    return parse_document(tokenize_file(file_path))


if __name__ == "__main__":
    tokens = tokenize_file("sample.txrx")
    document = parse_document(tokens)
