"""Tests for DeepMIMO Materials module."""

import pytest
import numpy as np
from dataclasses import asdict
from deepmimo.materials import Material, MaterialList
from deepmimo import consts as c

def test_material_initialization():
    """Test Material class initialization."""
    mat = Material(
        id=0,
        name="Concrete",
        permittivity=5.0,
        conductivity=0.1,
        scattering_model="none"
    )
    assert mat.id == 0
    assert mat.name == "Concrete"
    assert mat.permittivity == 5.0
    assert mat.conductivity == 0.1
    assert mat.scattering_model == "none"
    assert mat.scattering_coefficient == 0.0
    assert mat.cross_polarization_coefficient == 0.0

    # Test dict conversion (using asdict as Material is a dataclass)
    mat_dict = asdict(mat)
    assert mat_dict['name'] == "Concrete"
    assert mat_dict['permittivity'] == 5.0

def test_material_from_dict():
    """Test creating Material from dictionary."""
    data = {
        'id': 1,
        'name': "Glass",
        'permittivity': 6.0,
        'conductivity': 0.0,
        'scattering_model': "diffuse",
        'scattering_coefficient': 0.2,
        'cross_polarization_coefficient': 0.1
    }
    mat = Material(**data)
    assert mat.id == 1
    assert mat.name == "Glass"
    assert mat.permittivity == 6.0
    assert mat.scattering_model == "diffuse"
    assert mat.scattering_coefficient == 0.2

def test_material_list():
    """Test MaterialList class."""
    ml = MaterialList()
    assert len(ml) == 0

    mat1 = Material(0, "M1", 2.0, 0.0)
    ml.add_materials([mat1])
    assert len(ml) == 1
    # MaterialList iterates over Material objects
    assert mat1 in list(ml)
    assert ml[0] == mat1

    # Test adding more
    mat2 = Material(1, "M2", 3.0, 0.1)
    ml.add_materials([mat2])
    assert len(ml) == 2
    assert ml[1].permittivity == 3.0

    # Test to_dict
    d = ml.to_dict()
    # Keys are "material_{id}"
    assert "material_0" in d
    assert "material_1" in d
    assert d["material_0"]['permittivity'] == 2.0

def test_material_list_load():
    """Test loading MaterialList from dictionary."""
    data = {
        "material_0": {
            "id": 0,
            "name": "M1",
            "permittivity": 2.0,
            "conductivity": 0.0
        },
        "material_1": {
            "id": 1,
            "name": "M2",
            "permittivity": 3.0,
            "conductivity": 0.1
        }
    }
    ml = MaterialList.from_dict(data)
    assert len(ml) == 2
    # IDs are reassigned in from_dict/add_materials? 
    # add_materials reassigns IDs based on index.
    assert ml[0].permittivity == 2.0
    assert ml[1].conductivity == 0.1
