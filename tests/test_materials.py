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

def test_material_realistic_concrete():
    """Test Material with realistic concrete properties."""
    concrete = Material(
        id=0,
        name="Concrete",
        permittivity=5.31,  # Typical concrete at 2.4 GHz
        conductivity=0.0462,  # S/m
        scattering_model="lambertian",
        scattering_coefficient=0.1,
        roughness=0.001,  # 1mm
        thickness=0.2  # 20cm wall
    )
    assert concrete.permittivity == 5.31
    assert concrete.scattering_model == Material.SCATTERING_LAMBERTIAN
    assert concrete.thickness == 0.2

def test_material_realistic_glass():
    """Test Material with realistic glass properties."""
    glass = Material(
        id=1,
        name="Glass",
        permittivity=6.27,
        conductivity=0.0001,
        scattering_model="none",
        thickness=0.006  # 6mm thick glass
    )
    assert glass.permittivity == 6.27
    assert glass.thickness == 0.006

def test_material_foliage():
    """Test Material with foliage/vegetation properties."""
    foliage = Material(
        id=2,
        name="Tree Foliage",
        permittivity=1.5,
        conductivity=0.0,
        vertical_attenuation=0.5,  # dB/m
        horizontal_attenuation=1.0,  # dB/m
        thickness=5.0  # 5m tree crown
    )
    assert foliage.vertical_attenuation == 0.5
    assert foliage.horizontal_attenuation == 1.0

def test_material_list_indexing():
    """Test MaterialList advanced indexing."""
    ml = MaterialList()
    mats = [
        Material(i, f"M{i}", float(i+1), 0.0)
        for i in range(5)
    ]
    ml.add_materials(mats)
    
    # Single index
    assert ml[2].permittivity == 3.0
    
    # Multiple indices
    sub_ml = ml[[0, 2, 4]]
    assert isinstance(sub_ml, MaterialList)
    assert len(sub_ml) == 3
    assert sub_ml[0].permittivity == 1.0
    assert sub_ml[2].permittivity == 5.0

def test_material_list_iteration():
    """Test MaterialList iteration."""
    ml = MaterialList()
    mats = [Material(i, f"M{i}", float(i), 0.0) for i in range(3)]
    ml.add_materials(mats)
    
    # Iterate and collect names
    names = [m.name for m in ml]
    assert names == ["M0", "M1", "M2"]

def test_material_with_itu_params():
    """Test Material with ITU-R P.2040 parameters."""
    mat = Material(
        id=0,
        name="Frequency Dependent Material",
        permittivity=5.0,  # Base value
        conductivity=0.1,
        itu_a=5.0,
        itu_b=0.01,
        itu_c=1.0,
        itu_d=0.05
    )
    assert mat.itu_a == 5.0
    assert mat.itu_b == 0.01
    assert mat.itu_c == 1.0
    assert mat.itu_d == 0.05

def test_material_directive_scattering():
    """Test Material with directive scattering model."""
    mat = Material(
        id=0,
        name="Directive Scatter",
        permittivity=4.0,
        conductivity=0.01,
        scattering_model="directive",
        scattering_coefficient=0.3,
        alpha_r=3.5,
        alpha_i=2.5,
        lambda_param=0.7
    )
    assert mat.scattering_model == Material.SCATTERING_DIRECTIVE
    assert mat.alpha_r == 3.5
    assert mat.alpha_i == 2.5
    assert mat.lambda_param == 0.7

def test_material_list_empty_operations():
    """Test MaterialList operations on empty list."""
    ml = MaterialList()
    assert len(ml) == 0
    
    # to_dict on empty list
    d = ml.to_dict()
    assert d == {}
    
    # Iteration
    for m in ml:
        assert False, "Should not iterate over empty list"

def test_material_list_repr():
    """Test MaterialList __repr__ method."""
    ml = MaterialList()
    ml.add_materials([Material(0, "Concrete", 5.0, 0.1)])
    
    repr_str = repr(ml)
    assert "MaterialList" in repr_str
    assert "1" in repr_str  # Should show count
    
def test_material_list_str():
    """Test MaterialList __str__ method."""
    ml = MaterialList()
    ml.add_materials([
        Material(0, "Concrete", 5.0, 0.1),
        Material(1, "Glass", 6.0, 0.01)
    ])
    
    str_str = str(ml)
    assert "Concrete" in str_str or "Material" in str_str  # Should show materials