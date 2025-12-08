"""Tests for Wireless Insite XML Parser."""

import xml.etree.ElementTree as ET

from deepmimo.converters.wireless_insite import xml_parser


def test_xml_to_dict() -> None:
    xml = """
    <root>
        <remcom::rxapi::Child Value="10" />
        <remcom::rxapi::List>
            <Item>A</Item>
            <Item>B</Item>
        </remcom::rxapi::List>
    </root>
    """
    root = ET.fromstring(xml.replace("::", "_"))  # Simulate clean XML
    d = xml_parser.xml_to_dict(root)
    assert d["remcom_rxapi_Child"] == 10
    assert d["remcom_rxapi_List"]["Item"] == ["A", "B"]


def test_parse_insite_xml(tmp_path) -> None:
    xml_content = """<remcom::rxapi::Job>
        <remcom::rxapi::Scene>
            <remcom::rxapi::TxRxSetList>
                <remcom::rxapi::TxRxSetList>
                    <TxRxSet>
                        <remcom::rxapi::GridSet>
                            <OutputID Value="1"/>
                        </remcom::rxapi::GridSet>
                    </TxRxSet>
                </remcom::rxapi::TxRxSetList>
            </remcom::rxapi::TxRxSetList>
        </remcom::rxapi::Scene>
    </remcom::rxapi::Job>"""

    test_file = tmp_path / "dummy.xml"
    test_file.write_text(xml_content, encoding="utf-8")

    data = xml_parser.parse_insite_xml(str(test_file))
    # Check structure
    assert "remcom_rxapi_Job" in data
    # Check replace worked
    # If :: was replaced by _, then keys should have _
    assert "remcom_rxapi_Scene" in data["remcom_rxapi_Job"]
