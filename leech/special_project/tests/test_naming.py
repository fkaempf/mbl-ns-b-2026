from pathlib import Path

from leechtemplate.naming import NameList

NAME_LIST = Path(__file__).resolve().parents[1] / "src" / "leechtemplate" / "cell_names.yaml"


def test_seed_list_loads():
    nl = NameList.from_yaml(NAME_LIST)
    assert "Retzius" in nl.names
    assert "DE-3" in nl.names
    assert "NS/151" in nl.names


def test_known_unknown():
    nl = NameList.from_yaml(NAME_LIST)
    assert nl.is_known("Retzius")
    assert not nl.is_known("Totally-Made-Up-Cell")


def test_metadata_accessible():
    nl = NameList.from_yaml(NAME_LIST)
    de3 = nl.get("DE-3")
    assert de3 is not None
    assert de3.typical_aspect == "dorsal"
