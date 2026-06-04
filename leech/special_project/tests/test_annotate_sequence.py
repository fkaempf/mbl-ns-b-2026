from leecharena.annotate import ROLES, ROLES_2, ROLES_3, advance_role


def test_two_node_mode_cycles_anterior_posterior_only():
    assert advance_role("anterior", 0, roles=ROLES_2) == ("posterior", 0)
    assert advance_role("posterior", 0, roles=ROLES_2) == ("anterior", 1)
    assert "middle" not in ROLES_2


def test_three_node_mode_is_default_roles():
    assert ROLES == ROLES_3 == ["anterior", "posterior", "middle"]


def test_advance_cycles_through_roles():
    assert advance_role("anterior", 0) == ("posterior", 0)
    assert advance_role("posterior", 0) == ("middle", 0)


def test_advance_past_last_role_bumps_leech_and_wraps():
    assert advance_role("middle", 0) == ("anterior", 1)
    assert advance_role("middle", 7) == ("anterior", 8)


def test_advance_uses_configured_role_order():
    assert advance_role(ROLES[0], 2)[0] == ROLES[1]
    assert advance_role(ROLES[-1], 2) == (ROLES[0], 3)


def test_advance_unknown_role_falls_back_to_first():
    assert advance_role("bogus", 4) == (ROLES[0], 4)
