"""RD1 (docs/reviews/research_decisions.md): a finite cylindrical region.

Red-first suite: written against the decision record before the code. Covers
parse/format idempotency for both cylinder forms (the old axis-infinite form
must keep working), backward compatibility of the IR, and the exact builder
membership semantics of `chaord.build.regions`."""
import numpy as np
import pytest

from chaord.lang.fmt import format_program
from chaord.lang.ir import ir_equal
from chaord.lang.parser import parse_text

NL = chr(10)


def _parse(text):
    return parse_text(text)


# ------------------------------------------------------------ parse / format --

def test_finite_cylinder_parses():
    text = (f"chaord 0.1{NL}{NL}crystal tube : cylinder axis z "
            f"center 0. 0. 20. radius 10.0 A length 40.0 A {{{NL}}}{NL}")
    program = _parse(text)
    region = program.blocks[0]
    shp = region.geometry.parts[0]
    assert shp.t == "cylinder"
    assert shp.axis == "z"
    assert shp.cx.num == "0." and shp.cy.num == "0."
    assert shp.cz is not None and shp.cz.num == "20."
    assert shp.radius.num == "10.0" and shp.radius.unit == "A"
    assert shp.length is not None and shp.length.num == "40.0"
    assert shp.length.unit == "A"


def test_finite_cylinder_format_is_canonical_and_idempotent():
    text = (f"chaord 0.1{NL}{NL}crystal tube : cylinder axis z "
            f"center 0. 0. 20. radius 10.0 A length 40.0 A {{{NL}}}{NL}")
    program = _parse(text)
    assert format_program(program) == text
    again = _parse(format_program(program))
    assert ir_equal(again, program)


def test_finite_cylinder_without_unit_round_trips():
    text = (f"chaord 0.1{NL}{NL}liquid core : cylinder axis x "
            f"center 1 2 3 radius 5 length 100 {{{NL}}}{NL}")
    program = _parse(text)
    assert format_program(program) == text
    assert ir_equal(_parse(format_program(program)), program)


def test_infinite_cylinder_form_unchanged():
    """The pre-existing axis-infinite form parses and prints as before."""
    text = (f"chaord 0.1{NL}{NL}liquid core : cylinder axis z "
            f"center 1.5 2.5 radius 3.0 A {{{NL}}}{NL}")
    program = _parse(text)
    shp = program.blocks[0].geometry.parts[0]
    assert shp.cz is None and shp.length is None
    assert format_program(program) == text
    assert ir_equal(_parse(format_program(program)), program)


def test_cylinder_in_geometry_chain_combines():
    text = (f"chaord 0.1{NL}{NL}liquid core : cylinder axis z center 0. 0. 0. "
            f"radius 10 length 20 and slab z 0 .. 10 {{{NL}}}{NL}")
    program = _parse(text)
    geo = program.blocks[0].geometry
    assert [p.t for p in geo.parts] == ["cylinder", "slab"]
    assert geo.ops == ["and"]
    assert format_program(program) == text


def test_chirality_free_malformed_cylinder_rejected():
    from chaord.lang.errors import ChaordSyntaxError
    with pytest.raises(ChaordSyntaxError):
        _parse(f"chaord 0.1{NL}{NL}crystal t : cylinder axis z "
               f"center 0. 0. 20. length 40.0 A {{{NL}}}{NL}")  # radius required


# ---------------------------------------------------------------- membership --

def _mask(geometry, points, L=(100.0, 100.0, 100.0)):
    from chaord.build.regions import geometry_mask
    pos = np.asarray(points, float)
    return geometry_mask(geometry, pos, np.asarray(L, float))


def _cyl(axis="z", cx="0.", cy="0.", cz=None, r="10.", ln=None):
    from chaord.lang.ir import GeoChain, Quantity, ShCylinder
    return GeoChain(parts=[ShCylinder(
        axis=axis, cx=Quantity(num=cx), cy=Quantity(num=cy),
        cz=Quantity(num=cz) if cz is not None else None,
        radius=Quantity(num=r),
        length=Quantity(num=ln) if ln is not None else None)], ops=[])


def test_cylinder_membership_inside_outside():
    geo = _cyl(cz="50.", ln="40.")  # radius 10, axial window [30, 70]
    pts = [(0, 0, 40),      # on the axis: inside
           (10, 0, 40),     # exactly at radius: inside (inclusive)
           (10.0001, 0, 40),  # just outside the radius
           (0, 10, 30),     # at radius and at the lower end plane: inside
           (0, 9.9, 29.99),  # below the axial window: outside
           (0, 0, 50.0001),  # mid-window: inside
           (5, 5, 70),      # inside radius, exactly the top end plane: inside
           (5, 5, 70.0001),  # above the window: outside
           ]
    assert list(_mask(geo, pts)) == [True, True, False, True, False, True,
                                     True, False]


def test_cylinder_membership_infinite_form_has_no_axial_window():
    geo = _cyl()  # no length: every z, radius 10 around (0, 0)
    pts = [(0, 0, -40.0), (9.9, 0, 99.0), (10.1, 0, 0.0)]
    assert list(_mask(geo, pts)) == [True, True, False]


def test_cylinder_membership_minimum_image_radial():
    """A tube hugging the periodic boundary is measured through it."""
    geo = _cyl(cx="99.", cy="0.", cz="50.", ln="40.", r="5.")
    pts = [(1.0, 0, 40),    # distance 2 through the x boundary: inside
           (94.5, 0, 40),   # distance |94.5-99|=4.5 direct: inside
           (0.0, 0, 40)]    # distance 1: inside
    assert list(_mask(geo, pts)) == [True, True, True]
    geo2 = _cyl(cx="99.", cy="0.", cz="50.", ln="40.", r="0.5")
    pts2 = [(1.0, 0, 40),    # distance 1 through the boundary: outside
            (94.5, 0, 40),   # distance 4.5: outside
            (99.4, 0, 40)]   # distance 0.4 direct: inside
    assert list(_mask(geo2, pts2)) == [False, False, True]


def test_cylinder_membership_axial_window_wraps():
    """lo > hi in the wrapped-centre convention wraps through the boundary.

    centre 5, length 90 -> the window [5 - 45, 5 + 45] = [-40, 50] wraps to
    [50, 100]+[0, 50]; equivalently the EXCLUDED window is (50, 60)."""
    geo = _cyl(cz="5.", ln="90.")
    pts = [(0, 0, 0.0),    # in [0, 50]: inside
           (0, 0, 49.9),   # inside
           (0, 0, 55.0),   # in the excluded gap
           (0, 0, 60.0),   # back inside (wrap boundary, inclusive)
           (0, 0, 60.1)]   # inside
    assert list(_mask(geo, pts)) == [True, True, False, True, True]


def test_geometry_chain_and_or_minus():
    from chaord.lang.ir import GeoChain, Quantity, RangeVal, ShSlab
    slab = ShSlab(axis="z", rng=RangeVal(lo=Quantity(num="0."),
                                         hi=Quantity(num="40.")))
    geo = GeoChain(parts=[_cyl(cz="50.", ln="40.").parts[0], slab],
                   ops=["minus"])
    pts = [(0, 0, 40),   # in the cylinder, in the slab window: removed
           (0, 0, 65)]   # in the cylinder (window [30, 70]), outside slab: kept
    assert list(_mask(geo, pts)) == [False, True]
    both = GeoChain(parts=[_cyl(cz="50.", ln="40.").parts[0], slab], ops=["and"])
    assert list(_mask(both, pts)) == [True, False]


def test_sphere_and_box_membership():
    from chaord.lang.ir import GeoChain, Quantity, RangeVal, ShBox, ShSphere
    sph = GeoChain(parts=[ShSphere(cx=Quantity(num="0."), cy=Quantity(num="0."),
                                   cz=Quantity(num="0."),
                                   radius=Quantity(num="5."))], ops=[])
    assert list(_mask(sph, [(0, 0, 0), (4.9, 0, 0), (5.0, 0, 0), (5.1, 0, 0)])) \
        == [True, True, True, False]
    box = GeoChain(parts=[ShBox(
        xs=RangeVal(lo=Quantity(num="1."), hi=Quantity(num="2.")),
        ys=RangeVal(lo=Quantity(num="1."), hi=Quantity(num="2.")),
        zs=RangeVal(lo=Quantity(num="1."), hi=Quantity(num="2.")))], ops=[])
    assert list(_mask(box, [(1.5, 1.5, 1.5), (1.5, 1.5, 2.5), (2., 2., 2.)])) \
        == [True, False, True]


def test_all_rest_and_slab_membership_wrap():
    from chaord.lang.errors import ChaordError
    from chaord.lang.ir import (GeoChain, Quantity, RangeVal, ShAll, ShRest,
                                ShSlab)
    from chaord.build.regions import geometry_mask
    L = np.array([20.0, 20.0, 20.0])
    pos = np.array([[1.0, 1.0, 1.0], [10.0, 10.0, 10.0]])
    assert list(geometry_mask(GeoChain(parts=[ShAll()], ops=[]), pos, L)) \
        == [True, True]
    slab = GeoChain(parts=[ShSlab(axis="z", rng=RangeVal(
        lo=Quantity(num="15."), hi=Quantity(num="5.")))], ops=[])
    # wraps: z in [15, 20) + [0, 5]
    slabmask = geometry_mask(slab, pos, L)
    assert list(slabmask) == [True, False]
    # `rest` alone has no standalone membership: fail closed without the
    # complement mask, and with it, rest = the complement supplied
    with pytest.raises(ChaordError):
        geometry_mask(GeoChain(parts=[ShRest()], ops=[]), pos, L)
    got = geometry_mask(GeoChain(parts=[ShRest()], ops=[]), pos, L,
                        rest=slabmask)
    assert list(got) == [False, True]
