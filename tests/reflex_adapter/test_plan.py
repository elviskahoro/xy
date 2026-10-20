"""Chart plans: compile-time validation, content addressing, binding."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

import xy
from reflex_xy.plan import (
    PLAN_VERSION,
    PlanBindError,
    PlanError,
    PlanMissError,
    _code_fingerprint,
    build_plan,
    plan_of,
    require_plan,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = str(_REPO_ROOT / "python")


def _run_child(code: str, *, hash_seed):
    """Run ``code`` in a fresh interpreter with a controlled PYTHONHASHSEED.

    ``hash_seed=None`` leaves the seed unset (a random per-process value),
    which is the strongest cross-process determinism check: a canonicalized
    fingerprint must be identical even when the OS hands each child a
    different seed.
    """
    env = os.environ.copy()
    if hash_seed is None:
        env.pop("PYTHONHASHSEED", None)
    else:
        env["PYTHONHASHSEED"] = str(hash_seed)
    env["PYTHONPATH"] = _PYTHON_DIR + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _fingerprint_in_child(src: str, *, hash_seed):
    code = (
        "import sys\n"
        f"sys.path.insert(0, {_PYTHON_DIR!r})\n"
        "ns = {}\n"
        f"exec({src!r}, ns)\n"
        "from reflex_xy.plan import _code_fingerprint\n"
        "sys.stdout.write(_code_fingerprint(ns['fn']))\n"
    )
    return _run_child(code, hash_seed=hash_seed)


def scatter_plan(**mark_options):
    return build_plan(
        "scatter_chart",
        (xy.scatter("x", "y", color="mag", **mark_options), xy.x_axis(label="sigma")),
        {"title": "cloud"},
    )


def test_plan_records_probed_columns_in_resolution_order():
    plan = scatter_plan()
    assert plan.columns == ("x", "y", "mag")


def test_digest_is_stable_across_identical_builds():
    """Content addressing: separately constructed identical trees agree, so
    every worker evaluating the same page derives the same digest (X4)."""
    assert scatter_plan().digest == scatter_plan().digest
    assert scatter_plan().digest != scatter_plan(opacity=0.5).digest


def test_digest_golden_pins_the_plan_format():
    """plan_version is part of the serialization: accidental format churn
    shows up here as a digest change. An *intentional* format change bumps
    PLAN_VERSION and re-records this golden (old digests are content
    addresses — stale subscribers resync, nothing migrates)."""
    assert PLAN_VERSION == 1
    plan = build_plan("scatter_chart", (xy.scatter("x", "y"),), {})
    assert plan.digest == "b7d0b4245b686130e37d"


def test_probe_fires_the_full_validation_gate():
    with pytest.raises(ValueError, match="colormap"):
        scatter_plan(colormap="virids")
    with pytest.raises(ValueError, match="symbol"):
        scatter_plan(symbol="marsian")
    # unresolved axis-id references are figure-compile errors too (X2)
    with pytest.raises(ValueError, match="axis"):
        build_plan("scatter_chart", (xy.scatter("x", "y", y_axis="y2"),), {})


def test_aggregating_kinds_build_zero_row_plans():
    """Every aggregating kind compiles a plan zero-row (the old exclusion is
    gone, and so are the synthetic columns that replaced it): under the
    core's structural_probe() mode the marks validate config and skip
    aggregation, with grouped/coordinate/weight channels recorded like any
    other."""
    cases = [
        ("box_chart", (xy.box("v", group="g"),), {"v", "g"}),
        ("violin_chart", (xy.violin("v"),), {"v"}),
        ("hexbin_chart", (xy.hexbin("a", "b", C="w"),), {"a", "b", "w"}),
        ("contour_chart", (xy.contour("grid", x="xs", y="ys"),), {"grid", "xs", "ys"}),
        ("heatmap_chart", (xy.heatmap("grid"),), {"grid"}),
        ("stairs_chart", (xy.stairs("counts", "edges"),), {"counts", "edges"}),
        ("ecdf_chart", (xy.ecdf("v"),), {"v"}),
    ]
    for kind, children, expected in cases:
        plan = build_plan(kind, children, {})
        assert set(plan.columns) == expected, kind


def test_structural_probe_still_fails_bad_aggregating_config():
    """No synthetic data does not mean no validation: configuration errors
    of the aggregating kinds fail the zero-row probe exactly like scatter's
    bad colormap does."""
    cases = [
        ("box orientation", (xy.box("v", orientation="diagonal"),)),
        ("violin bins", (xy.violin("v", bins=2),)),
        ("hexbin gridsize", (xy.hexbin("a", "b", gridsize=0),)),
        ("hexbin range", (xy.hexbin("a", "b", range=(0.0, 0.5)),)),
        ("hexbin mincnt", (xy.hexbin("a", "b", mincnt=-1),)),
        ("contour levels", (xy.contour("grid", levels=0),)),
        ("contour extend", (xy.contour("grid", extend="sideways"),)),
        ("heatmap colormap", (xy.heatmap("grid", colormap="virids"),)),
        ("stairs where", (xy.stairs("counts", where="diagonal"),)),
        ("ecdf bins", (xy.ecdf("v", bins=-1),)),
    ]
    for label, children in cases:
        with pytest.raises(ValueError, match=label.split(" ")[-1]):
            build_plan("chart", children, {})


def test_funnel_plan_records_all_channels_and_binds_real_rows():
    """An empty probe emits no funnel trace, but still discovers the key;
    row/type validation and transition-key encoding run when data binds."""
    plan = build_plan(
        "funnel_chart",
        (
            xy.funnel(
                "stage",
                "value",
                key="id",
                animation=xy.animation(match="key"),
            ),
        ),
        {},
    )
    assert plan.columns == ("stage", "value", "id")
    fig = plan.bind(
        {
            "stage": ["Visit", "Signup", "Pay"],
            "value": [100.0, 62.0, 21.0],
            "id": ["visit", "signup", "pay"],
        }
    ).figure()
    assert [trace.kind for trace in fig.traces] == ["funnel"]
    assert fig.traces[0].transition_keys is not None
    assert fig.traces[0].transition_keys.shape == (3, 2)


def test_funnel_plan_probe_still_fails_bad_config():
    with pytest.raises(ValueError, match="orientation"):
        build_plan(
            "funnel_chart",
            (xy.funnel("stage", "value", orientation="diagonal"),),
            {},
        )


def test_shared_columns_between_aggregating_and_zero_row_marks_probe():
    """The review's repro: stairs (values len k, edges len k+1) composed
    with a scatter that reads the same 'edges' column. Synthetic per-name
    shapes made the scatter probe see lengths 9 and 0; the all-empty
    structural probe has no lengths to disagree about, and the real mixed-
    length data binds."""
    plan = build_plan(
        "chart",
        (xy.stairs("counts", "edges"), xy.scatter("edges", "other")),
        {},
    )
    assert set(plan.columns) == {"counts", "edges", "other"}
    fig = plan.bind(
        {
            "counts": np.arange(8.0),
            "edges": np.arange(9.0),
            "other": np.arange(9.0) * 2.0,
        }
    ).figure()
    assert len(fig.traces) == 2


def test_hexbin_value_dependent_configs_probe_without_aggregating():
    """The review's other repro class: a range that excludes any invented
    points, a mincnt that filters them, or a maximal gridsize must not fail
    (or allocate) at page evaluation — those are data outcomes, computed
    only when real data binds."""
    for kwargs in (
        {"range": ((0.0, 0.5), (0.0, 0.5))},
        {"mincnt": 5},
        {"gridsize": 2048},
    ):
        plan = build_plan("hexbin_chart", (xy.hexbin("a", "b", **kwargs),), {})
        assert set(plan.columns) == {"a", "b"}
    # and the range case renders with real in-range data
    plan = build_plan("hexbin_chart", (xy.hexbin("a", "b", range=((0.0, 0.5), (0.0, 0.5))),), {})
    fig = plan.bind({"a": [0.1, 0.2, 0.3], "b": [0.1, 0.2, 0.3]}).figure()
    assert fig.traces[0].kind == "hexbin"


def test_shaped_and_zero_row_marks_compose_and_share_columns():
    plan = build_plan("chart", (xy.histogram("v"), xy.ecdf("v")), {})
    assert plan.columns == ("v",)


def test_named_callables_digest_and_lambdas_are_refused():
    """hexbin's reduce_C_function default (np.mean) content-addresses as its
    import path plus a code fingerprint, so identical trees agree across
    workers and different reducers disagree. A lambda has no stable name
    and cannot keep digests faithful; it is refused toward a module-level
    function or the hatch."""
    hexbin_plan = build_plan("hexbin_chart", (xy.hexbin("a", "b", C="w"),), {})
    again = build_plan("hexbin_chart", (xy.hexbin("a", "b", C="w"),), {})
    assert hexbin_plan.digest == again.digest
    reduced = build_plan(
        "hexbin_chart", (xy.hexbin("a", "b", C="w", reduce_C_function=np.median),), {}
    )
    assert reduced.digest != hexbin_plan.digest
    with pytest.raises(PlanError, match="stable qualified name"):
        build_plan(
            "hexbin_chart",
            (xy.hexbin("a", "b", C="w", reduce_C_function=lambda values: values.max()),),
            {},
        )


def test_bound_methods_are_refused_as_plan_callables():
    """The review's repro: two bound reducers with different instance state
    used to serialize identically (module.qualname) — last-write-wins then
    made the first chart execute the second reducer. Instance state has no
    content address, so bound methods are refused outright."""

    class Quantile:
        def __init__(self, q: float) -> None:
            self.q = q

        def reduce(self, values: np.ndarray) -> float:
            return float(np.quantile(values, self.q))

    with pytest.raises(PlanError, match="bound method"):
        build_plan(
            "hexbin_chart",
            (xy.hexbin("a", "b", C="w", reduce_C_function=Quantile(0.5).reduce),),
            {},
        )


def test_callable_digest_follows_the_body_not_only_the_name():
    """A qualified name is identity, not content: editing a module-level
    reducer's body must change the digest (a rolling deployment otherwise
    executes two behaviors behind one address)."""
    import sys
    import types

    def module_reducer(body: str):
        module = types.ModuleType("plan_test_reducers")
        sys.modules["plan_test_reducers"] = module
        exec(  # noqa: S102 - building a same-qualname function pair for the pin
            f"import numpy as np\ndef reduce(values):\n    return {body}\n",
            module.__dict__,
        )
        return module.__dict__["reduce"]

    first = module_reducer("float(np.mean(values))")
    digest_one = build_plan(
        "hexbin_chart", (xy.hexbin("a", "b", C="w", reduce_C_function=first),), {}
    ).digest
    second = module_reducer("float(np.max(values))")  # same module.qualname
    digest_two = build_plan(
        "hexbin_chart", (xy.hexbin("a", "b", C="w", reduce_C_function=second),), {}
    ).digest
    assert digest_one != digest_two
    sys.modules.pop("plan_test_reducers", None)


def test_closures_are_refused_as_plan_callables():
    def make_reducer(q: float):
        def reduce(values: np.ndarray) -> float:
            return float(np.quantile(values, q))

        return reduce

    with pytest.raises(PlanError, match="stable qualified name"):
        build_plan(
            "hexbin_chart",
            (xy.hexbin("a", "b", C="w", reduce_C_function=make_reducer(0.5)),),
            {},
        )


def test_plans_refuse_concrete_arrays():
    with pytest.raises(PlanError, match="data-free"):
        build_plan("scatter_chart", (xy.scatter(np.array([1.0]), np.array([2.0])),), {})


def test_plans_refuse_per_mark_data():
    with pytest.raises(PlanError, match="per-mark data="):
        build_plan("scatter_chart", (xy.scatter("x", "y", data={"x": [], "y": []}),), {})


def test_plans_refuse_render_components():
    with pytest.raises(PlanError, match="render"):
        build_plan(
            "scatter_chart",
            (xy.scatter("x", "y"), xy.legend(render=object())),
            {},
        )


def test_plan_is_a_snapshot_immune_to_later_node_mutation():
    """The registered plan holds a deep copy of what was hashed: mutating a
    reused mark node or the chart props afterwards must not change binding
    behavior behind an unchanged digest/columns record."""
    mark = xy.scatter("x", "y", color="mag")
    props = {"title": "cloud"}
    plan = build_plan("scatter_chart", (mark,), props)
    assert plan.columns == ("x", "y", "mag")

    mark.props["color"] = "sneaky"  # node reused and mutated by page code
    props["title"] = "renamed"
    assert plan.children[0] is not mark
    assert plan.children[0].props["color"] == "mag"
    assert plan.chart_props == {"title": "cloud"}
    # binding still resolves the snapshot's channels, not the mutated node's
    fig = plan.bind({"x": [1.0], "y": [2.0], "mag": [3.0]}).figure()
    assert fig.traces[0].n_points == 1


def test_bind_produces_fresh_figures_per_call():
    """X3: Chart.figure() memoizes, so bind() must mint a fresh Chart —
    two binds with different columns give independent figures."""
    plan = build_plan("scatter_chart", (xy.scatter("x", "y"),), {})
    small = plan.bind({"x": [1.0], "y": [2.0]}).figure()
    large = plan.bind({"x": [1.0, 2.0, 3.0], "y": [2.0, 3.0, 4.0]}).figure()
    assert small is not large
    assert small.traces[0].n_points == 1
    assert large.traces[0].n_points == 3


def test_bind_error_names_both_sides():
    plan = scatter_plan()
    with pytest.raises(
        PlanBindError, match=r"plan binds column 'mag'; Dash.cloud produced \{x, y\}"
    ):
        plan.bind({"x": [1.0], "y": [2.0]}, source="Dash.cloud")


def test_probe_collects_the_tailwind_inventory():
    plan = build_plan(
        "scatter_chart",
        (xy.scatter("x", "y"), xy.legend(class_name="max-h-24 overflow-y-auto")),
        {"class_name": "rounded-xl"},
    )
    assert "rounded-xl" in plan.tailwind_classes
    assert "max-h-24" in plan.tailwind_classes


def test_registry_lookup_and_miss():
    plan = scatter_plan()
    assert plan_of(plan.digest) is plan
    assert plan_of("feedfacefeedfacefeed") is None
    with pytest.raises(PlanMissError, match="feedfacefeedfacefeed"):
        require_plan("feedfacefeedfacefeed")


# Cross-process determinism of the content address.
#
# `_code_fingerprint`'s docstring promises a fingerprint "deterministic across
# processes for one interpreter version": every backend worker that re-runs a
# page's `component()` derives the same plan digest and registers it under
# that digest in its process-local `_PLANS` map (X4). `repr(set(...))` /
# `repr(frozenset(...))` iterate in per-process hash order (PYTHONHASHSEED), so
# any default — or bytecode constant — bearing a set/frozenset used to make
# the fingerprint, and therefore the digest, differ per worker. The frontend
# compile worker then baked digest A into the page while backend workers
# registered the plan under digest B, and the client's resync retries against
# the same dead digest left the chart permanently blank. These tests pin the
# fix by running the real repo function in child interpreters with explicit
# *and* random hash seeds.

# Explicit seeds plus two unset (random) runs: canonicalization must make the
# per-process hash order irrelevant, so all five outputs coincide.
_HASH_SEEDS = [0, 1, 2, None, None]

_SET_BEARING_DEFAULTS = {
    "frozenset_default": (
        "def fn(v, allowed=frozenset({'OPEN', 'SHIPPED', 'PENDING'})):\n    return v\n"
    ),
    "set_kwdefault": "def fn(*, v, allowed={'x', 'y', 'z'}):\n    return v\n",
    "tuple_of_frozenset": ("def fn(v, opts=(frozenset({'a', 'b', 'c'}), 5)):\n    return v\n"),
    "dict_of_frozenset": ("def fn(v, opts={'k': frozenset({'p', 'q'}), 'n': 7}):\n    return v\n"),
    "nested_tuple_of_sets": (
        "def fn(v, opts=({'k': frozenset({'a', 'b'}), 'set': {1, 2, 3}},)):\n    return v\n"
    ),
}


@pytest.mark.parametrize(
    "src", list(_SET_BEARING_DEFAULTS.values()), ids=list(_SET_BEARING_DEFAULTS)
)
def test_code_fingerprint_is_cross_process_stable_for_set_defaults(src):
    """The X4 contract: one callable, one fingerprint in every worker,
    independent of PYTHONHASHSEED — for set/frozenset defaults and defaults
    nesting them in tuples/dicts. The in-process value must agree too, so the
    children are not silently converging on a wrong constant."""
    local: dict = {}
    exec(src, local)
    in_process = _code_fingerprint(local["fn"])
    across_seeds = {_fingerprint_in_child(src, hash_seed=seed) for seed in _HASH_SEEDS}
    assert across_seeds == {in_process}, (
        f"fingerprint diverged across hash seeds or from in-process: "
        f"{across_seeds!r} vs in-process {in_process!r}"
    )


def test_code_fingerprint_canonicalizes_container_constants():
    """The same hazard lives in bytecode constants: CPython folds a constant
    frozenset literal into ``co_consts``, and its repr is per-process hash
    ordered. The fingerprint must canonicalize it (the pre-refactor dedicated
    frozenset branch is now unified with the defaults path through
    ``_canonical_bytes``)."""
    src = "def fn(v):\n    return frozenset({'a', 'b', 'c'})\n"
    local: dict = {}
    exec(src, local)
    in_process = _code_fingerprint(local["fn"])
    across_seeds = {_fingerprint_in_child(src, hash_seed=seed) for seed in _HASH_SEEDS}
    assert across_seeds == {in_process}


def test_code_fingerprint_stays_sensitive_to_set_contents():
    """Canonicalization removes per-process hash order, not content: a
    different set/frozenset default must still change the fingerprint, or the
    address would no longer track behavior (a rolling deployment could swap
    reducers behind one digest)."""
    variants = {
        "OPEN_SHIPPED": "def fn(v, allowed=frozenset({'OPEN', 'SHIPPED'})):\n    return v\n",
        "OPEN_CANCELLED": "def fn(v, allowed=frozenset({'OPEN', 'CANCELLED'})):\n    return v\n",
        "extra_element": "def fn(v, allowed=frozenset({'OPEN', 'SHIPPED', 'EXTRA'})):\n    return v\n",
    }
    fingerprints = {_fingerprint_in_child(src, hash_seed=0) for src in variants.values()}
    assert len(fingerprints) == len(variants)


_REDUCER_SRC = (
    "def reducer(values, modes=frozenset({'mean', 'robust'})):\n"
    "    if not len(values):\n"
    "        return 0.0\n"
    "    if 'mean' in modes:\n"
    "        return float(np.mean(values))\n"
    "    return float(np.median(values))\n"
)


def _plan_digest_in_child(reducer_src: str, *, hash_seed) -> str:
    code = (
        "import sys, types\n"
        f"sys.path.insert(0, {_PYTHON_DIR!r})\n"
        "import numpy as np\n"
        "import xy\n"
        "from reflex_xy.plan import build_plan\n"
        "mod = types.ModuleType('setbearing_reducers')\n"
        "mod.__name__ = 'setbearing_reducers'\n"
        "sys.modules['setbearing_reducers'] = mod\n"
        f"exec({reducer_src!r}, mod.__dict__)\n"
        "reducer = mod.__dict__['reducer']\n"
        "plan = build_plan('hexbin_chart', "
        "(xy.hexbin('a', 'b', C='w', reduce_C_function=reducer),), {})\n"
        "sys.stdout.write(plan.digest)\n"
    )
    return _run_child(code, hash_seed=hash_seed)


def test_plan_digest_is_stable_across_workers_for_set_bearing_reducer():
    """The end-to-end failure mode: a hexbin chart whose pure-Python reducer
    has a set/frozenset default must reach the *same* plan digest in every
    worker. The frontend compile worker embeds the digest in the page;
    backend-only workers register the plan under it at startup. Before the
    fix, distinct ``PYTHONHASHSEED`` values produced distinct digests, the
    subscription missed ``_PLANS`` in every worker that didn't compile the
    page, and the client burned its resync budget retrying the dead digest —
    a permanently blank chart with only a console warning.

    Two assertions pin the contract: every child worker derives one digest,
    and the in-process digest (the one the frontend compile worker bakes) is
    in that set (frontend ``A`` == backend ``B``)."""
    mod = types.ModuleType("setbearing_reducers")
    mod.__name__ = "setbearing_reducers"
    mod.__dict__["np"] = np
    sys.modules["setbearing_reducers"] = mod
    try:
        exec(_REDUCER_SRC, mod.__dict__)
        plan = build_plan(
            "hexbin_chart",
            (xy.hexbin("a", "b", C="w", reduce_C_function=mod.__dict__["reducer"]),),
            {},
        )
        across_seeds = {_plan_digest_in_child(_REDUCER_SRC, hash_seed=seed) for seed in _HASH_SEEDS}
        assert len(across_seeds) == 1, f"plan digest varied across workers: {across_seeds!r}"
        assert plan.digest in across_seeds, (
            f"in-process digest {plan.digest!r} (the one the frontend compile "
            f"worker bakes) is not the digest backend workers register under "
            f"({across_seeds!r})"
        )
    finally:
        sys.modules.pop("setbearing_reducers", None)
