#!/usr/bin/env -S uv run --script

# /// script
# dependencies = ["nox>=2025.10.16", "tomli-w>=1.2"]
# ///

"""Task runner sessions, so running the tooling is not a remembered ritual.

repo-review's PY007 asks for one of these, and without it every repository the
template renders fails that check. Each session names the tooling a change is
checked with, so `nox` runs the default set and `nox -s mutants` runs one. uv is
the backend, which reuses the environment uv.lock pins rather than resolving a
second one.

The shebang, the PEP 723 block, needs_version and the main block below are what
NOX101 and NOX201 through NOX203 ask for, and together they let this run as
./noxfile.py on a machine that has uv and nothing else. The default set is
marked on each session rather than listed in nox.options.sessions, which is
NOX103: a list there repeats every session name a second time and goes stale
when one is renamed.

reuse_venv is "yes", which nox's usage page documents as the preferred spelling
of -r: an environment that exists is kept rather than rebuilt, and the install
step still runs, where `uv sync` against an unchanged lock file returns in
well under a second. Rebuilding on every run cost 25 seconds against 12 and
made `-R` a flag to remember; -N rebuilds when one is wanted.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import nox
import tomli_w

nox.needs_version = ">=2025.10.16"
nox.options.default_venv_backend = "uv|virtualenv"
nox.options.reuse_venv = "yes"

_MUTATION_CONFIG = "mutation.toml"
_MUTATION_SESSION = "session.sqlite"


def _sync(session: nox.Session) -> None:
    """Install the project and its dependency groups into the session.

    Args:
        session: The nox session to install into.
    """
    session.run_install(
        "uv",
        "sync",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )


@nox.session
def lint(session: nox.Session) -> None:
    """Run the pre-commit hooks over what has changed since HEAD.

    The whole tree took thirteen seconds of a loop that otherwise costs three,
    and most of it went to files no edit had touched. `git ls-files` lists the
    modified and the untracked, which is what pre-commit's --files takes.
    `nox -s lint -- --all-files` is the whole tree, which is what CI runs.

    Args:
        session: The nox session running this.
    """
    session.install("pre-commit")
    if session.posargs:
        session.run("pre-commit", "run", *session.posargs)
        return
    listed = session.run(
        "git",
        "ls-files",
        "--modified",
        "--others",
        "--exclude-standard",
        external=True,
        silent=True,
    )
    changed = listed.split() if isinstance(listed, str) else []
    if not changed:
        session.log("nothing has changed since HEAD")
        return
    session.run("pre-commit", "run", "--files", *changed)


@nox.session
def tests(session: nox.Session) -> None:
    """Run the test suite, leaving the benchmarks to the bench session.

    Args:
        session: The nox session running this.
    """
    _sync(session)
    session.run("pytest", "--benchmark-skip", *session.posargs)


@nox.session(default=False)
def generate(session: nox.Session) -> None:
    """Write the equivalence test for a pair of implementations.

    `nox -s generate -- pkg.baselines.<old> pkg.<module>.<new>` runs the
    ghostwriter and writes what it prints to tests/test_<module>.py, taking the
    module from the second name. Redirecting the command by hand puts an edited
    file one keystroke away, and an edited generated file is the drift this
    repository keeps out of its tests.

    The ghostwriter prints its imports in its own order and its calls at its
    own width, so the first lint over the file always rewrote it and reported
    the rewrite as a failure. The two ruff hooks run over the written file
    here, through pre-commit so the version is the one the hook config pins.
    pre-commit exits 1 when a hook changes a file, which is the outcome wanted
    here rather than an error.

    Args:
        session: The nox session running this.
    """
    if len(session.posargs) != 2:
        session.error("name two functions, as `nox -s generate -- pkg.baselines.old pkg.mod.new`")
    old, new = session.posargs
    module = new.split(".")[-2]
    _sync(session)
    session.install("pre-commit")
    source = session.run(
        "hypothesis",
        "write",
        "--style=pytest",
        "--equivalent",
        old,
        new,
        silent=True,
    )
    if not isinstance(source, str):
        session.error("the ghostwriter printed nothing")
    written = Path("tests", f"test_{module}.py")
    written.write_text(source, encoding="utf-8")
    for hook in ("ruff-check", "ruff-format"):
        session.run("pre-commit", "run", hook, "--files", str(written), success_codes=[0, 1])
    session.log(f"wrote {written}")


@nox.session(default=False)
def fast(session: nox.Session) -> None:
    """Run the tests a change affects, rather than all of them.

    pytest-testmon reads the dependency database it wrote on the previous run
    and selects the tests that executed a line which has since changed. The
    first run collects and therefore runs everything. The benchmarks are left
    out for the reason the tests session leaves them out.

    Args:
        session: The nox session running this.
    """
    _sync(session)
    session.run("pytest", "--testmon", "--benchmark-skip", *session.posargs)


@nox.session
def typing(session: nox.Session) -> None:
    """Type-check the tree with the checker the pre-commit config runs.

    Args:
        session: The nox session running this.
    """
    _sync(session)
    session.install("pyright")
    session.run("pyright")


@nox.session(default=False)
def bench(session: nox.Session) -> None:
    """Time an implementation against the one it replaced.

    pytest-benchmark reports median, operations per second, standard deviation
    and rounds, which is what a choice between implementations rests on. A
    difference smaller than the spread is not a difference. A comparison made
    once runs through pyperf instead, which verification.md prescribes and
    which reads no file.

    --benchmark-only skips every test that takes no benchmark fixture, and
    pytest exits zero when the whole suite is skipped, so a repository without
    a benchmark reported a passing session that measured nothing.
    --benchmark-json is written only when a benchmark ran, and its absence is
    reported here as a skip rather than as a pass.

    The plugin warns `Not saving anything, no benchmarks have been run!` in
    that case, and filterwarnings in pyproject.toml turns every warning into an
    error, which would end the session before the report is read. -W ignores
    that one class, so the outcome comes from the report.

    Args:
        session: The nox session running this.
    """
    _sync(session)
    report = Path(session.create_tmp()) / "benchmarks.json"
    session.run(
        "pytest",
        "--benchmark-only",
        "--benchmark-columns=median,ops,stddev,rounds",
        f"--benchmark-json={report}",
        "-W",
        "ignore::pytest_benchmark.logger.PytestBenchmarkWarning",
        *session.posargs,
    )
    if not report.is_file() or not json.loads(report.read_text(encoding="utf-8"))["benchmarks"]:
        session.skip("no test takes the benchmark fixture, so there is nothing to measure")


@nox.session(default=False)
def mutants(session: nox.Session) -> None:
    """Mutate the code and report which mutants the tests fail to kill.

    A suite that passes says nothing about what it would catch; a surviving
    mutant is a change no test noticed.

    The module is a positional argument, `nox -s mutants -- <module>`, because
    mutation.toml naming one module went stale as soon as the work moved on. In
    the first repository to run this it still named a module whose body was one
    call to a standard library function, holding no operator to mutate, and the
    session reported `total jobs: 0` and passed. The module is found under src
    by glob, so the package name is read from the tree rather than repeated
    here, and its tests are tests/test_<module>.py by the same convention.
    Everything else in mutation.toml, the timeout and the distributor, is read
    from the file, and only these two keys are rewritten, into a copy under the
    session's temporary directory.

    cr-filter-pragma runs between init and exec. It is cosmic-ray's own filter
    for the `# pragma: no mutate` comment, which is how a line records that
    mutating it proves nothing. Without it that exclusion has nowhere to live
    except the operator regexes in cr-filter-operators, which name mutation
    kinds rather than lines.

    The distributor is the local one, which runs the jobs one after another.
    The http distributor spreads them, and cosmic-ray's concepts page states
    that "multiple workers can't share a single copy of the code; their
    mutations would interfere with one another". Workers started in this
    directory proved that, emptying the module under test mid-run and
    reporting a survivor that a rerun did not find. Its own launcher,
    cr-http-workers, "clones the git repository for each 'worker-url'", which
    on this machine meant ten clones and ten environments to build before the
    first mutant ran, against a sequential run of about a minute. The parallel
    path is for a suite large enough to pay that back, across machines.

    Args:
        session: The nox session running this.
    """
    if not session.posargs:
        session.error("name the module, as `nox -s mutants -- <module>`")
    module = session.posargs[0]
    sources = sorted(Path("src").glob(f"*/{module}.py"))
    tests = Path("tests", f"test_{module}.py")
    if not sources or not tests.is_file():
        session.error(f"expected src/*/{module}.py and {tests}")

    config = tomllib.loads(Path(_MUTATION_CONFIG).read_text(encoding="utf-8"))
    config["cosmic-ray"]["module-path"] = str(sources[0])
    config["cosmic-ray"]["test-command"] = f"uv run pytest -q -x --timeout=5 {tests}"
    rendered = Path(session.create_tmp()) / _MUTATION_CONFIG
    rendered.write_text(tomli_w.dumps(config), encoding="utf-8")

    session.install("cosmic-ray")
    session.run("cosmic-ray", "init", "--force", str(rendered), _MUTATION_SESSION)
    session.run("cr-filter-pragma", _MUTATION_SESSION)
    session.run("cosmic-ray", "exec", str(rendered), _MUTATION_SESSION)
    session.run("cr-report", _MUTATION_SESSION)


@nox.session(default=False)
def review(session: nox.Session) -> None:
    """Check the repository against the Scientific Python Development Guide.

    Args:
        session: The nox session running this.
    """
    session.install("sp-repo-review[cli]")
    session.run("repo-review", ".", *session.posargs)


@nox.session(default=False)
def problem(session: nox.Session) -> None:
    """Run the whole check for one piece of work, in the order it is read.

    The tests first, since a timing or mutation result means nothing while they
    fail, then the timings against the previous implementation, then what the
    tests would have caught.

    nox hands the same posargs to every session it runs, and tests and bench
    pass theirs to pytest, where a module name is neither a path nor a node id
    and pytest exits 4. Each notified session is given the posargs it can use
    and nothing else.

    Args:
        session: The nox session running this.
    """
    if not session.posargs:
        session.error("name the module, as `nox -s problem -- <module>`")
    session.notify("tests", posargs=[])
    session.notify("bench", posargs=[])
    session.notify("mutants", posargs=[session.posargs[0]])


if __name__ == "__main__":
    nox.main()
