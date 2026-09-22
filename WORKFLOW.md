# Working a problem

The tooling does the checking, so none of it is a judgement call. Every command
below is rendered from the template, so a repository describes the work in one
place rather than in a paragraph each repository rewrites.

| Step                                   | Command                                                                                                                    |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Write the obvious solution             | `src/deephedging/<module>.py`                                                                                              |
| Keep it when you rewrite               | move it to `src/deephedging/baselines.py`                                                                                  |
| Generate the test                      | `nox -s generate -- deephedging.baselines.<old> deephedging.<module>.<new>`                                                |
| Narrow the generated strategy          | the annotations do not carry the problem's bounds                                                                          |
| Run everything                         | `nox`                                                                                                                      |
| Re-run only what a change affects      | `nox -s fast`                                                                                                              |
| Time one implementation                | `uv run python -m pyperf timeit --fast --name '<input>' -o <fn>.json -s '<setup>' '<call>'`, `--append` for the next input |
| Compare two implementations            | `uv run python -m pyperf compare_to --table --table-format md <old>.json <new>.json`                                       |
| Track a timing in CI instead           | `nox -s bench`, once a test takes the benchmark fixture                                                                    |
| Measure what the tests would catch     | `nox -s mutants -- <module>`                                                                                               |
| Run the whole check for one piece      | `nox -s problem -- <module>`                                                                                               |
| Check the repository against the guide | `nox -s review`                                                                                                            |
| Profile one implementation             | `uv run --with scalene scalene run <script>`, then `scalene view --cli`                                                    |

## The test is generated

`hypothesis write` writes the test rather than anyone writing assertions by
hand, and `--equivalent` compares a rewrite against the version it replaced. The
generated strategy comes from the type annotations alone, so it carries none of
a problem's bounds. Narrow the strategy, and leave the assertion as generated.

## A passing suite is not a measured one

`nox -s mutants -- <module>` mutates that module and reruns its tests against
each mutant. A surviving mutant is a change no test noticed. The module is named
on the command line because a module named in `mutation.toml` goes stale as soon
as the work moves on, and a run against a module with nothing to mutate reports
`total jobs: 0` and passes.

## Timing

`pyperf timeit` reports a mean with its standard deviation and runs from the
command line, which is why a timing harness under `tests/` is refused by name.
`-o` writes a run to JSON, `--append` adds the next input to the same file, and
`compare_to --table --table-format md` prints the two files as one table with
the ratio in it, which is what a commit message quotes; the files themselves are
not kept, since a timing is a record of the change it justified and the commit
that made it is where that record cannot go stale. The `benchmark` fixture and
`nox -s bench` are for a number rechecked in CI, which is what answering
`track_benchmarks` true renders a workflow for.

Measure at the input the constraints allow, and on the input that is hard for
the implementation rather than the one that is convenient. Random data often
lets the naive version finish early, which reverses the result: over one
poisonous plants pair, random values put the simulation ahead at 1.87 ms against
4.50 ms, while the input forcing one death per day put it behind at 74.8 ms
against 484 us.

## What each check costs, and when to run it

Measured on one repository, five problems, 46 tests.

| Command                      | Time | When                                   |
| ---------------------------- | ---- | -------------------------------------- |
| `nox -s fast`                | 3 s  | after every edit                       |
| `uv run pytest -q`           | 2 s  | when the whole suite is wanted         |
| `nox`                        | 12 s | before leaving a problem               |
| `nox -N`                     | 25 s | when an environment is suspected stale |
| `nox -s mutants -- <module>` | 60 s | once, when a solution's tests are new  |

The noxfile sets `reuse_venv` to yes, so `nox` keeps each environment and
`uv sync` brings it up to date, which against an unchanged lock file is well
under a second; a changed dependency is synced on the same run. `uv sync`
installs the project editable, so a module added or removed is seen on the next
run without a reinstall. `-N` deletes and rebuilds the environments.

Mutation measures whether the tests would catch a wrong edit, which a passing
suite does not report. It earns its minute once per solution: on one problem it
produced wrong answers at `(5, 1, 2, 17)`, `(1, 4, 3, 10)`, `(1, 1, 3, 15)` and
`(1, 4, 9, 36)`, every one inside the box the drawing sampled and every one the
drawing missed. Running it after every edit measures the same thing again.
cosmic-ray's `local` distributor runs the jobs sequentially, so the minute is
one core; its `http` distributor is what parallelises them.

## Committing

Work does not commit. A solution is finished when the checks pass, and the file
on disk is the deliverable. A commit runs the hook set over the staged files and
runs vale over the message, where a rejected draft re-runs every hook, measured
at 10 to 15 seconds clean and 40 to 75 with rejections. The message is checked
on its own first: write the draft to `.git/COMMIT_EDITMSG` and run
`pre-commit run --hook-stage commit-msg --commit-msg-filename .git/COMMIT_EDITMSG`,
which is the same hook over the same styles in half a second. Commit when you
want to, not as part of solving a problem.

## What runs without being asked

The language servers report while an edit is still in flight, ruff and pyright
for Python and Harper, Vale and LTeX+ for prose. `pre-commit` runs the rest at
the commit, and `pre-commit install --install-hooks` puts it there.
