# Research reuse and contributions

Start with [RESEARCH_REUSE.md](../RESEARCH_REUSE.md) and the
[case evidence index](../CASE_EVIDENCE_INDEX.md). The submission is a frozen
computational benchmark, not a collection of independently measured experiments.

To extend a case, copy the release into a new working directory, preserve the
archived release, implement the BaseCase interface and register a new case ID.
Document equations, units, domains, coefficients, physical assumptions, solver
settings, boundary conditions and failure predicates. Add an adjacent
`<case_id>.evidence.md` distinguishing verified source passages, adaptations,
author-selected parameters and unresolved attribution. Add tests of an analytic
limit, conservation property or an independent reference where possible.
Agreement with the same implemented expression is a regression test, not
independent validation. Record convergence studies for numerical solvers.

Use fresh output directories and record software versions, backends, seeds,
hardware, thread settings and hashes. Never replace the original results,
repin their source hashes to edited code, or mix outputs from different
implementations under the same release. If a parameter is deliberately inactive
or a failure boundary cannot be reached, disclose that behavior explicitly.

Run the affected case and benchmark tests with `python -m pytest`, then the
appropriate integrity checks in `docs/REPRODUCIBILITY.md`. Before a full new
sweep, use the documented preflight and smoke gates. The historical full sweep
is resource intensive; regenerating its derived analyses is a separate task.

Preserve the MIT licence and third-party notices. Cite the software release and
original method references. Until a public deposit is assigned, do not invent a
repository URL, DOI or published article citation. Report the release hash and
the actual archive used. See CITATION.cff for author and title metadata.
