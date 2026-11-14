# Release preparation and publication procedure

TrustTab is not published to PyPI. This guide describes the future release
process; it does not authorize a release or change repository settings.

## Before requesting release approval

1. Review the changelog, supported Python range, dependency groups, license,
   and public API changes.
2. Confirm the research outputs and package artifacts are preserved and the
   full test suite passes on Python 3.11 and 3.12.
3. From a clean checkout, run `python -m build --sdist --wheel` and inspect
   both archives. Install the wheel into a clean environment outside the
   checkout and verify `import trusttab`, `trusttab --help`, and the offline
   example.
4. Upload the same artifacts to TestPyPI and install them in another clean
   environment. TestPyPI validation is a release rehearsal, not a substitute
   for verifying the exact PyPI artifacts.
5. Confirm the TabPFN package/weights licensing notes do not imply that the
   restricted model weights are bundled or commercially licensed.

PyPA recommends building and distributing both an sdist and wheel; these
serve source-build and ready-to-install use cases respectively. See the
[Python Packaging User Guide](https://packaging.python.org/en/latest/discussions/package-formats/).

## Build and inspect locally

```bash
python -m pip install ".[dev]"
python -m build --sdist --wheel --outdir dist
python -m twine check dist/*
```

Install artifacts into clean environments outside the repository, for example
with a new `venv` and `python -m pip install /absolute/path/to/trusttab-*.whl`.
Do not test an installed package while the current directory is the repository
root, since that can import the checkout instead of the installed artifact.

## Secure publication

After explicit owner approval and creation/configuration of the PyPI project,
prefer PyPI Trusted Publishing from a protected GitHub Actions release
workflow. Configure the publisher in the PyPI project settings, use a protected
GitHub environment and environment approval, grant only the required
`id-token: write` permission to the publishing job, and publish only previously
built and validated artifacts. Do not store a long-lived PyPI token in the
repository. PyPI documents Trusted Publishing as OIDC-based short-lived
credentials that avoid managing an upload token; see the [PyPI Trusted
Publishers guide](https://docs.pypi.org/trusted-publishers/) and [publishing
guide](https://docs.pypi.org/publishing/publishing-to-pypi/).

No PyPI or TestPyPI upload, release tag, trusted-publisher configuration, or
publication workflow is performed by this repository preparation step.
