# Releasing

The release workflow (`.github/workflows/release.yml`) publishes through
**Trusted Publishing**: GitHub Actions proves its identity to the index over
OIDC and receives a short-lived token scoped to this one upload. There is no
API token in this repository's secrets — nothing to leak, nothing to rotate,
and nothing that keeps working if someone copies it.

That identity is checked against a publisher you register once per index.
Until you do, the workflow builds and verifies fine and then fails at the
upload step with a permissions error. This page is that setup, and the
release procedure after it.

## One-time setup

### 1. Register the publisher on each index

The project does not exist on either index yet, so both start as a **pending
publisher** — a claim on a name that converts to a normal project on the
first successful upload.

Do this twice, once per index:

| | |
| --- | --- |
| TestPyPI | <https://test.pypi.org/manage/account/publishing/> |
| PyPI | <https://pypi.org/manage/account/publishing/> |

Fill in exactly these values. Every field is compared literally against the
OIDC claim, so a wrong repository name or environment fails closed:

| Field | Value |
| --- | --- |
| PyPI Project Name | `finport-optengine` |
| Owner | `alanvaa06` |
| Repository name | `finport-optengine` |
| Workflow name | `release.yml` |
| Environment name | `testpypi` on TestPyPI, `pypi` on PyPI |

The two indexes are separate accounts. Registering on one does nothing for
the other.

### After renaming the repository

The repository was renamed from `Optimization_Engine` to `finport-optengine`.
A publisher registered under the old name no longer matches the OIDC claim —
GitHub redirects the old URL, but the claim carries the current repository
name — so the next tag builds, verifies, and then has its upload rejected.
**Before pushing that tag**, open the project's publishing settings on each
index (PyPI: *Manage → Publishing*; TestPyPI likewise), add a publisher with
the repository name `finport-optengine` and the other values in the table
above, and remove the one that names `Optimization_Engine`. A manual run of
the workflow (the TestPyPI path) is a free way to confirm the new publisher
before a real release depends on it.

### 2. Create the two GitHub environments

**Settings → Environments**, then add `testpypi` and `pypi`. The names have
to match what you registered above and what the workflow declares.

Leave `testpypi` unprotected — the point of it is to be cheap to run.

On `pypi`, add yourself under **Required reviewers**. A PyPI upload cannot
be undone: yanking a release hides it from resolution but the version number
is burned forever, and the file stays downloadable. A review gate is the
last point at which a mistake is still free.

## Publishing to TestPyPI

**Actions → Release → Run workflow.** That is the whole procedure.

Every manual run publishes `<current version>.devN`, where `N` is the run
number. This is deliberate: TestPyPI rejects a version it has already seen,
and the alternative — skipping the upload and carrying on green — would
leave you believing you had tested an artifact that was never uploaded. A
fresh `.devN` each time means the round trip is always real.

The run does four things in order, and stops at the first that fails:

1. **build** — checks that `pyproject.toml` and `__init__.py` declare the
   same version, stamps the dev suffix, builds the sdist and wheel, runs
   `twine check --strict`, and asserts the wheel actually contains
   `py.typed`.
2. **verify** — installs the **wheel**, not the checkout, with no extras,
   and solves a portfolio end to end. A packaging mistake that leaves a
   module out of the distribution is invisible to tests that run against the
   source tree; this is what catches it. Then installs `[all]` and drives
   the `optengine` console script through a real report.
3. **testpypi** — uploads.
4. **testpypi-install** — installs the version it just published *from
   TestPyPI* and asserts `optimization_engine.__version__` matches. This is
   the only step that proves the artifact is reachable and installable from
   an index rather than from a local file.

### Installing a TestPyPI build by hand

TestPyPI carries none of the real dependencies, so it needs PyPI as a
fallback index or the install fails resolving numpy:

```bash
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  finport-optengine==0.4.0.dev3
```

## Publishing to PyPI

1. Update `CHANGELOG.md`: move what is under `## [Unreleased]` into a new
   version section with today's date.
2. Bump the version **in both places** — `pyproject.toml` and
   `src/optimization_engine/__init__.py`. The build fails if they disagree,
   so a half-done bump cannot ship.
3. Commit, and push to `main`.
4. Tag and push the tag:

   ```bash
   git tag v0.5.0
   git push origin v0.5.0
   ```

The tag is what triggers the PyPI path. The build re-checks that the tag
matches the version in the source and fails if it does not, so a mistyped
tag stops the release instead of publishing under the wrong number.

Two more jobs run on a tag, and the upload waits for both. `on-main` refuses
a tag whose commit is not on `main` (`git merge-base --is-ancestor`), so a
tag pushed from a feature branch cannot publish that branch. `test` runs the
full suite on the tagged commit itself — CI ran on the branch, and this is
the commit being published. Tag `main` after the merge, never before.

Approve the `pypi` environment when GitHub asks, and the upload runs.

### If something goes wrong

A published version cannot be replaced. `pip install finport-optengine`
resolves to whatever is on the index, and re-uploading the same version is
rejected.

- **Bad release, caught early** — yank it on PyPI (Manage → Releases →
  Yank). Existing pins keep working, new resolutions skip it. Then fix
  forward with a new patch version.
- **A secret got published** — yank, then rotate the secret. Assume it was
  scraped; the file remains downloadable after a yank.

Fixing forward is always the answer. There is no version number to reuse.

## Why the publish action is pinned by tag, not SHA

Every other action in the workflow is pinned by commit SHA, with the release
it came from in a comment, and `build` and `twine` are pinned by version. To
bump one, resolve the new tag to its commit (`gh api
repos/actions/checkout/git/ref/tags/v4.4.0`, dereferencing an annotated tag)
and update both the SHA and the comment. The publish action is the one
exception:

```yaml
uses: pypa/gh-action-pypi-publish@v1.14.2
```

Pinning a third-party action by commit SHA is the usual advice, and it is
wrong here — this one is a Docker action. The runner pulls
`ghcr.io/pypa/gh-action-pypi-publish` tagged with whatever ref the `uses:`
line carries, and PyPA publishes that image only under release tags. A SHA
ref resolves to no manifest, and the step dies with `manifest unknown`
before it ever contacts the index. That is exactly how the first run of this
workflow failed, and it is still how v1.14.2 works: its
`create-docker-action.py` names the image after `github.action_ref`.

So the reference has to be a tag PyPA has published an image for. Confirm one
exists before bumping:

```bash
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:pypa/gh-action-pypi-publish:pull&service=ghcr.io" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['token'])")
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://ghcr.io/v2/pypa/gh-action-pypi-publish/tags/list?n=200" \
  | python3 -c "import json,sys; print([t for t in json.load(sys.stdin)['tags'] if t.startswith('v1.')][-10:])"
```

`release/v1` is PyPA's own floating recommendation and also works; an exact
version is preferred here so that the action cannot change under a release
without the change being a commit in this repository.
