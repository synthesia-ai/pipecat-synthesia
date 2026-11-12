# Releasing `pipecat-synthesia`

Releases are published to PyPI by `.github/workflows/release.yml`, which runs on a pushed `v*` tag and uses PyPI
**trusted publishing** (OpenID Connect). There is no PyPI API token anywhere in this repository or in GitHub secrets,
and there never should be.

The workflow has two jobs. `build` first checks that the tagged commit is on `main`, then runs the checks and builds
the wheel and sdist; it has no permission to publish. `publish` waits for a human to approve it, then publishes the
files `build` made, each with a PyPI attestation that ties it to this repository and workflow. Only `publish` can mint
the OIDC token PyPI accepts, and it runs no code from the repository.

**The workflow cannot enforce any of this by itself.** On a tag push GitHub runs `release.yml` as it is in the tagged
commit, so whoever pushes the tag chooses the workflow: a tag on a branch commit whose `release.yml` was edited can
drop the `main` check, or put the publish permission back in the job that runs the repository's code. What does enforce it is the `pypi` environment's reviewer, who checks each run before approving it (see
step 6 in section 2), and a tag ruleset that lets only release maintainers create `v*` tags (see 1.3).

---

## 1. One-time human setup

The steps below are done once, by a human, around the first release. Nothing in CI can do them.

### 1.1 PyPI pending trusted publisher

On PyPI, under *Your projects → Publishing → Add a pending publisher*, register:

| Field | Value |
|---|---|
| PyPI project name | `pipecat-synthesia` |
| Owner | `synthesia-ai` |
| Repository name | `pipecat-synthesia` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

A *pending* publisher is the right kind because the project does not exist on PyPI yet; it becomes a normal trusted
publisher when the first release lands.

**If the GitHub organisation or repository is ever renamed, the trusted publisher must be re-registered** — PyPI
matches on the owner/repository pair, not on a redirect, so a rename silently breaks publishing until it is
re-registered.

### 1.2 GitHub environment `pypi`

In the repository settings, create an environment named `pypi` (exactly matching the name registered above) and add a
**required reviewer**. The `publish` job declares `environment: pypi`, so every release pauses for that human approval
before it can mint an OIDC token.

### 1.3 First-release checks

- [ ] **The trusted publisher matches this repository.** Before pushing the first tag, check on PyPI that the
  publisher in 1.1 names the workflow file `release.yml`, the environment `pypi`, and the owner and name of this
  repository, `synthesia-ai/pipecat-synthesia`. A tag pushed to any other copy of this repository cannot publish.
- [ ] **Confirm provenance (attestations) appears on PyPI for the first release.** On PyPI, the details of each file
  in the release should show provenance naming the public repository and `release.yml`. If they do not, the release
  went out without attestations: find out why before the next one.
- [ ] **Know what to do if the first release fails on attestations.** The first release is published while this
  repository is still private, and publishing PEP 740 attestations from a private repository has not been tried
  here. If `publish` fails on them, the `v1.0.0` tag is used up: never move or re-push it. Instead, in a new commit
  on `main`, set `attestations: false` on the publish step of `release.yml`, add a `## 1.0.1` section to
  `CHANGELOG.md`, bump the version to 1.0.1, and tag `v1.0.1`. In the same commit, correct the two sentences that
  promise attestations, in the README's Installation section and in the 1.0.0 changelog entry, since the README
  is also the PyPI page.
  Turn attestations back on in the next release once the repository is public.
- [ ] **If the go-public PR's `integration` check fails for an infrastructure reason**, such as a Docker Hub pull
  limit on livekit-server's image, fix the cause (a read-only Docker Hub token in the org's secrets, or a mirrored
  image). Never merge past `required` with an admin bypass: it is the check that proves the plugin works.
- [ ] **A tag ruleset protects `v*` in the public repository.** Before the first tag, check that a ruleset on
  `refs/tags/v*` restricts creating, updating and deleting those tags to release maintainers. Without it, anyone who
  can push to the repository can start a release, and the reviewer in 1.2 is the only check left.

---

## 2. Per-release steps

1. **Changelog.** Rename the `## Unreleased` heading in `CHANGELOG.md` to `## X.Y.Z` and add a fresh empty
   `## Unreleased` section above it. The release workflow asserts that `CHANGELOG.md` contains a `## <version>`
   section, and CI's `changelog` job already required every `src/` change to touch this file.
2. **Version.** Bump `version` in `pyproject.toml` **and** `__version__` in `src/pipecat_synthesia/version.py` to
   `X.Y.Z`. The two must agree; the release workflow asserts the tag equals `v` + `project.version`.
3. **Commit.** `git commit -am "Release X.Y.Z"`, and get it onto `main`.
4. **Tag.** `git tag vX.Y.Z <commit>`, on a commit on `main` that includes the release commit. The release workflow
   checks that the tagged commit is on `main` and fails if it is not.
5. **Push.** `git push origin vX.Y.Z`.
6. **Approve.** The `build` job checks that the tagged commit is on `main`, re-runs ruff,
   pyright and the unit tests, checks the tag/version/changelog assertions, and runs `uv build`. Then the `publish`
   job stops on the `pypi` environment. It publishes the files `build` made, with attestations, using
   `pypa/gh-action-pypi-publish`. Before you approve it, check the run yourself, because the run's own checks come
   from the tagged commit:
   - [ ] **The run's commit is on `main`.** Take the commit from the run's page, then
     `git fetch origin && git branch -r --contains <sha>` must list `origin/main`.
   - [ ] **The run used the current two-job `release.yml`.** The run has both a `build` job and a `publish` job, and
     `build` succeeded. A run with only a `publish` job comes from an old or edited workflow: reject it.

   If `build` fails on something flaky, *Re-run all jobs* is safe: the rerun replaces the earlier wheel and sdist, and
   `publish` takes the ones from the same run.

After the release, bump `pyproject.toml` and `version.py` to the next `X.Y.Z.dev0` if you want the main branch to stop
claiming to be the released version.

---

## 3. Breaking changes and deprecations

Versioning is **SemVer**: a breaking change to the public API waits for the next major version. The README tells users
to pin `pipecat-synthesia[livekit]>=1.0,<2`, which admits every 1.x and no 2.0.

**Changelog marker.** Every breaking change gets a bullet that starts with `**BREAKING**` in the release's section of
`CHANGELOG.md`, and says what to change in calling code.

**Deprecation wording.** Use Pipecat's canonical format, verbatim except for the names and
versions, and mark the symbol with PEP 702 `@deprecated`:

> `X` is deprecated since A.B.C and will be removed in D.E.F. Use `Y` instead.

**Server-driven notices use logging or `FutureWarning`, not `DeprecationWarning`.** A *client-side* deprecation (a
parameter or class we are removing) raises `DeprecationWarning` through `@deprecated`; a *server-driven* notice (the API telling a running bot that something it relies on is going away) is
emitted with the logger or as a `FutureWarning`, because Python hides `DeprecationWarning` outside `__main__` and the
user would never see it.
