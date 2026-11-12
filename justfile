# Developer shortcuts.
#
# Every recipe is the command CI runs, verbatim, so "green locally" and "green in CI" mean the same
# thing. `just --list` shows them all; it prints the LAST comment line above each recipe, so that line
# is the summary and anything above it is the detail.

# Use this instead of a bare `uv sync`. Plain `uv sync` installs no optional extras, so it
# UNINSTALLS tenacity and livekit (measured: "Would uninstall 7 packages") and leaves a venv where
# `pipecat_synthesia.livekit` cannot import — which reads like a broken checkout rather than a
# missing extra. uv 0.12 has no `default-extras` setting to fix this in pyproject.toml (adding one
# is worse than the disease: the unknown field makes uv discard the whole [tool.uv] table, taking
# with it the index override that, if your pip/uv config points at a private index, keeps public
# PyPI as the source). So the extra has to be named explicitly, every time.
# Install the development environment.
sync:
    uv sync --extra livekit

# zizmor runs offline here unless GH_TOKEN is set. CI sets it, which adds the audits that ask GitHub
# about each pinned action.
# Lint, format check, type check and workflow audit — the CI `lint` job.
lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run pyright
    uv run --no-project --with zizmor==1.31.1 zizmor .github/

# No docker, no network; this is the CI `unit` job's selection.
# Run the unit tier.
test:
    uv run pytest -m "not integration and not integration_extended and not staging and not quarantine"

# Needs a livekit-server: run `just livekit-up` first, or let the fixture start one with docker.
# Budget is 3 min including container start.
# Run the PR integration gate.
test-integration:
    uv run pytest -m integration --timeout=120

# Timing-relative and multi-client tests, plus whatever is in quarantine (tests that flaked twice).
# Run the nightly / `integration-extended`-label tier.
test-integration-extended:
    uv run pytest -m "integration_extended or quarantine" --timeout=120

# The fixture reuses a server that is already answering on 7880, so leaving one running saves the
# container start on every integration run. It fails instead if the server's TCP media port is closed:
# a container started with the older command published only 7880; `just livekit-down`, `just livekit-up`.
# keep in sync with tests/integration/conftest.py and .github/actions/livekit-dev/action.yml
# Start the dev livekit-server and wait for it.
livekit-up:
    docker run -d --rm --name livekit -p 7880:7880 -p 7881:7881/tcp -p 7882:7882/udp livekit/livekit-server:v1.9.1 --dev --bind 0.0.0.0 --node-ip 127.0.0.1
    for i in $(seq 1 40); do curl -sf http://localhost:7880/ >/dev/null && exit 0; sleep 0.5; done; docker logs livekit; exit 1

# Stop the dev livekit-server started by `livekit-up`.
livekit-down:
    docker stop livekit
