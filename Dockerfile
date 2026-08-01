# Verdict's sandbox image (Phase 8): one fat, multi-language image
# DockerSandbox runs agent-influenced code inside. "Fat" over per-language
# images is a deliberate Phase 8 choice — see DESIGN.md's Phase 8 section
# for the tradeoff — with the image itself referenced by version tag
# (`verdict-sandbox:0.1.0`, not `:latest`) so a given Verdict version always
# grades against the same toolchain: reproducible verdicts across rebuilds.
#
# Pinned by tag, not yet by digest — digest-pinning is a follow-up
# hardening step (see DESIGN.md's Phase 8 "deferred" list), tracked but not
# done here.
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYENV_ROOT=/opt/pyenv \
    NVM_DIR=/opt/nvm \
    GO_VERSION=1.22.5 \
    NODE_VERSION=20.15.1 \
    PYTHON_VERSION=3.11.9

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Go ----------------------------------------------------------------
RUN curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-$(dpkg --print-architecture).tar.gz" \
    | tar -C /usr/local -xz
ENV PATH="/usr/local/go/bin:${PATH}"

# --- pyenv + Python ------------------------------------------------------
# The version manager is installed so a later phase can resolve a repo's
# own .python-version pin instead of silently running against whatever
# this image ships (see DESIGN.md's Phase 8 section, "repo pins a
# different version than the image" case) — Phase 8 itself doesn't use it
# for anything beyond installing the one default interpreter below.
RUN curl -fsSL https://pyenv.run | bash
ENV PATH="${PYENV_ROOT}/bin:${PYENV_ROOT}/shims:${PATH}"
RUN pyenv install "${PYTHON_VERSION}" && pyenv global "${PYTHON_VERSION}" \
    && pip install --no-cache-dir --upgrade pip

# --- nvm + Node ------------------------------------------------------
RUN mkdir -p "${NVM_DIR}" && curl -fsSL \
    https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
RUN . "${NVM_DIR}/nvm.sh" && nvm install "${NODE_VERSION}" && nvm alias default "${NODE_VERSION}"
ENV PATH="${NVM_DIR}/versions/node/v${NODE_VERSION}/bin:${PATH}"

# --- Playwright/Chromium ------------------------------------------------
# Installed via npm rather than pip since Node is already present and
# Playwright's own browser-download step is identical either way; the
# `--with-deps` flag pulls the apt-level system libraries Chromium needs
# to run headless inside a container without a real X display.
RUN npm install -g playwright@1.46 \
    && npx playwright install --with-deps chromium

WORKDIR /workspace
CMD ["sleep", "infinity"]
