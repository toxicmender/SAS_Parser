# Wheel-only v2 runtime used for deployment verification and as a minimal
# credential-free operational image. The runtime stage contains neither the
# checkout nor build tooling.

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.9.21

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project \
 && uv build --wheel --out-dir /dist \
 && uv pip install --python /opt/venv --no-deps /dist/*.whl

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

RUN groupadd --system --gid 10001 sas-migrate \
 && useradd --system --uid 10001 --gid sas-migrate \
      --create-home --home-dir /home/sas-migrate sas-migrate

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /opt/venv /opt/venv

WORKDIR /workspace
USER 10001:10001

HEALTHCHECK --interval=30s --timeout=20s --start-period=5s --retries=3 \
  CMD ["sas-migrate", "smoke", "--require-wheel", "--require-non-root", "--quiet"]

CMD ["sas-migrate", "smoke", "--require-wheel", "--require-non-root", "--json"]
