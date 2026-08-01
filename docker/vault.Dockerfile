# HashiCorp Vault, plus the tooling the provisioning script needs.
#
# One image serves two compose services:
#   vault       — the server itself (dev mode; see docker-compose.yml)
#   vault-init  — a one-shot that enables kv-v2 + AppRole, seeds the secrets
#                 app_config.vault reads, and hands the app its AppRole
#                 credentials (docker/vault/init.sh)
#
# Build context is the repo root:
#   docker build -f docker/vault.Dockerfile -t sas-parser-vault .

ARG VAULT_VERSION=1.18

FROM hashicorp/vault:${VAULT_VERSION}

# jq parses the AppRole login/secret-id responses; the base image has neither
# it nor curl.
#
# No `USER` directive anywhere in this file, deliberately: the base image runs
# as root and its entrypoint drops to the `vault` user itself (su-exec) after
# applying the binary's file capabilities. Forcing USER vault here makes the
# container die at startup with "unable to set CAP_SETFCAP effective
# capability: Operation not permitted".
RUN apk add --no-cache jq curl bash

COPY docker/vault/init.sh /usr/local/bin/vault-init.sh
RUN chmod +x /usr/local/bin/vault-init.sh

# init.sh drops the AppRole credentials here for the app container to read.
# Owned by `vault` so the path stays writable if the init service is ever run
# as that user: docker seeds a new named volume with the ownership of the
# image directory it is mounted over.
RUN mkdir -p /vault/shared && chown vault:vault /vault/shared

EXPOSE 8200
