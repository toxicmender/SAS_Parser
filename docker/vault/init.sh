#!/usr/bin/env bash
# Provision the dev Vault the way app_config.vault expects to find it.
#
# Idempotent — compose re-runs it on every `up`, and a dev-mode Vault starts
# empty after every restart, so it must be safe either way.
#
# What it creates
# ---------------
#   secret/            KV v2 (dev mode mounts this already; enabled if not)
#   secret/appsvc/ai_gateway   api_key [+ base_url]  <- the DEFAULT path
#                              (app_config.vault.AI_GATEWAY_PATH); read by
#                              demo_run.py when no --vault-secret is passed
#   secret/llm/anthropic       api_key               <- the explicit path used
#                              by `demo_run.py ... --vault-secret llm/anthropic`
#   auth/approle       with role `sas-parser`, policy-scoped to read the two
#                      secrets above
#
# The role's credentials are *pinned* to DEV_VAULT_ROLE_ID / DEV_VAULT_SECRET_ID
# (Vault's custom-role-id and custom-secret-id endpoints) rather than generated.
# That is what lets docker-compose.yml hand the same pair to the app as plain
# environment: generated ones would have to be passed through a shared volume,
# and a volume only reaches PID 1 — `docker compose exec`, the documented way
# to drive the CLI, would start without them.
#
# DEV ONLY. A real deployment neither pins a secret_id to a known constant nor
# runs Vault with an in-memory, auto-unsealed dev server; there the AppRole
# credentials are delivered by the platform (or the azuread/JWT chain is used
# instead, which needs a real Entra ID tenant). See docker/README.md.
set -euo pipefail

export VAULT_ADDR="${VAULT_ADDR:-http://vault:8200}"
export VAULT_TOKEN="${VAULT_TOKEN:-${VAULT_DEV_ROOT_TOKEN_ID:-root}}"

POLICY_NAME="${POLICY_NAME:-sas-parser}"
ROLE_NAME="${ROLE_NAME:-sas-parser}"
MOUNT="${VAULT_KV_MOUNT:-secret}"

log() { echo "vault-init: $*"; }

# --- wait for the server ----------------------------------------------------
# `vault status` exits 2 while sealed and non-zero while unreachable; dev mode
# unseals itself, so anything other than 0 here just means "not up yet".
for _ in $(seq 1 60); do
    if vault status >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
vault status >/dev/null || { log "Vault at $VAULT_ADDR never became ready"; exit 1; }
log "Vault is up at $VAULT_ADDR"

# --- KV v2 ------------------------------------------------------------------
if vault secrets list -format=json | jq -e --arg m "$MOUNT/" '.[$m]' >/dev/null; then
    log "kv mount '$MOUNT/' already present"
else
    vault secrets enable -path="$MOUNT" -version=2 kv
    log "enabled kv-v2 at '$MOUNT/'"
fi

# --- seed the secrets -------------------------------------------------------
# The placeholder is intentionally not a plausible key: a run that reaches the
# gateway with it fails at the API with a clear 401 rather than looking like a
# Vault problem.
API_KEY="${OPENAI_API_KEY:-sk-placeholder-set-OPENAI_API_KEY-in-.env}"

if [ -n "${OPENAI_BASE_URL:-}" ]; then
    vault kv put "$MOUNT/appsvc/ai_gateway" \
        api_key="$API_KEY" base_url="$OPENAI_BASE_URL" >/dev/null
else
    # No base_url field: app_config.vault then leaves the endpoint to
    # config.json's llm_client.base_url, which is the documented behaviour.
    vault kv put "$MOUNT/appsvc/ai_gateway" api_key="$API_KEY" >/dev/null
fi
log "wrote $MOUNT/appsvc/ai_gateway"

vault kv put "$MOUNT/llm/anthropic" api_key="$API_KEY" >/dev/null
log "wrote $MOUNT/llm/anthropic"

# --- policy -----------------------------------------------------------------
# KV v2 puts the data under <mount>/data/<path>; the metadata path is granted
# read-only so `vault kv get` can resolve versions.
vault policy write "$POLICY_NAME" - <<POLICY >/dev/null
path "$MOUNT/data/appsvc/*" {
  capabilities = ["read"]
}

path "$MOUNT/data/llm/*" {
  capabilities = ["read"]
}

path "$MOUNT/metadata/*" {
  capabilities = ["read", "list"]
}
POLICY
log "wrote policy '$POLICY_NAME'"

# --- AppRole ----------------------------------------------------------------
if vault auth list -format=json | jq -e '."approle/"' >/dev/null; then
    log "approle auth already enabled"
else
    vault auth enable approle
    log "enabled approle auth"
fi

vault write "auth/approle/role/$ROLE_NAME" \
    token_policies="$POLICY_NAME" \
    token_ttl="${APPROLE_TOKEN_TTL:-1h}" \
    token_max_ttl="${APPROLE_TOKEN_MAX_TTL:-4h}" \
    secret_id_ttl="${APPROLE_SECRET_ID_TTL:-24h}" \
    secret_id_num_uses=0 >/dev/null
log "wrote approle role '$ROLE_NAME'"

# Pin the credentials the compose file already gave the app.
ROLE_ID="${DEV_VAULT_ROLE_ID:?DEV_VAULT_ROLE_ID is required}"
SECRET_ID="${DEV_VAULT_SECRET_ID:?DEV_VAULT_SECRET_ID is required}"

# role-id is a plain upsert.
vault write "auth/approle/role/$ROLE_NAME/role-id" role_id="$ROLE_ID" >/dev/null
log "pinned role_id=$ROLE_ID"

# custom-secret-id is NOT: re-registering a value that is already there fails
# with "SecretID is already registered" (HTTP 500). So the login is what
# decides — it doubles as the check that the pair really works, which is worth
# doing on every run anyway. A Vault that restarted has forgotten the secret_id
# and lands in the `else`.
approle_login() {
    vault write -format=json auth/approle/login \
        role_id="$ROLE_ID" secret_id="$SECRET_ID" 2>/dev/null \
        | jq -e '.auth.client_token != null' >/dev/null
}

if approle_login; then
    log "secret_id already registered; AppRole login verified"
else
    vault write "auth/approle/role/$ROLE_NAME/custom-secret-id" \
        secret_id="$SECRET_ID" >/dev/null
    approle_login || { log "AppRole login failed after registering secret_id"; exit 1; }
    log "pinned secret_id (from DEV_VAULT_SECRET_ID); AppRole login verified"
fi
log "done"
