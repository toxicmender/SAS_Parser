# V2 deployment smoke

The deployment smoke is an operational v2 command and a CI gate, not a
source-tree unit test. `sas-migrate smoke` exercises a credential-free path
through the installed application:

1. discover the installed `sas-parser` distribution;
2. load the packaged schema-v2 resource;
3. parse one SAS DATA step with the v2 semantic chunker;
4. resolve Spark SQL and require the SQLGlot `databricks` dialect;
5. normalize a raw-response fallback and run mandatory target validation;
6. build a deterministic validation report with SAS source, instruction, and
   output token attribution.

`docker/v2.Dockerfile` builds the project wheel in a builder stage and copies
only its virtual environment into the runtime stage. The runtime has no source
checkout or build tool, uses UID/GID 10001, and exposes the smoke as both its
default command and health check. The CI job adds a read-only filesystem,
drops Linux capabilities, and enables `no-new-privileges`.

Run the application contract locally:

```bash
uv run sas-migrate smoke --json
```

Build and exercise the deployment boundary exactly as CI does:

```bash
docker build -f docker/v2.Dockerfile -t sas-parser/v2-deployment:local .
docker run --rm --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=64m \
  --cap-drop=ALL --security-opt=no-new-privileges \
  sas-parser/v2-deployment:local
```

The strict container command requires both a wheel installation and a non-root
effective user. Its versioned JSON output is uploaded by CI as
`v2-deployment-smoke-report`.

This closes the deployment-smoke portion of G-018. Scheduled, budgeted
real-model quality evaluation remains a separate open gate because this smoke
is intentionally deterministic, offline, and secret-free.
