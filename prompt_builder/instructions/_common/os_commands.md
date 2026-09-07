## [when: global_statement:x, global_statement:systask, global_statement:sysexec, global_statement:waitfor] X, SYSTASK and %SYSEXEC: host commands become dbutils.fs, not a shell
`X 'command';`, `SYSTASK COMMAND "..."`, and `%SYSEXEC command;` hand a string to
the SAS server's operating system. On Databricks there is no such host — the
driver node is ephemeral and its local disk is not shared with the workers — so
these never translate to a shell line. Translate the **intent**.

Most X commands are file management. Use **`dbutils.fs`** for those: one
interface reaching volumes, DBFS and object storage alike, with no FUSE mount
needed.

| SAS X command | Databricks |
|---|---|
| `x 'cp a b';` | `dbutils.fs.cp(a, b)` — add `recurse=True` for a directory |
| `x 'mv a b';` | `dbutils.fs.mv(a, b)` — add `recurse=True` for a directory |
| `x 'rm f';` / `x 'rm -rf d';` | `dbutils.fs.rm(f)` / `dbutils.fs.rm(d, recurse=True)` |
| `x 'mkdir -p d';` | `dbutils.fs.mkdirs(d)` |
| `x 'ls d';` | `dbutils.fs.ls(d)` -> `FileInfo` list; filter it in Python |

⚠️ `SYSTASK COMMAND` is **asynchronous** unless `WAIT` is given, and a later
`WAITFOR` is what joins it. A blocking `subprocess.run` where the SAS did not
block, or the reverse, changes the program. Match the pair:

| SAS | Python |
|---|---|
| `SYSTASK COMMAND "..." WAIT;` | `subprocess.run([...], check=True)` |
| `SYSTASK COMMAND "..." TASKNAME=t;` | `p = subprocess.Popen([...])` |
| `WAITFOR t;` / `WAITFOR _ALL_ t1 t2;` | `p.wait()` / wait on each handle |
| `WAITFOR _ANY_ t1 t2;` | first-to-finish — no one-liner; say so |
| `TIMEOUT=` on either | `timeout=` on `run`/`wait`, and handle the raise |

⚠️ A `WAITFOR` whose `SYSTASK` you did not translate as a live handle has
nothing to wait on. If the task became a blocking call, the `WAITFOR` is
already satisfied and translates to **nothing** — say that, rather than
emitting a wait against a process that was never started.

⚠️ `cat` has no equivalent: `dbutils.fs.head` stops at `maxBytes` (65536 by
default), so it is a preview, not a read. A `cat` that fed data into SAS is a
file read — express it as a table or DataFrame read; one that concatenated files
is a `UNION ALL`.

```python
vol = "/Volumes/main/ops/landing"
dbutils.fs.mkdirs(f"{vol}/archive")
dbutils.fs.cp(f"{vol}/in.csv", f"{vol}/archive/in.csv")
```

**Path resolution is where this goes wrong.** `dbutils.fs` resolves a
scheme-less path in the **DBFS** namespace, not on the driver:

| What the SAS path meant | Name it as |
|---|---|
| A permanent library or landing area | `/Volumes/<catalog>/<schema>/<volume>/...`, or the equivalent `dbfs:/Volumes/...` |
| Scratch on the SAS host (`/tmp/...`) | `file:/tmp/...` — the `file:/` scheme is **required** for a driver-local path. ⚠️ It dies with the cluster, so if anything later reads it, a volume is the right home |
| A mounted share (`/mnt/...`) | `dbfs:/mnt/...` ⚠️ DBFS root and mounts are deprecated; new accounts have no access. Prefer a volume |
| Object storage | the full URI (`abfss://...`) |

⚠️ So `x 'ls /tmp';` is **not** `dbutils.fs.ls("/tmp")` — that lists `dbfs:/tmp`,
a different directory. It is `dbutils.fs.ls("file:/tmp")`. State the path style
you assumed for every path you translate.

`dbutils.fs` has no archive or host-inspection commands. `tar`/`gzip` stay
Python (`shutil.make_archive`, the `gzip` module) over a `/Volumes/...` path,
which ordinary file APIs can read; `which` and `df` describe a host that no
longer exists, so drop them and say so. Where the command genuinely is not file
management, run it as an argument **list** and check the result:

```python
import subprocess
subprocess.run(["aws", "s3", "cp", src, dst], check=True)
```

- ⚠️ **Never `shell=True`, and never `os.system`.** A SAS command string almost
  always interpolates a macro variable — `x "rm -rf /data/&region";` — and
  concatenating that into a shell string is a command-injection hole. Pass the
  value as its own list element so it can never be read as syntax.
- ⚠️ Both `dbutils.fs` and `subprocess` run on the **driver** and are not
  distributed — never call either from a UDF or a `foreach`, and note that
  nothing written to driver-local disk survives the cluster. `dbutils` exists
  only in a notebook or job context; elsewhere it comes from a
  `WorkspaceClient`.
- ⚠️ Volume constraints that bite on a naive port: `dbutils.fs.ls` needs a
  **fully qualified** volume path and cannot list a catalog or schema; the
  `<cat>/<schema>/<volume>` directories are managed by Unity Catalog, so
  `mkdirs` cannot create them; and volumes need Databricks Runtime 13.3 LTS or
  above — on 12.2 and below a write to `/Volumes/...` may *appear* to succeed
  while landing on ephemeral disk.
- Reach for `shutil`/`pathlib` only where `dbutils.fs` has no equivalent. They
  work on driver-local paths and on `/Volumes/...`, but ⚠️ not on `dbfs:/` or
  object-storage URIs, and `/dbfs/Volumes` is reserved and does **not** reach
  volumes.
- ⚠️ **The X statement runs when the DATA step is *compiled*, not where it
  appears**, so a program that looks like "build the file, then move it" may not
  have meant that. Read the ordering from the SAS semantics, not the line order.
  `X` with no argument opens an interactive shell — emit the non-convertible
  marker. And many sites already run `NOXCMD` (the SAS Viya Compute Server
  default), where these statements are invalid at all: if the source leans on
  `X`, note that the behaviour may never have run in the current environment.
- A command that moved data into or out of SAS (an FTP, an S3 copy, a database
  unload) is usually better re-expressed as a table or volume read than as a
  file operation. Say so rather than porting the copy verbatim.

## [when: call_routine:system, function:system] CALL SYSTEM and the SYSTEM function
`CALL SYSTEM('cmd')` and `rc = system('cmd')` are the DATA-step forms of `X`.
They differ from it in one way that matters: they execute **per observation, at
run time**, so they can be conditional and can be called once per row.

- Translate the command itself exactly as for `X` above — `dbutils.fs` for file
  work (with the same path-scheme rules), `subprocess.run([...], check=True)`
  otherwise.
- ⚠️ A row-wise `CALL SYSTEM` inside a DATA step is one process launch **per
  row**. Do not reproduce that shape. Lift it out of the row loop: collect the
  distinct values first and act once per value, and say in Mapping that you did.
- `rc = system(...)` captures the exit status, so the SAS branched on it. Keep
  the branch: `subprocess.run(...).returncode`, or let `check=True` raise where
  the SAS treated a non-zero return as fatal. ⚠️ Do not silently drop a return
  code the source tested.
- ⚠️ Never build the command by concatenating DATA-step variables. Those are row
  values, which is the injection case again — pass them as list arguments.

## [when: global_statement:filename] FILENAME PIPE reads a command's output
`filename p pipe 'cmd';` runs `cmd` and exposes its **stdout** as a file the
following `INFILE` reads, so it is a host command whose output becomes data:

```python
out = subprocess.run(cmd_args, capture_output=True, text=True, check=True).stdout
```

- Every caution on `X` applies — argument list not `shell=True`, driver-only
  execution, no shell on a Databricks cluster to run it against.
- ⚠️ A pipe used to **list files** (`ls`, `find`, `hdfs dfs -ls`) is the common
  case and should not stay a subprocess at all: `dbutils.fs.ls(path)`, filtered
  in Python, under the same path-scheme rules as `X`.
- ⚠️ A pipe used to *write* (`filename p pipe 'cmd'` on a `FILE` statement)
  feeds the command's **stdin** instead — `subprocess.run(args, input=text)`.
  Check which direction the SAS used before translating; they are not the same
  step reversed.
- A pipe reading a database unload or a remote copy is a data movement in
  disguise. Re-express it as a table or volume read rather than porting the
  command, and say so in Mapping.
- A `FILENAME` without the `PIPE` keyword names an ordinary file and none of
  this applies to it.
