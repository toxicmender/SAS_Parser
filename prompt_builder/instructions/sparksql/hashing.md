## [category: hashing_security] Hashing functions
⚠️ **SAS and Spark disagree about the return type, and the mismatch is
silent.** Spark's `md5(x)` and `sha2(x, n)` return a **hex STRING**; SAS
`MD5()` and `SHA256()` return **raw binary**, and only `SHA256HEX()` returns
hex. Translating `MD5(x)` to `md5(x)` therefore produces a 32-character
string where SAS produced 16 bytes — a join on that column silently matches
nothing, and a stored digest silently changes format.

| SAS | Spark SQL |
|---|---|
| `SHA256HEX(x)` | `sha2(x, 256)` |
| `PUT(MD5(x), $hex32.)` | `md5(x)` |
| `MD5(x)` (raw binary) | `unhex(md5(x))` |
| `SHA256(x)` (raw binary) | `unhex(sha2(x, 256))` |
| `HASHING('MD5', x)` | `md5(x)` |
| `HASHING('SHA256', x)` | `sha2(x, 256)` |

Check what the SAS code does with the result before choosing: a value written
straight to a dataset wants the binary form, one wrapped in `PUT(..., $hex.)`
or compared against a hex literal wants the string form.

Two further cautions:
- **Encoding.** Both hash the *bytes* of the input, so the digest depends on
  the session encoding and on trailing blanks. SAS character variables are
  blank-padded to their declared width and Spark `STRING` is not, so
  `MD5(name)` over a `$20.` column hashes the padded value. Reproduce it with
  `rpad(name, 20, ' ')` when the digest must match SAS output, and ⚠️ flag it
  — this is the most common cause of "the same data hashes differently".
- **No HMAC.** `HASHING_HMAC`, `HASHING_HMAC_FILE`, and the incremental
  `HASHING_INIT`/`HASHING_PART`/`HASHING_TERM` sequence have **no Spark SQL
  equivalent**; there is no HMAC built-in. Emit the non-convertible marker and
  note that a UDF or an upstream step is required. `HASHING_FILE` hashes file
  contents, which is not a SQL operation at all.
