## [category: regular_expression_prx] PRX regular-expression functions
SAS Perl-regex functions map onto Spark's regex built-ins, but the SAS API is
built around a *compiled pattern handle* that SQL has no equivalent for.

`PRXPARSE('/pat/flags')` compiles a pattern and returns an id that later calls
reuse. There is no handle in Spark SQL — **inline the pattern literal at every
use site** and drop the `PRXPARSE` call entirely. When the SAS code compiles
once outside a loop and reuses the id, that is a performance idiom, not
semantics; inlining is still correct.

| SAS | Spark SQL |
|---|---|
| `PRXMATCH(re, s)` | `regexp_instr(s, 'pat')` |
| `PRXMATCH(re, s) > 0` | `s RLIKE 'pat'` |
| `PRXCHANGE(re, -1, s)` | `regexp_replace(s, 'pat', 'repl')` |
| `PRXPOSN(re, n, s)` | `regexp_extract(s, 'pat', n)` |
| `CALL PRXSUBSTR(re, s, pos, len)` | `regexp_instr` + `substr` |
| `PRXPAREN(re)` | no equivalent — restructure |

⚠️ `PRXMATCH` returns the **1-based position** of the match, 0 when there is
none. `regexp_instr` has exactly that convention, so it is the faithful
mapping; `RLIKE` is correct only when the SAS code just tests the result for
truth. Using `RLIKE` where the position was consumed loses data silently.

⚠️ `PRXCHANGE(re, times, s)` takes a replacement **count**: `-1` means all
occurrences and maps to `regexp_replace`, but a positive count (replace the
first *n* only) has no Spark equivalent — flag it rather than silently
replacing everything.

**Pattern syntax.** SAS takes a Perl regex wrapped in delimiters with trailing
flags (`/abc/i`); Spark takes a bare `java.util.regex` pattern string. Strip
the delimiters and translate the flags to inline groups — `/i` becomes
`(?i)`, `/s` becomes `(?s)`, `/x` becomes `(?x)`. Most Perl syntax carries
over unchanged; the constructs that do not are possessive quantifiers,
recursion, and code callouts, none of which Java supports. Remember the
pattern is also a SQL string literal, so a backslash class such as `\d`
survives as-is but an embedded single quote must be doubled.
