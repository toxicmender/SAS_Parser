## [when: proc:cas] [kind: PROC_STEP] CAS actions and action sets

Treat a `PROC CAS` block as an ordered CASL program, not as an ordinary SAS
procedure. Identify every action by its `actionSet.action` name and preserve its
input tables, parameter values, result objects, side effects, and table scope.
Do not infer a PySpark equivalent from the action's name alone. Use a DataFrame
or Spark API only when it provides the documented behaviour; otherwise emit the
non-convertible marker and record the unresolved action and required output
under Risks.

CAS tables can be session-scoped or promoted. A replacement that writes a
temporary DataFrame or view is not equivalent to an action that promotes or
persists a table, so state the lifetime and storage assumptions explicitly.
