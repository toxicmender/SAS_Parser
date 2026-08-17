## [when: proc:cas] [kind: PROC_STEP] CAS actions and action sets

Treat a `PROC CAS` block as an ordered CASL program, not as an ordinary SAS
procedure. Identify every action by its `actionSet.action` name and preserve its
input tables, parameter values, result objects, side effects, and table scope.
Do not translate an action into similarly named SQL without documented semantic
parity. When Databricks SQL cannot express the action, emit the non-convertible
marker and record the unresolved action and required output under Risks rather
than inventing a query.

CAS tables can be session-scoped or promoted. A temporary view is not equivalent
to an action that promotes or persists a table, so state the lifetime and
storage assumptions explicitly.
