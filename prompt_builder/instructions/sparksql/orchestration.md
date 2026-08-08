## [topic] Orchestration: a SAS program is a job, not one script
A SAS program runs top to bottom in one session, so ordering and shared `work`
state are implicit. On Databricks that becomes a **job**: tasks in a DAG, with
`work.*` temp views living only as long as the session that made them.

**Still emit SQL only** — no job definition, bundle, or task JSON. What
orchestration changes is what you must *say*:

- Name cross-step dependencies rather than relying on block order.
- ⚠️ Flag under Risks any step that cannot share a task with its neighbours: it
  reads an external system, its failure should not roll back the others, or a
  different schedule drives it.
- ⚠️ A `work.*` view read by what would become a *different* task will not
  exist there — that step needs a real table, or the two must stay in one task.
  This is the commonest way a working SAS program breaks once split up.
- A value that varies per run (a reporting date, a region) is a job
  **parameter** bound into the query, not a literal — the durable home for what
  a `%LET` at the top of the program held.
