## [topic] Orchestration: a SAS program is a job, not one script
A SAS program runs steps top to bottom in one session, so ordering and shared
`work` state are implicit. On Databricks that becomes a **job**: tasks in a
DAG, each with its own dependencies, and `work.*` temp views living only as
long as the session that made them.

**Still emit SQL only.** Do not write a job definition, a bundle, or task JSON
— the translation's output is the SQL a task runs. What orchestration changes
is what you must *say* about it:

- Make cross-step dependencies explicit. A step reading a dataset an earlier
  step produced is an edge in that DAG; name it rather than relying on the
  order of the blocks.
- ⚠️ Flag under Risks any step that cannot share a task with its neighbours:
  one that reads an external system, one whose failure should not roll back the
  others, one that a different schedule drives, or one whose `work.*` inputs
  would not exist in a fresh session.
- A value that varies per run — a reporting date, a region — is a job
  **parameter**, bound into the query, not a literal baked into the SQL. That
  is the durable home for what a `%LET` at the top of the program held.
- Note where a `work.*` view is read by what would become a *different* task:
  a temp view does not survive the session, so that dependency needs a real
  table or the two steps need to stay in one task. This is the single most
  common way a working SAS program breaks once split into tasks.
