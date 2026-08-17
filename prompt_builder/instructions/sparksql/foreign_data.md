## [when: global_statement:libname] Lakehouse Federation: querying a foreign database in place
When a SAS `LIBNAME` binds a database engine and the decision is to **federate**
rather than ingest, Databricks reaches it through two objects, in this order:

| Object | What it is |
| --- | --- |
| `CREATE CONNECTION` | The credentialled link to the server. Created once, by an operator, and reused. |
| `CREATE FOREIGN CATALOG` | Mirrors that server's databases into Unity Catalog, so its tables are addressable as `catalog.schema.table`. |

```sql
CREATE FOREIGN CATALOG edwprod USING CONNECTION oracle_prod
  OPTIONS (database 'EDWPRO');
```

Once the foreign catalog exists, `edwprod.fr_dm_pro.accounts` is queried like any
other three-level name and the SAS libref maps onto its **schema**.

⚠️ **Do not emit the `CREATE CONNECTION`.** It carries a credential, and the SAS
it would come from almost always spells that credential as a macro variable
(`user="&username." pass="&user_pass."`) — so what you would emit is either a
leaked secret or a wrong one. Name the connection you assumed and put the
creation under Risks.

⚠️ **Federation is not free, and the cost is invisible in the SQL.** A federated
query pushes down what the remote engine can run and pulls back the rest. Filters
and column pruning usually push down; a join across two catalogs, a Spark-only
function, or a `QUALIFY` generally do not, and then the whole table crosses the
network per query. Where the SAS ran a heavy nightly aggregation, say so under
Risks — that is the case where ingesting beats federating, and it is the
operator's call to make.

## [when: global_statement:libname] Foreign data vs ingested data: which the translation assumes
The same SAS reads the same way whichever answer was chosen, so state the
assumption once and keep it consistent:

- **Ingested** — the table is in the lakehouse. Read it directly; emit no
  connection, no foreign catalog, and no load.
- **Federated** — the table stays remote. Read it through the foreign catalog's
  three-level name; emit no load.

Either way the translation contains **no credentials and no ingestion of data
that already exists**. If which one applies cannot be determined from the SAS,
assume ingested — it is the one that produces portable SQL — and flag the
assumption.
