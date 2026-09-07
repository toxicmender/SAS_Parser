## [when: global_statement:filename] Sending mail: smtplib, not a mail fileref
A `FILENAME <ref> EMAIL` plus a DATA step that writes to it — and equally an
`X 'mailx ...'` or `X 'sendmail ...'` — sends a message through the SAS server's
SMTP configuration. There is no fileref equivalent on Databricks: translate it
to `smtplib` with `email.message.EmailMessage`.

```python
import smtplib
from email.message import EmailMessage
from pathlib import Path

msg = EmailMessage()
msg["From"] = "sas@example.com"
msg["To"] = "ops@example.com"          # several -> one comma-joined string
msg["Subject"] = "Nightly load finished"
msg.set_content("Load completed.")     # the non-directive PUT lines

report = Path("/Volumes/main/ops/out/report.csv")     # attach='...'
msg.add_attachment(
    report.read_bytes(), maintype="text", subtype="csv", filename=report.name
)

with smtplib.SMTP(smtp_host, smtp_port) as smtp:      # job parameters
    smtp.starttls()
    smtp.login(smtp_user, dbutils.secrets.get("ops", "smtp_password"))
    smtp.send_message(msg)
```

`TO=`, `CC=`, `BCC=`, `FROM=`, `SUBJECT=`, `REPLYTO=` become headers of the same
name; `ATTACH=` becomes one `add_attachment` per file; `TYPE='text/html'`
becomes `set_content(body, subtype="html")`. The `PUT` lines that are *not*
directives are the message body — collect them into one string, not a statement
each. The `!EM_*!` directives set the same things at run time:

| Directive | Python |
|---|---|
| `!EM_TO!` / `!EM_FROM!` / `!EM_SUBJECT!` / `!EM_REPLYTO!` | set that header |
| `!EM_ATTACH!` | `msg.add_attachment(...)` |
| `!EM_SEND!` | `smtp.send_message(msg)` **now** |
| `!EM_NEWMSG!` | start a fresh `EmailMessage()` |
| `!EM_ABORT!` | discard it and send nothing |

⚠️ **The implicit send is the trap.** SAS sends the message when the fileref
closes — at the next `FILE` statement or the end of the DATA step — *whether or
not* `!EM_SEND!` appears. A step containing `!EM_SEND!` and nothing after it
sends **twice**, which is why the canonical recipient loop ends `!EM_SEND!`
`!EM_NEWMSG!` `!EM_ABORT!`. Reproduce the number of messages the SAS actually
sent: one `send_message` per send, plus the implicit one only if the SAS ended
with it. A loop over recipients is one connection reused — open the `SMTP`
context once and build an `EmailMessage` per recipient inside it.

- ⚠️ **Never inline credentials.** `OPTIONS EMAILHOST= EMAILPORT= EMAILID=
  EMAILPW=` carry a host and a password. The password comes from a secret scope
  (`dbutils.secrets.get(scope, key)`), never a literal; host and port are job
  parameters. Flag any credential the SAS spelled out.
- An SSL or TLS protocol option on `EMAILHOST` means an encrypted connection —
  `starttls()`, or `smtplib.SMTP_SSL` for implicit TLS. Never silently downgrade
  to plaintext. SAS also defaulted the sender to the session user, an identity
  that does not exist here, so `FROM=` is no longer optional: state the service
  address you assumed.
- ⚠️ **Sending mail is an outward-facing side effect and the recipients are
  real** — a job run for testing will actually mail whoever the SAS named. Flag
  it under Risks and recommend gating the send on an explicit parameter so a
  validation run cannot page the ops list.
- Where the message only reported that a job finished or failed, the durable
  form is the orchestrator's own task notification; offer that rather than
  porting the mailer unchanged.
- A `FILENAME` without the `EMAIL` keyword is an ordinary file reference and
  none of this applies to it.
