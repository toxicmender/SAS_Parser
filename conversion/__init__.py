"""SharePoint-driven conversion runs: request rows, source files, uploads.

The domain layer over :mod:`app_config.sharepoint`. That module owns the
transport — folders, files, list items, and the addressing that reaches them.
This package owns what those things *mean* for a conversion run:

* :mod:`conversion.paths` — the folder conventions an application's scripts
  live under, all built on
  :meth:`app_config.sharepoint.SharePointConfig.drive_path` so no segment is
  concatenated by hand.
* :mod:`conversion.requests` — the request and conversion lists, and the
  SharePoint column names their rows are read through.
* :mod:`conversion.sources` — discovering and loading the SAS sources for one
  application.
* :mod:`conversion.upload` — writing converted scripts and validation
  artefacts back, including the notebook rendering (delegated to
  :mod:`pipeline.notebook`, never reimplemented).
* :mod:`conversion.run` — the orchestration over all four: read a row's
  scripts, translate the application as one corpus, upload, and write the
  row's ``Status``. It takes a transport and a pipeline *factory*, so a whole
  run is testable without a network or an LLM.

The split is deliberate and runs both ways: the transport stays free of domain
knowledge, and nothing here touches Microsoft Graph directly. Everything takes
an optional ``client`` so a run can be driven against a fake transport in
tests, defaulting to the process-wide
:func:`app_config.sharepoint.get_sharepoint_client`.

This replaces the old ``app_config.powerapps`` module, which modelled the same
concept — a request row with a selected model — against a list that does not
exist in the deployment.

Logger names: ``conversion.*``.
"""

from __future__ import annotations

from .paths import (
    converted_scripts,
    original_scripts,
    upload_target,
    validation,
)
# `requests` (the function listing every row) is deliberately NOT re-exported:
# the name would shadow the `conversion.requests` submodule on this package,
# and `from conversion import requests` reading as either one is worse than
# spelling the module out. Reach it as `conversion.requests.requests`.
from .requests import (
    ConversionItem,
    ConversionRequest,
    conversion_items,
    format_conversion_item_params,
    format_request_item_params,
    pending_requests,
    update_request_status,
)
from .run import RunOutcome, model_for, run_request, select_requests, utc_stamp
from .sources import load, source_files
from .upload import upload_converted_script, upload_validation_file

__all__ = [
    "ConversionItem",
    "ConversionRequest",
    "RunOutcome",
    "conversion_items",
    "converted_scripts",
    "format_conversion_item_params",
    "format_request_item_params",
    "load",
    "model_for",
    "original_scripts",
    "pending_requests",
    "run_request",
    "select_requests",
    "source_files",
    "update_request_status",
    "upload_converted_script",
    "upload_target",
    "upload_validation_file",
    "utc_stamp",
    "validation",
]
