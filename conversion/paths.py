"""Where an application's scripts live in the document library.

Four conventions, all relative to
:attr:`~app_config.sharepoint.SharePointConfig.file_server_base_path`::

    {base}/{application}/scripts_original
    {base}/{application}/scripts_converted
    {base}/{application}/scripts_converted/validation
    {base}/{application}/scripts_converted/{model}/{timestamp}   <- upload

They are functions rather than f-strings at the call sites so the layout is
stated once: a deployment that renames a folder changes it here and nowhere
else. Every one goes through
:meth:`~app_config.sharepoint.SharePointConfig.drive_path`, which owns the
joining (and the ``Shared Documents/`` stripping the base needs).

Logger name: ``conversion.paths``.
"""

from __future__ import annotations

import logging

from app_config.sharepoint import SharePointConfig

logger = logging.getLogger(__name__)

ORIGINAL_FOLDER = "scripts_original"
CONVERTED_FOLDER = "scripts_converted"
VALIDATION_FOLDER = "validation"


def _config(config: SharePointConfig | None) -> SharePointConfig:
    return config if config is not None else SharePointConfig.from_env()


def original_scripts(
    application: str, *, config: SharePointConfig | None = None
) -> str:
    """The folder holding *application*'s untranslated SAS sources."""
    return _config(config).drive_path(application, ORIGINAL_FOLDER)


def converted_scripts(
    application: str, *, config: SharePointConfig | None = None
) -> str:
    """The root of *application*'s converted output."""
    return _config(config).drive_path(application, CONVERTED_FOLDER)


def validation(application: str, *, config: SharePointConfig | None = None) -> str:
    """Where *application*'s validation artefacts go — beside the converted
    scripts, not inside a run folder, so a reviewer finds the latest verdicts
    in one place."""
    return _config(config).drive_path(
        application, CONVERTED_FOLDER, VALIDATION_FOLDER
    )


def upload_target(
    application: str,
    model: str,
    timestamp: str,
    *,
    config: SharePointConfig | None = None,
) -> str:
    """
    Where one run's converted scripts land:
    ``{converted}/{model}/{timestamp}``.

    Both segments are part of the identity of the output, not decoration —
    the model because two models translate the same source differently, and
    the timestamp because a re-run must never overwrite what came before.
    """
    return _config(config).drive_path(
        application, CONVERTED_FOLDER, model, timestamp
    )
