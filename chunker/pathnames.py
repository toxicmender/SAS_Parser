"""What a pathname says about itself, read as SAS on UNIX reads it.

Whether a place a statement names is one known place before the job runs can
mostly be read off the path alone:

- a **wildcard** — ``*``, ``?`` or a ``[ ]`` set — names every file it
  matches: a set of files, not one (SAS 9.4 Companion for UNIX Environments,
  Using Wildcards in Pathnames);
- a part the **SAS server** fills in when it reads the path — ``~/``,
  ``~name/``, ``!SASROOT``, ``$VARIABLE`` (Character Substitutions in
  Pathnames) — names a place only that server knows;
- a **relative** path is read from SAS's current directory.

:class:`~chunker.models.SasPathRef` classifies its place with these
(:attr:`~chunker.models.SasPathRef.wildcard`,
:attr:`~chunker.models.SasPathRef.unresolved`), which is how the chunk
metadata keeps wildcard and unresolved paths apart from the concrete ones.

Pure functions of a string, with no package imports, so :mod:`chunker.models`
can use them. The hydration planner keeps its own copy of the same rules —
:mod:`data_hydration` never imports the chunker — and
``tests/test_pathnames.py`` holds the two to one answer.
"""

from __future__ import annotations

import re

# A UNIX wildcard: `*` one or more characters, `?` one, `[ ]` one of a set.
_WILDCARD_RE = re.compile(r"[*?]|\[[^\]]*\]")

# An environment variable in a UNIX path: $HOME, $PROJ.
_ENV_VAR_RE = re.compile(r"\$([A-Za-z_]\w*)")


def has_wildcard(path: str) -> bool:
    """Whether *path* holds a UNIX wildcard: ``*``, ``?`` or a ``[ ]`` set.

    "You can use the ``*``, ``?``, and ``[ ]`` wildcards to specify pathnames
    in the FILENAME (only if the fileref is to be used for input), INFILE, and
    %INCLUDE statements": ``filename wild 'wild*.dat';`` "reads input from
    every file in the current directory that begins with the string wild and
    ends with .dat" (SAS 9.4 Companion for UNIX Environments, Using Wildcards
    in Pathnames). For input only; which statements read is the caller's to
    say."""
    return _WILDCARD_RE.search(path) is not None


def is_absolute(path: str) -> bool:
    """Whether *path* is a full path: from the root (``/data``), or, on a
    Windows server, a drive (``D:\\data``) or a share (``\\\\host\\share``)."""
    return path.startswith(("/", "\\\\")) or (
        len(path) > 2 and path[1] == ":" and path[2] in "/\\"
    )


def server_parts(path: str) -> list[str]:
    """What in *path* the SAS server fills in when it reads it, each said in
    words: on UNIX, ``~/`` is ``$HOME/``, ``~name/`` that user's home
    directory, ``!sasroot`` the directory SAS is installed in — each only at
    the start — and ``$VARIABLE`` an environment variable (SAS 9.4 Companion
    for UNIX Environments, Character Substitutions in Pathnames)."""
    parts: list[str] = []
    if path.startswith("~"):
        user = path[1:].replace("\\", "/").split("/", 1)[0]
        parts.append(
            f"~{user} is {user}'s home directory"
            if user
            else "~ is the home directory of the user SAS runs as"
        )
    elif path.lower().startswith("!sasroot"):
        parts.append("!SASROOT is the directory SAS is installed in")
    parts += [
        f"${name} is an environment variable of the SAS server"
        for name in _ENV_VAR_RE.findall(path)
    ]
    return parts


def is_relative(path: str) -> bool:
    """Whether *path* is relative: SAS reads it from its current directory.
    One the server completes (:func:`server_parts`) is not taken for one. A
    ``&`` is taken as written: whether it is a macro reference is the
    statement's to say."""
    return bool(path) and not is_absolute(path) and not server_parts(path)
