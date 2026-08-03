"""License allow-list for the Socbench audit pipeline."""

from __future__ import annotations

ALLOWED_LICENSES = {
    "apache-2.0",
    "apache 2.0",
    "mit",
    "bsd",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "cc0",
    "cc0-1.0",
    "cc-by",
    "cc-by-3.0",
    "cc-by-4.0",
    "mpl",
    "mpl-2.0",
    "bsl-1.0",
    "bsl-1",
    "unlicense",
    "0bsd",
    "zlib",
    "postgresql",
    "ncsa",
    "odc-by",
    "odbl",
    "pddl",
}

DENIED_SUBSTRINGS = (
    "cc-by-nc",
    "noncommercial",
    "non-commercial",
    "commercial",
    "unlicensed",
    "restricted",
    "proprietary",
    "gpl",
    "agpl",
    "lgpl",
    "cc-by-sa",
)


def is_license_allowed(license_text: str | None, source_default_ok: bool = True) -> bool:
    """Return True if a dataset license is acceptable for training use.

    Parameters
    ----------
    license_text:
        Raw license string from a dataset card.
    source_default_ok:
        Treat missing or empty license as acceptable if the source has been
        verified upstream. Set to False to reject missing licenses.
    """
    if not license_text:
        return source_default_ok

    normalized = license_text.lower().strip()

    for denied in DENIED_SUBSTRINGS:
        if denied in normalized:
            return False

    for allowed in ALLOWED_LICENSES:
        if allowed in normalized:
            return True

    # Strict mode: if we cannot positively identify an allowed license, reject it.
    return False
