"""Registry of navigation pages, filled in by ``ui.app`` so views can ``st.switch_page`` to each other."""

from __future__ import annotations

PAGES: dict[str, object] = {}
