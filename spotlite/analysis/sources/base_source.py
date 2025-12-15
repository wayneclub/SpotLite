# spotlite/analysis/sources/base_source.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseSourceAdapter(ABC):
    """
    不同 source (Google/Yelp/Tripadvisor) 共用介面：
      - to_unified_schema(raw)
    """

    @abstractmethod
    def to_unified_schema(self, raw_obj: Any) -> Dict[str, Any]:
        ...
