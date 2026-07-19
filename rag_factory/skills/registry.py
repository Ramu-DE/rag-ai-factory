# -*- coding: utf-8 -*-
"""
Skill Registry
==============
Maps doc_type → skill class.
Central import point for the IDP pipeline.

Usage:
    from rag_factory.skills.registry import SKILL_REGISTRY, get_skill
    skill  = get_skill("invoice")
    result = skill.run(extracted_doc)
"""
from __future__ import annotations
from typing import Dict, Type, Optional
from .base      import BaseDocumentSkill
from .invoice   import InvoiceSkill
from .contract  import ContractSkill
from .medical   import MedicalSkill
from .idcard    import IDCardSkill
from .custom    import CustomSkill

_REGISTRY: Dict[str, Type[BaseDocumentSkill]] = {
    "invoice":     InvoiceSkill,
    "contract":    ContractSkill,
    "medical":     MedicalSkill,
    "id_document": IDCardSkill,
    "other":       None,
    "report":      None,
    "form":        None,
    "letter":      None,
}


def get_skill(doc_type: str) -> Optional[BaseDocumentSkill]:
    """Return an instantiated skill for the given doc_type, or None."""
    cls = _REGISTRY.get(doc_type)
    if cls is None:
        return None
    return cls()


def list_skills() -> Dict[str, str]:
    """Return dict of doc_type → description for all registered skills."""
    result = {}
    for dt, cls in _REGISTRY.items():
        if cls is not None:
            result[dt] = cls.description
        else:
            result[dt] = "No skill registered — raw text extraction only"
    return result


SKILL_REGISTRY = _REGISTRY
