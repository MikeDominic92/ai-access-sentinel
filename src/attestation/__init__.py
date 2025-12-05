"""
Attestation Module for AI Access Sentinel

This module provides role attestation and segregation of duties (SoD) workflows
for compliance with SOC 2, ISO 27001, and NIST frameworks.

Chainguard Relevance: Implements "role-attestation processes" and
"segregation-of-duties assessments" required for IT Engineer (Identity/IAM) role.
"""

from .role_attestation import (
    # Enums
    AttestationStatus,
    RoleType,
    AttestationDecision,
    SoDConflictSeverity,

    # Models
    Permission,
    RoleDefinition,
    RoleAssignment,
    SoDConflict,
    AttestationItem,
    AttestationCampaign,

    # Engine
    RoleAttestationEngine,

    # Utility
    generate_sample_attestation,
)

__all__ = [
    # Enums
    "AttestationStatus",
    "RoleType",
    "AttestationDecision",
    "SoDConflictSeverity",

    # Models
    "Permission",
    "RoleDefinition",
    "RoleAssignment",
    "SoDConflict",
    "AttestationItem",
    "AttestationCampaign",

    # Engine
    "RoleAttestationEngine",

    # Utility
    "generate_sample_attestation",
]
