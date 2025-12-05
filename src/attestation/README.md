# Role Attestation Module

> **Chainguard Relevance:** This module demonstrates expertise in "role-attestation processes" and "segregation-of-duties assessments" - key requirements for IT Engineer (Identity/IAM) roles.

## Overview

The Role Attestation module provides comprehensive workflows for validating role definitions, role assignments, and detecting segregation of duties (SoD) conflicts. This supports compliance requirements for SOC 2, ISO 27001, NIST, and PCI DSS frameworks.

## Features

### Core Capabilities

- **Role Definition Attestation** - Role owners attest that role definitions are accurate and follow least privilege principles
- **Role Assignment Attestation** - Managers attest that role assignments are appropriate for job functions
- **SoD Conflict Detection** - Automatic detection of segregation of duties violations
- **SoD Review Campaigns** - Security team review and attestation of SoD conflict mitigations
- **Compliance Reporting** - Generate audit evidence for compliance frameworks

## Quick Start

### Basic Usage

```python
from src.attestation import (
    RoleAttestationEngine,
    RoleDefinition,
    RoleType,
    Permission
)
from datetime import date, timedelta

# Initialize engine
engine = RoleAttestationEngine()

# Register a role
admin_role = RoleDefinition(
    name="Cloud Administrator",
    description="Full administrative access to cloud resources",
    role_type=RoleType.ADMIN,
    owner_id="security-team",
    owner_email="security@example.com",
    department="IT",
    is_privileged=True,
    requires_justification=True,
    max_assignment_duration_days=1,
    risk_score=95.0,
    permissions=[
        Permission(
            name="Full Cloud Access",
            resource="aws:*",
            actions=["*"],
            is_sensitive=True,
            risk_score=100.0
        )
    ]
)
engine.register_role(admin_role)

# Create attestation campaign
campaign = engine.create_role_definition_campaign(
    name="Q4 2025 Role Attestation",
    created_by="security@example.com",
    start_date=date.today(),
    end_date=date.today() + timedelta(days=30)
)

# Start campaign
engine.start_campaign(campaign.id)
```

### SoD Conflict Detection

```python
from src.attestation import RoleAttestationEngine, SoDConflictSeverity

engine = RoleAttestationEngine()

# Define SoD rules
sod_rules = [
    {
        "role_1": "Accounts Payable",
        "role_2": "Accounts Receivable",
        "conflict_type": "financial_fraud",
        "description": "User cannot have both AP and AR access",
        "severity": "critical",
        "risk_score": 95.0
    },
    {
        "role_1": "Developer",
        "role_2": "Production Deploy",
        "conflict_type": "change_management",
        "description": "Developer cannot deploy own code to production",
        "severity": "high",
        "risk_score": 80.0
    },
    {
        "role_1": "User Admin",
        "role_2": "Security Admin",
        "conflict_type": "privilege_escalation",
        "description": "Single user cannot manage both users and security",
        "severity": "high",
        "risk_score": 85.0
    }
]

# Detect conflicts for a user
conflicts = engine.detect_sod_conflicts(
    user_id="user-123",
    user_email="user@example.com",
    sod_rules=sod_rules
)

for conflict in conflicts:
    print(f"SoD Conflict: {conflict.role_1_name} / {conflict.role_2_name}")
    print(f"Severity: {conflict.severity}")
    print(f"Risk Score: {conflict.risk_score}")
```

### Recording Attestation Decisions

```python
from src.attestation import AttestationDecision

# Record attestation
engine.record_attestation(
    campaign_id=campaign.id,
    item_id=item_id,
    decision=AttestationDecision.APPROVE,
    reviewer_id="role-owner@example.com",
    comment="Role definition verified, permissions are appropriate for function"
)

# For modifications needed
engine.record_attestation(
    campaign_id=campaign.id,
    item_id=item_id,
    decision=AttestationDecision.MODIFY,
    reviewer_id="role-owner@example.com",
    comment="Remove write access to production bucket - violates least privilege"
)
```

## Campaign Types

### 1. Role Definition Attestation

Role owners verify that role definitions are accurate:

```python
campaign = engine.create_role_definition_campaign(
    name="Q4 Role Definition Review",
    created_by="security@example.com",
    start_date=date(2025, 10, 1),
    end_date=date(2025, 10, 31),
    role_types=[RoleType.ADMIN, RoleType.PRIVILEGED]  # Focus on high-risk roles
)
```

**Review Criteria:**
- Permissions follow least privilege
- Description accurately reflects access granted
- Risk classification is appropriate
- Ownership is current and accurate

### 2. Role Assignment Attestation

Managers verify role assignments for their teams:

```python
campaign = engine.create_role_assignment_campaign(
    name="Q4 Role Assignment Review",
    created_by="security@example.com",
    start_date=date(2025, 10, 1),
    end_date=date(2025, 10, 31),
    departments=["Engineering", "Finance"]  # Target specific departments
)
```

**Review Criteria:**
- Assignment is appropriate for job function
- Access is still required
- Justification is documented
- Duration limits are appropriate

### 3. SoD Review Campaign

Security team reviews detected SoD conflicts:

```python
campaign = engine.create_sod_review_campaign(
    name="Q4 SoD Conflict Review",
    created_by="security@example.com",
    start_date=date(2025, 10, 1),
    end_date=date(2025, 10, 14)
)
```

**Review Criteria:**
- Conflict is valid (not false positive)
- Mitigation controls are in place
- Compensating controls documented
- Risk acceptance documented if applicable

## Compliance Mapping

### SOC 2 Trust Services Criteria

| Control | Attestation Feature |
|---------|---------------------|
| CC6.1 | Role definition attestation |
| CC6.2 | SoD conflict detection |
| CC6.3 | Role assignment attestation |
| CC6.6 | Campaign completion reporting |

### ISO 27001:2022 Controls

| Control | Attestation Feature |
|---------|---------------------|
| A.5.15 | Access control attestation |
| A.5.16 | Identity management verification |
| A.5.17 | Authentication requirements |
| A.5.18 | Access rights review |
| A.8.2 | Privileged access attestation |

### NIST 800-53 Controls

| Control | Attestation Feature |
|---------|---------------------|
| AC-2 | Account management attestation |
| AC-5 | Separation of duties detection |
| AC-6 | Least privilege attestation |

## Reporting

### Campaign Report

```python
report = engine.get_campaign_report(campaign_id)

print(f"Campaign: {report['campaign']['name']}")
print(f"Completion: {report['summary']['completion_percentage']}%")
print(f"Approved: {report['by_decision']['approved']}")
print(f"Modifications Required: {report['by_decision']['modified']}")
print(f"Revocations: {report['by_decision']['revoked']}")
print(f"High Risk Items: {len(report['high_risk_items'])}")
```

### Audit Evidence Generation

```python
# Export for SOC 2 audit
evidence = {
    "campaign_summary": engine.get_campaign_report(campaign_id),
    "attestation_decisions": [item.dict() for item in engine.items[campaign_id]],
    "sod_conflicts": [c.dict() for c in engine.sod_conflicts],
    "compliance_frameworks": ["SOC2", "ISO27001"]
}
```

## Data Models

### RoleDefinition

```python
{
    "id": "uuid",
    "name": "Cloud Administrator",
    "description": "Full administrative access",
    "role_type": "admin",
    "permissions": [...],
    "owner_id": "security-team",
    "owner_email": "security@example.com",
    "department": "IT",
    "is_privileged": true,
    "requires_justification": true,
    "max_assignment_duration_days": 1,
    "risk_score": 95.0,
    "sod_conflicts": ["Developer"],
    "last_attested": "2025-10-15T10:00:00Z"
}
```

### SoDConflict

```python
{
    "id": "uuid",
    "user_id": "user-123",
    "user_email": "user@example.com",
    "role_1_id": "role-ap",
    "role_1_name": "Accounts Payable",
    "role_2_id": "role-ar",
    "role_2_name": "Accounts Receivable",
    "conflict_type": "financial_fraud",
    "description": "User cannot have both AP and AR access",
    "severity": "critical",
    "risk_score": 95.0,
    "status": "open",
    "mitigation_notes": null,
    "accepted_by": null
}
```

## Best Practices

### Campaign Frequency

| Role Type | Attestation Frequency |
|-----------|----------------------|
| Admin/Privileged | Quarterly |
| Standard | Semi-annually |
| Service Accounts | Annually |
| SoD Conflicts | Upon detection + Quarterly |

### Reviewer Assignment

1. **Role Definition** → Role Owner
2. **Role Assignment** → User's Manager
3. **SoD Conflicts** → Security Team
4. **Privileged Access** → CISO or delegate

### Evidence Retention

- Keep attestation records for 7 years
- Include decision rationale
- Timestamp all decisions
- Maintain chain of custody

## Integration with AI Access Sentinel

This module integrates with the broader AI Access Sentinel platform:

```python
from src.attestation import RoleAttestationEngine
from src.models import RiskAnalyzer

# Use AI risk scoring
analyzer = RiskAnalyzer()
engine = RoleAttestationEngine()

for role in roles:
    risk_score = analyzer.calculate_role_risk(role)
    role.risk_score = risk_score
    engine.register_role(role)
```

---

*This module supports enterprise role attestation aligned with Chainguard IT Engineer (Identity/IAM) requirements for governance and compliance.*

**Author:** Mike Dominic - December 2025
