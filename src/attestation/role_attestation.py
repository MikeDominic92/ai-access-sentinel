"""
Role Attestation Module for AI Access Sentinel

This module implements role attestation workflows for validating that role
definitions are accurate and role assignments follow least privilege principles.

Chainguard Relevance: Demonstrates "role-attestation processes" required for
IT Engineer (Identity/IAM) role - a key compliance requirement.
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
import uuid


# =============================================================================
# ENUMS
# =============================================================================

class AttestationStatus(str, Enum):
    """Status of an attestation campaign or item"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ATTESTED = "attested"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REMEDIATION_REQUIRED = "remediation_required"


class RoleType(str, Enum):
    """Classification of role types"""
    STANDARD = "standard"
    PRIVILEGED = "privileged"
    ADMIN = "admin"
    SERVICE_ACCOUNT = "service_account"
    BREAK_GLASS = "break_glass"


class AttestationDecision(str, Enum):
    """Attestation decision options"""
    APPROVE = "approve"          # Role definition is accurate
    MODIFY = "modify"            # Role needs modification
    REVOKE = "revoke"            # Role should be removed
    ESCALATE = "escalate"        # Needs security team review
    DEFER = "defer"              # Cannot decide now


class SoDConflictSeverity(str, Enum):
    """Severity of segregation of duties conflicts"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# MODELS
# =============================================================================

class Permission(BaseModel):
    """Individual permission within a role"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    resource: str
    actions: List[str]
    conditions: Optional[Dict[str, Any]] = None
    is_sensitive: bool = False
    risk_score: float = 0.0


class RoleDefinition(BaseModel):
    """Complete role definition for attestation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    role_type: RoleType

    # Permissions
    permissions: List[Permission] = Field(default_factory=list)

    # Ownership
    owner_id: str
    owner_email: str
    department: str

    # Classification
    is_privileged: bool = False
    requires_justification: bool = False
    max_assignment_duration_days: Optional[int] = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_modified: datetime = Field(default_factory=datetime.utcnow)
    last_attested: Optional[datetime] = None

    # Risk assessment
    risk_score: float = 0.0
    sod_conflicts: List[str] = Field(default_factory=list)


class RoleAssignment(BaseModel):
    """Assignment of a role to a user"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role_id: str
    role_name: str

    # User info
    user_id: str
    user_email: str
    user_name: str
    department: str

    # Assignment details
    assigned_at: datetime
    assigned_by: str
    expires_at: Optional[datetime] = None
    justification: Optional[str] = None

    # Status
    is_active: bool = True
    last_used: Optional[datetime] = None
    usage_count_30d: int = 0


class SoDConflict(BaseModel):
    """Segregation of duties conflict"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    user_email: str

    # Conflicting roles
    role_1_id: str
    role_1_name: str
    role_2_id: str
    role_2_name: str

    # Conflict details
    conflict_type: str
    description: str
    severity: SoDConflictSeverity
    risk_score: float

    # Status
    status: str = "open"  # open, mitigated, accepted, resolved
    mitigation_notes: Optional[str] = None
    accepted_by: Optional[str] = None


class AttestationItem(BaseModel):
    """Individual item requiring attestation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    campaign_id: str

    # What is being attested
    attestation_type: str  # "role_definition", "role_assignment", "sod_conflict"
    target_id: str
    target_name: str

    # Context
    role_definition: Optional[RoleDefinition] = None
    role_assignment: Optional[RoleAssignment] = None
    sod_conflict: Optional[SoDConflict] = None

    # Reviewer
    reviewer_id: str
    reviewer_email: str

    # Status
    status: AttestationStatus = AttestationStatus.PENDING
    decision: Optional[AttestationDecision] = None
    decision_date: Optional[datetime] = None
    decision_comment: Optional[str] = None

    # Risk
    risk_score: float = 0.0


class AttestationCampaign(BaseModel):
    """Role attestation campaign"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str

    # Campaign type
    campaign_type: str  # "role_definition", "role_assignment", "sod_review", "comprehensive"

    # Schedule
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str
    start_date: date
    end_date: date

    # Status
    status: AttestationStatus = AttestationStatus.PENDING

    # Progress
    total_items: int = 0
    attested_items: int = 0
    rejected_items: int = 0
    pending_items: int = 0

    # Compliance
    compliance_frameworks: List[str] = Field(default_factory=list)

    @property
    def completion_percentage(self) -> float:
        if self.total_items == 0:
            return 0.0
        return round((self.attested_items + self.rejected_items) / self.total_items * 100, 2)

    @property
    def is_overdue(self) -> bool:
        return date.today() > self.end_date and self.status == AttestationStatus.IN_PROGRESS


# =============================================================================
# ATTESTATION ENGINE
# =============================================================================

class RoleAttestationEngine:
    """Engine for managing role attestation workflows"""

    def __init__(self):
        self.campaigns: Dict[str, AttestationCampaign] = {}
        self.items: Dict[str, List[AttestationItem]] = {}
        self.roles: Dict[str, RoleDefinition] = {}
        self.assignments: Dict[str, List[RoleAssignment]] = {}
        self.sod_conflicts: List[SoDConflict] = []

    # -------------------------------------------------------------------------
    # ROLE MANAGEMENT
    # -------------------------------------------------------------------------

    def register_role(self, role: RoleDefinition) -> RoleDefinition:
        """Register a role definition for attestation tracking"""
        self.roles[role.id] = role
        self.assignments[role.id] = []
        return role

    def add_role_assignment(
        self,
        role_id: str,
        assignment: RoleAssignment
    ) -> RoleAssignment:
        """Add a role assignment"""
        if role_id not in self.assignments:
            self.assignments[role_id] = []
        self.assignments[role_id].append(assignment)
        return assignment

    # -------------------------------------------------------------------------
    # SOD DETECTION
    # -------------------------------------------------------------------------

    def detect_sod_conflicts(
        self,
        user_id: str,
        user_email: str,
        sod_rules: List[Dict[str, Any]]
    ) -> List[SoDConflict]:
        """
        Detect segregation of duties conflicts for a user

        This implements the "segregation-of-duties assessments" requirement.
        """
        user_roles = []
        for role_id, assignments in self.assignments.items():
            for assignment in assignments:
                if assignment.user_id == user_id and assignment.is_active:
                    user_roles.append(self.roles[role_id])

        conflicts = []

        for rule in sod_rules:
            role_1_pattern = rule.get("role_1")
            role_2_pattern = rule.get("role_2")
            conflict_type = rule.get("conflict_type")
            severity = rule.get("severity", "high")

            # Check if user has both conflicting roles
            has_role_1 = any(r.name == role_1_pattern for r in user_roles)
            has_role_2 = any(r.name == role_2_pattern for r in user_roles)

            if has_role_1 and has_role_2:
                role_1 = next(r for r in user_roles if r.name == role_1_pattern)
                role_2 = next(r for r in user_roles if r.name == role_2_pattern)

                conflict = SoDConflict(
                    user_id=user_id,
                    user_email=user_email,
                    role_1_id=role_1.id,
                    role_1_name=role_1.name,
                    role_2_id=role_2.id,
                    role_2_name=role_2.name,
                    conflict_type=conflict_type,
                    description=rule.get("description", f"User has conflicting roles: {role_1.name} and {role_2.name}"),
                    severity=SoDConflictSeverity(severity),
                    risk_score=rule.get("risk_score", 80.0)
                )
                conflicts.append(conflict)
                self.sod_conflicts.append(conflict)

        return conflicts

    # -------------------------------------------------------------------------
    # CAMPAIGN MANAGEMENT
    # -------------------------------------------------------------------------

    def create_role_definition_campaign(
        self,
        name: str,
        created_by: str,
        start_date: date,
        end_date: date,
        role_types: List[RoleType] = None
    ) -> AttestationCampaign:
        """
        Create a role definition attestation campaign

        Role owners attest that role definitions are accurate and follow
        least privilege principles.
        """
        campaign = AttestationCampaign(
            name=name,
            description="Attest that role definitions are accurate and permissions are appropriate",
            campaign_type="role_definition",
            created_by=created_by,
            start_date=start_date,
            end_date=end_date,
            compliance_frameworks=["SOC2", "ISO27001", "NIST"]
        )

        self.campaigns[campaign.id] = campaign
        self.items[campaign.id] = []

        # Create attestation items for each role
        for role in self.roles.values():
            if role_types and role.role_type not in role_types:
                continue

            item = AttestationItem(
                campaign_id=campaign.id,
                attestation_type="role_definition",
                target_id=role.id,
                target_name=role.name,
                role_definition=role,
                reviewer_id=role.owner_id,
                reviewer_email=role.owner_email,
                risk_score=role.risk_score
            )
            self.items[campaign.id].append(item)

        campaign.total_items = len(self.items[campaign.id])
        campaign.pending_items = campaign.total_items

        return campaign

    def create_role_assignment_campaign(
        self,
        name: str,
        created_by: str,
        start_date: date,
        end_date: date,
        departments: List[str] = None
    ) -> AttestationCampaign:
        """
        Create a role assignment attestation campaign

        Managers attest that role assignments for their direct reports are appropriate.
        """
        campaign = AttestationCampaign(
            name=name,
            description="Attest that role assignments are appropriate for job functions",
            campaign_type="role_assignment",
            created_by=created_by,
            start_date=start_date,
            end_date=end_date,
            compliance_frameworks=["SOC2", "ISO27001"]
        )

        self.campaigns[campaign.id] = campaign
        self.items[campaign.id] = []

        # Create attestation items for each assignment
        for role_id, assignments in self.assignments.items():
            role = self.roles.get(role_id)
            if not role:
                continue

            for assignment in assignments:
                if not assignment.is_active:
                    continue
                if departments and assignment.department not in departments:
                    continue

                item = AttestationItem(
                    campaign_id=campaign.id,
                    attestation_type="role_assignment",
                    target_id=assignment.id,
                    target_name=f"{assignment.user_name} - {role.name}",
                    role_assignment=assignment,
                    reviewer_id=assignment.assigned_by,
                    reviewer_email=assignment.assigned_by,  # Would lookup manager
                    risk_score=role.risk_score
                )
                self.items[campaign.id].append(item)

        campaign.total_items = len(self.items[campaign.id])
        campaign.pending_items = campaign.total_items

        return campaign

    def create_sod_review_campaign(
        self,
        name: str,
        created_by: str,
        start_date: date,
        end_date: date
    ) -> AttestationCampaign:
        """
        Create a SoD conflict review campaign

        Security team reviews and attests to SoD conflict mitigations.
        """
        campaign = AttestationCampaign(
            name=name,
            description="Review and attest to segregation of duties conflict mitigations",
            campaign_type="sod_review",
            created_by=created_by,
            start_date=start_date,
            end_date=end_date,
            compliance_frameworks=["SOC2", "ISO27001", "PCI_DSS"]
        )

        self.campaigns[campaign.id] = campaign
        self.items[campaign.id] = []

        # Create attestation items for each SoD conflict
        for conflict in self.sod_conflicts:
            if conflict.status == "resolved":
                continue

            item = AttestationItem(
                campaign_id=campaign.id,
                attestation_type="sod_conflict",
                target_id=conflict.id,
                target_name=f"SoD: {conflict.role_1_name} / {conflict.role_2_name}",
                sod_conflict=conflict,
                reviewer_id="security-team",
                reviewer_email="security@example.com",
                risk_score=conflict.risk_score
            )
            self.items[campaign.id].append(item)

        campaign.total_items = len(self.items[campaign.id])
        campaign.pending_items = campaign.total_items

        return campaign

    # -------------------------------------------------------------------------
    # ATTESTATION WORKFLOW
    # -------------------------------------------------------------------------

    def start_campaign(self, campaign_id: str) -> AttestationCampaign:
        """Start an attestation campaign"""
        campaign = self.campaigns.get(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        campaign.status = AttestationStatus.IN_PROGRESS
        return campaign

    def record_attestation(
        self,
        campaign_id: str,
        item_id: str,
        decision: AttestationDecision,
        reviewer_id: str,
        comment: str = None
    ) -> AttestationItem:
        """Record an attestation decision"""
        items = self.items.get(campaign_id, [])
        item = next((i for i in items if i.id == item_id), None)

        if not item:
            raise ValueError(f"Item {item_id} not found in campaign {campaign_id}")

        item.decision = decision
        item.decision_date = datetime.utcnow()
        item.decision_comment = comment

        if decision == AttestationDecision.APPROVE:
            item.status = AttestationStatus.ATTESTED
            self.campaigns[campaign_id].attested_items += 1
        elif decision in [AttestationDecision.MODIFY, AttestationDecision.REVOKE]:
            item.status = AttestationStatus.REMEDIATION_REQUIRED
            self.campaigns[campaign_id].rejected_items += 1
        elif decision == AttestationDecision.ESCALATE:
            item.status = AttestationStatus.PENDING
        else:
            item.status = AttestationStatus.PENDING

        self.campaigns[campaign_id].pending_items -= 1

        # Update role last attested date if role definition
        if item.role_definition:
            item.role_definition.last_attested = datetime.utcnow()

        return item

    def complete_campaign(self, campaign_id: str) -> AttestationCampaign:
        """Complete an attestation campaign"""
        campaign = self.campaigns.get(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        campaign.status = AttestationStatus.ATTESTED
        return campaign

    # -------------------------------------------------------------------------
    # REPORTING
    # -------------------------------------------------------------------------

    def get_campaign_report(self, campaign_id: str) -> Dict[str, Any]:
        """Generate a campaign report"""
        campaign = self.campaigns.get(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        items = self.items.get(campaign_id, [])

        return {
            "campaign": campaign.dict(),
            "summary": {
                "total_items": campaign.total_items,
                "attested": campaign.attested_items,
                "rejected": campaign.rejected_items,
                "pending": campaign.pending_items,
                "completion_percentage": campaign.completion_percentage,
                "is_overdue": campaign.is_overdue
            },
            "by_decision": {
                "approved": sum(1 for i in items if i.decision == AttestationDecision.APPROVE),
                "modified": sum(1 for i in items if i.decision == AttestationDecision.MODIFY),
                "revoked": sum(1 for i in items if i.decision == AttestationDecision.REVOKE),
                "escalated": sum(1 for i in items if i.decision == AttestationDecision.ESCALATE),
            },
            "high_risk_items": [
                i.dict() for i in items if i.risk_score >= 75
            ],
            "compliance_frameworks": campaign.compliance_frameworks
        }


# =============================================================================
# SAMPLE DATA GENERATOR
# =============================================================================

def generate_sample_attestation() -> Dict[str, Any]:
    """Generate sample attestation data for demonstration"""

    engine = RoleAttestationEngine()

    # Register sample roles
    admin_role = RoleDefinition(
        name="Administrator",
        description="Full system administrator access",
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
                name="Full Access",
                resource="*",
                actions=["*"],
                is_sensitive=True,
                risk_score=100.0
            )
        ]
    )

    developer_role = RoleDefinition(
        name="Developer",
        description="Standard developer access",
        role_type=RoleType.STANDARD,
        owner_id="eng-manager",
        owner_email="eng-manager@example.com",
        department="Engineering",
        is_privileged=False,
        risk_score=40.0,
        permissions=[
            Permission(
                name="Code Repository",
                resource="github/*",
                actions=["read", "write", "create_pr"],
                risk_score=30.0
            )
        ]
    )

    engine.register_role(admin_role)
    engine.register_role(developer_role)

    # Add assignments
    engine.add_role_assignment(
        admin_role.id,
        RoleAssignment(
            role_id=admin_role.id,
            role_name=admin_role.name,
            user_id="admin-user",
            user_email="admin@example.com",
            user_name="Admin User",
            department="IT",
            assigned_at=datetime.utcnow() - timedelta(days=30),
            assigned_by="cto@example.com",
            justification="Emergency access for production issues"
        )
    )

    # Create campaign
    campaign = engine.create_role_definition_campaign(
        name="Q4 2025 Role Attestation",
        created_by="security@example.com",
        start_date=date(2025, 10, 1),
        end_date=date(2025, 10, 31)
    )

    engine.start_campaign(campaign.id)

    return {
        "campaign": campaign.dict(),
        "items": [i.dict() for i in engine.items[campaign.id]],
        "roles": [r.dict() for r in engine.roles.values()]
    }


if __name__ == "__main__":
    import json
    result = generate_sample_attestation()
    print(json.dumps(result, indent=2, default=str))
