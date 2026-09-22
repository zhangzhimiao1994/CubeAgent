"""Allow distinct artifact lineage with duplicate content hashes.

Revision ID: 0020_run_artifact_lineage
Revises: 0019_run_actor_role
Create Date: 2026-09-20
"""

from collections.abc import Sequence

from alembic import op

revision = "0020_run_artifact_lineage"
down_revision = "0019_run_actor_role"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "uq_agent_hub_run_artifacts_run_hash",
        "agent_hub_run_artifacts",
        type_="unique",
    )


def downgrade() -> None:
    op.create_unique_constraint(
        "uq_agent_hub_run_artifacts_run_hash",
        "agent_hub_run_artifacts",
        ["run_id", "content_sha256"],
    )


__all__: Sequence[str] = ("downgrade", "upgrade")
