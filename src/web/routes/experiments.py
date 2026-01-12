"""
Experiment API endpoints.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from web.db.session import get_db
from web.db.models import Experiment
from web.schemas import ExperimentResponse
from web.routes.deps import get_experiment_or_404

router = APIRouter()


@router.get("/", response_model=List[ExperimentResponse])
async def list_all_experiments(db: AsyncSession = Depends(get_db)):
	"""List all experiments."""
	result = await db.execute(select(Experiment).order_by(Experiment.created_at.desc()))
	experiments = result.scalars().all()

	return experiments


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment: Experiment = Depends(get_experiment_or_404)):
	"""Get experiment by ID."""
	return experiment


@router.get("/task/{task_id}", response_model=List[ExperimentResponse])
async def list_task_experiments(task_id: int, db: AsyncSession = Depends(get_db)):
	"""List all experiments for a task."""
	result = await db.execute(
		select(Experiment).where(Experiment.task_id == task_id).order_by(Experiment.created_at.desc())
	)
	experiments = result.scalars().all()

	return experiments
