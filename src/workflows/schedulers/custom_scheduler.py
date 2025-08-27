"""Custom scheduler for data processing jobs."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
import threading
import time


class JobStatus(Enum):
    """Job execution status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledJob:
    """Represents a scheduled job."""
    job_id: str
    name: str
    function: Callable
    schedule: str  # Cron-like schedule or interval
    status: JobStatus = JobStatus.PENDING
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300  # 5 minutes default
    metadata: Dict = field(default_factory=dict)


class CustomScheduler:
    """Custom job scheduler implementation."""
    
    def __init__(self):
        self.jobs: Dict[str, ScheduledJob] = {}
        self.running = False
        self.scheduler_thread = None
        self.logger = logging.getLogger(__name__)
    
    def add_job(
        self,
        job_id: str,
        name: str,
        function: Callable,
        schedule: str,
        max_retries: int = 3,
        timeout: int = 300,
        **metadata
    ) -> None:
        """Add a job to the scheduler."""
        job = ScheduledJob(
            job_id=job_id,
            name=name,
            function=function,
            schedule=schedule,
            max_retries=max_retries,
            timeout=timeout,
            metadata=metadata
        )
        
        job.next_run = self._calculate_next_run(schedule)
        self.jobs[job_id] = job
        self.logger.info(f"Added job: {name} ({job_id})")
    
    def remove_job(self, job_id: str) -> bool:
        """Remove a job from the scheduler."""
        if job_id in self.jobs:
            del self.jobs[job_id]
            self.logger.info(f"Removed job: {job_id}")
            return True
        return False
    
    def start(self) -> None:
        """Start the scheduler."""
        if self.running:
            self.logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        self.logger.info("Scheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.logger.info("Scheduler stopped")
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get the status of a specific job."""
        job = self.jobs.get(job_id)
        if job:
            return {
                'job_id': job.job_id,
                'name': job.name,
                'status': job.status.value,
                'last_run': job.last_run,
                'next_run': job.next_run,
                'retry_count': job.retry_count
            }
        return None
    
    def list_jobs(self) -> List[Dict]:
        """List all jobs and their statuses."""
        return [self.get_job_status(job_id) for job_id in self.jobs.keys()]
    
    def _run_scheduler(self) -> None:
        """Main scheduler loop."""
        while self.running:
            try:
                current_time = datetime.now()
                
                for job in self.jobs.values():
                    if (job.next_run and 
                        current_time >= job.next_run and 
                        job.status != JobStatus.RUNNING):
                        
                        self._execute_job(job)
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _execute_job(self, job: ScheduledJob) -> None:
        """Execute a scheduled job."""
        job.status = JobStatus.RUNNING
        job.last_run = datetime.now()
        
        self.logger.info(f"Executing job: {job.name}")
        
        try:
            # Execute the job function
            result = job.function()
            
            job.status = JobStatus.SUCCESS
            job.retry_count = 0
            job.next_run = self._calculate_next_run(job.schedule)
            
            self.logger.info(f"Job completed successfully: {job.name}")
            
        except Exception as e:
            job.retry_count += 1
            
            if job.retry_count <= job.max_retries:
                job.status = JobStatus.PENDING
                job.next_run = datetime.now() + timedelta(minutes=5)  # Retry in 5 minutes
                self.logger.warning(f"Job failed, will retry: {job.name} - {e}")
            else:
                job.status = JobStatus.FAILED
                job.next_run = self._calculate_next_run(job.schedule)
                self.logger.error(f"Job failed after {job.max_retries} retries: {job.name} - {e}")
    
    def _calculate_next_run(self, schedule: str) -> datetime:
        """Calculate the next run time based on schedule."""
        # Simple implementation - extend for cron-like scheduling
        current_time = datetime.now()
        
        if schedule.startswith('interval:'):
            # Format: interval:minutes
            minutes = int(schedule.split(':')[1])
            return current_time + timedelta(minutes=minutes)
        elif schedule == 'daily':
            return current_time + timedelta(days=1)
        elif schedule == 'hourly':
            return current_time + timedelta(hours=1)
        else:
            # Default to 1 hour
            return current_time + timedelta(hours=1)


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    def sample_job():
        print(f"Sample job executed at {datetime.now()}")
        return "Job completed"
    
    scheduler = CustomScheduler()
    scheduler.add_job(
        job_id="sample_job_1",
        name="Sample Job",
        function=sample_job,
        schedule="interval:2"  # Run every 2 minutes
    )
    
    scheduler.start()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler.stop()
