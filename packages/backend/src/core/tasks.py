import logging

from redis import Redis
from celery import Task
from celery.exceptions import Retry

from .exceptions import LockException

logger = logging.getLogger(__name__)


class DeletingTask(Task):
    redis: Redis
    locked_task_expiration: int
    idempotency_key: str = "delete_task_idempotency_key"
    countdown: int
    max_retries: int

    def before_start(self, task_id, args, kwargs) -> None:
        status = self.redis.set(
            self.idempotency_key,
            'lock',
            ex=self.locked_task_expiration,
            nx=True
        )
        if not status:
            logger.error(
                f"Deleting with idempotency key {self.idempotency_key!r} "
                f"has already locked by another task"
            )
            raise LockException()

    def on_success(self, retval, task_id, args, kwargs) -> None:
        self.redis.delete(self.idempotency_key)

    def on_failure(self, exc, task_id, args, kwargs, einfo) -> None:
        if not isinstance(exc, (Retry, LockException)):
            self.redis.delete(self.idempotency_key)
        if isinstance(exc, LockException):
            self.retry(
                args=args,
                kwargs=kwargs,
                countdown=self.countdown,
                max_retries=self.max_retries
            )
