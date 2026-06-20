web: uvicorn backend.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2
worker: celery -A backend.celery_app:celery_app worker --loglevel=info --concurrency=2 -Q training,celery
