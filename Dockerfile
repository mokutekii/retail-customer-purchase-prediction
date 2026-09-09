# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install only the runtime package before copying data so dependency layers can be reused.
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

# The small, versioned UCI source data permits a self-contained training image.
# The much larger optional Online Retail II data is intentionally excluded.
COPY data/online_shoppers_intention.csv ./data/online_shoppers_intention.csv
RUN mkdir -p /app/artifacts

ENTRYPOINT ["python", "-m", "retail_purchase.train"]
CMD ["--fast"]
