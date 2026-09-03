FROM python:3.13-slim

LABEL org.opencontainers.image.title="BioSearch AI" \
    org.opencontainers.image.description="Reproducible biomedical literature search test environment" \
    org.opencontainers.image.source="https://github.com/Namra-5/BioSearch" \
    org.opencontainers.image.documentation="https://github.com/Namra-5/BioSearch#archival--computational-reproducibility" \
    org.opencontainers.image.url="https://doi.org/10.5281/zenodo.22078606"

WORKDIR /app

COPY requirements-lock.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-lock.txt

RUN useradd --create-home --shell /usr/sbin/nologin appuser

COPY --chown=appuser:appuser . .

USER appuser

CMD ["python", "-m", "pytest"]
