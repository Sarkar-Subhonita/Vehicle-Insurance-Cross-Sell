FROM python:3.10-slim-bookworm

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt pyproject.toml setup.py app.py ./
COPY src/ ./src/
COPY config/ ./config/
COPY templates/ ./templates/
COPY static/ ./static/

RUN python -m pip install --no-cache-dir -r requirements.txt
RUN touch /app/.project-root


EXPOSE 5000
CMD ["python", "app.py"]