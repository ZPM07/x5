FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl build-essential git && apt-get clean

RUN curl -Ls https://astral.sh/uv/install.sh | bash
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY . .

RUN uv sync --frozen

RUN uv pip install -e .

CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]