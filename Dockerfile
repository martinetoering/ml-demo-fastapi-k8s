FROM python:3.11

WORKDIR /app

# Install poetry version
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install poetry dependencies
COPY ./pyproject.toml /app/pyproject.toml
COPY ./poetry.lock /app/poetry.lock
RUN poetry install --no-root

# Copy code
COPY ./fastapi_app /app
# Create empty README.md (otherwise poetry will give error)
RUN touch README.md

RUN poetry install

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
