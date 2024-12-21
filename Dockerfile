FROM python:3.11-slim-buster AS base


# TODO: DO NOT RUN FILE

# Install required packages in base stage
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Use a scratch image for the final stage to reduce image size
# FROM base AS streamlit

# Copy only Streamlit-related files and dependencies
# WORKDIR /app/streamlit_app
# COPY ./streamlit_app/app.py ./
# COPY ./streamlit_app/data ./
# COPY ./streamlit_app/templates ./
# COPY ./streamlit_app/styles ./
# COPY ./streamlit_app/static ./

# Set the command to run Streamlit
CMD ["streamlit", "run", "app.py"]

# Use a scratch image for the final stage to reduce image size
FROM base AS fastapi

# Copy only FastAPI-related files and dependencies
WORKDIR /app/server
COPY ./server/* ./

# Install additional packages needed by FastAPI
# RUN pip install --no-cache-dir uvicorn[standard]

# Set the command to run FastAPI using Uvicorn
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["fastapi", "run"]

# Use a scratch image for the final stage to reduce image size
FROM base AS final

# Copy only Streamlit-related files and dependencies
WORKDIR /app/streamlit_app
COPY --from=streamlit ./app ./

# Set environment variables
ENV PORT 8501

# Expose the port that Streamlit will run on
EXPOSE 8501

# Start Streamlit server
CMD ["streamlit", "run", "app.py"]