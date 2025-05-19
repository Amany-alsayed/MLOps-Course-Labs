FROM python


WORKDIR /app

# Copy all files to the container
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run FastAPI with Uvicorn when container starts
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
 

