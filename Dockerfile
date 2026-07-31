# 1. Use an official, lightweight Python 3.11 image as the base OS
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements file first
COPY requirements.txt .

# 4. Install the lightweight CPU-only version of PyTorch FIRST
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Then install the rest of your requirements
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY . .

# 6. Expose the port that FastAPI runs on
EXPOSE 8000

# 7. Define the command to start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
