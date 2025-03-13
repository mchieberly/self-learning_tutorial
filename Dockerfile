e the official PyTorch image
FROM pytorch/pytorch:latest

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code into container
COPY src/ src/
COPY data/ data/

# Set the default command
CMD ["python", "src/train.py"]

