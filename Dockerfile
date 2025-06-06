FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.yaml .

# Create conda environment
RUN conda env create -f requirements.yaml

# Make conda environment available in shell
SHELL ["conda", "run", "-n", "hifinn_env", "/bin/bash", "-c"]

# Install ESM (required for FASTA sequences)
RUN git clone https://github.com/facebookresearch/esm.git && \
    cd esm && \
    pip install . && \
    cd ..

# Install unzip and download model data
RUN apt-get update && apt-get install -y unzip && \
    wget https://zenodo.org/records/15013616/files/ModelData.zip && \
    unzip ModelData.zip && \
    rm ModelData.zip

# Copy source code
COPY . .

# Create hifinn_embeddings directory (required by annotate.py)
RUN mkdir -p hifinn_embeddings

# Set default command to run annotation
CMD ["conda", "run", "-n", "hifinn_env", "python", "annotate.py"]