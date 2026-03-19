# Build on top of the pre-built ORB-SLAM3 image.
# This gives us the grabette_slam binary and all its C++ dependencies
# without needing Docker-in-Docker.
FROM pollenrobotics/orbslam3-headless

# Install Python 3.12 via deadsnakes PPA (manually, no add-apt-repository needed)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates gnupg curl \
    && curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776" \
       | gpg --dearmor -o /etc/apt/trusted.gpg.d/deadsnakes.gpg \
    && echo "deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" \
       > /etc/apt/sources.list.d/deadsnakes.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the grabette-data package files
# (build context: grabette-data/ repo root)
COPY grabette_data/ ./grabette_data/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY pyproject.toml .

RUN python3.12 -m pip install --no-cache-dir . gradio

# Copy the Gradio app
COPY space/app.py .

# Native SLAM: bypass Docker, call the binary directly
ENV ORBSLAM3_DIR=/ORB_SLAM3
ENV GRABETTE_SLAM_BINARY=/ORB_SLAM3/Examples/Monocular-Inertial/grabette_slam
ENV GRABETTE_SLAM_SETTINGS=/app/config/rpi_bmi088_slam_settings.yaml

# Gradio listens on 7860
EXPOSE 7860

CMD ["python3.12", "app.py"]
