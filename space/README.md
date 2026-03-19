---
title: Grabette Data Pipeline
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Grabette Data Pipeline

Run the full GRABETTE data pipeline in your browser:

1. **Create Map** — upload a raw episode (video + IMU), run two-pass ORB-SLAM3 to produce a camera trajectory and map
2. **Batch SLAM** — localize multiple episodes against an existing map
3. **Generate Dataset** — convert SLAM outputs into a [LeRobot v3](https://huggingface.co/docs/lerobot) dataset
