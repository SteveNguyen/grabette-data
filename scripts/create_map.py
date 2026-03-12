#!/usr/bin/env python3
"""Create SLAM map from a single mapping video (two-pass)."""

import click
from pathlib import Path

from grabette_data.slam import create_map, DEFAULT_DOCKER_IMAGE, DEFAULT_SETTINGS


@click.command()
@click.option("-i", "--input_dir", required=True, type=click.Path(exists=True),
              help="Directory containing raw_video.mp4 and imu_data.json")
@click.option("--retries", type=int, default=3,
              help="Number of extra pass-1 attempts (total = 1 + retries)")
@click.option("-n", "--parallel", type=int, default=1,
              help="Number of pass-1 attempts to run simultaneously")
@click.option("-d", "--docker_image", default=DEFAULT_DOCKER_IMAGE,
              help="Docker image name")
@click.option("-s", "--settings", default=str(DEFAULT_SETTINGS), type=click.Path(exists=True),
              help="SLAM settings YAML")
@click.option("--deterministic", is_flag=True, default=False,
              help="Run in deterministic mode (slower, reproducible)")
@click.option("--max_lost_pct", type=float, default=-1,
              help="Max lost frame %% before early abort (-1=disabled)")
@click.option("--warmup_frames", type=int, default=300,
              help="Frames before checking lost rate")
def main(input_dir, retries, parallel, docker_image, settings,
         deterministic, max_lost_pct, warmup_frames):
    video_dir = Path(input_dir).expanduser().absolute()
    for fn in ["raw_video.mp4", "imu_data.json"]:
        if not (video_dir / fn).is_file():
            raise click.ClickException(f"Missing {fn} in {video_dir}")

    map_path = create_map(
        video_dir,
        retries=retries,
        parallel=parallel,
        deterministic=deterministic,
        max_lost_pct=max_lost_pct,
        warmup_frames=warmup_frames,
        docker_image=docker_image,
        settings_path=Path(settings),
    )
    print(f"\nMap saved to: {map_path}")


if __name__ == "__main__":
    main()
