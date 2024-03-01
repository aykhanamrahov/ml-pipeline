#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging

import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Download the artifact and logging
    logger.info("Getting artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Creating the dataframe
    logger.info("Creating the dataframe")
    df = pd.read_csv(artifact_local_path)

    # Selecting the data
    logger.info("Dropping missing values")
    idx = df['price'].between(args.min_price, args.max_price)
    # Copying dataframe
    logger.info("Copying dataframe")
    df = df[idx].copy()

    # Selecting the data
    logger.info("Dropping outliers: longitude & latitude")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    # Copying dataframe
    logger.info("Copying dataframe")
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Convert last_review attribute to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"])

    logger.info(f"Saving cleaned data to {args.output_artifact}")
    # save cleaned data
    df.to_csv(args.output_artifact, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    artifact.add_file(args.output_artifact)

    logger.info("Logging artifact")
    run.log_artifact(artifact)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")


    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type for the artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the artifact",
        required=True
    )
    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum for the dataset",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum for the dataset",
        required=True
    )

    args = parser.parse_args()

    go(args)