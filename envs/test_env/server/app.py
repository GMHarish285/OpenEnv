# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the RagOptimizer Environment.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from models import RagAction, RagObservation
    from .rag_environment import RagEnvironment
except ModuleNotFoundError:
    from models import RagAction, RagObservation
    from server.rag_environment import RagEnvironment


# Create the app with web interface and README integration
app = create_app(
    RagEnvironment,
    RagAction,
    RagObservation,
    env_name="test_env", # Keeping test_env as requested by user
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
