# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import contextlib
import functools
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import Any, Literal, Optional

import cloudpickle
import requests
from typing_extensions import ParamSpec
from vllm.assets.audio import AudioAsset
from vllm.platforms import current_platform

VLLM_PATH = Path(__file__).parent.parent.parent
"""Path to root of the vLLM repository."""


_P = ParamSpec("_P")


def fork_new_process_for_each_test(func: Callable[_P, None]) -> Callable[_P, None]:
    """Decorator to fork a new process for each test function.
    See https://github.com/vllm-project/vllm/issues/7053 for more details.
    """

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        # Make the process the leader of its own process group
        # to avoid sending SIGTERM to the parent process
        os.setpgrp()
        from _pytest.outcomes import Skipped

        # Create a unique temporary file to store exception info from child
        # process. Use test function name and process ID to avoid collisions.
        with (
            tempfile.NamedTemporaryFile(
                delete=False,
                mode="w+b",
                prefix=f"vllm_test_{func.__name__}_{os.getpid()}_",
                suffix=".exc",
            ) as exc_file,
            ExitStack() as delete_after,
        ):
            exc_file_path = exc_file.name
            delete_after.callback(os.remove, exc_file_path)

            pid = os.fork()
            print(f"Fork a new process to run a test {pid}")
            if pid == 0:
                # Parent process responsible for deleting, don't delete
                # in child.
                delete_after.pop_all()
                try:
                    func(*args, **kwargs)
                except Skipped as e:
                    # convert Skipped to exit code 0
                    print(str(e))
                    os._exit(0)
                except Exception as e:
                    import traceback

                    tb_string = traceback.format_exc()

                    # Try to serialize the exception object first
                    exc_to_serialize: dict[str, Any]
                    try:
                        # First, try to pickle the actual exception with
                        # its traceback.
                        exc_to_serialize = {"pickled_exception": e}
                        # Test if it can be pickled
                        cloudpickle.dumps(exc_to_serialize)
                    except (Exception, KeyboardInterrupt):
                        # Fall back to string-based approach.
                        exc_to_serialize = {
                            "exception_type": type(e).__name__,
                            "exception_msg": str(e),
                            "traceback": tb_string,
                        }
                    try:
                        with open(exc_file_path, "wb") as f:
                            cloudpickle.dump(exc_to_serialize, f)
                    except Exception:
                        # Fallback: just print the traceback.
                        print(tb_string)
                    os._exit(1)
                else:
                    os._exit(0)
            else:
                pgid = os.getpgid(pid)
                _pid, _exitcode = os.waitpid(pid, 0)
                # ignore SIGTERM signal itself
                old_signal_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
                # kill all child processes
                os.killpg(pgid, signal.SIGTERM)
                # restore the signal handler
                signal.signal(signal.SIGTERM, old_signal_handler)
                if _exitcode != 0:
                    # Try to read the exception from the child process
                    exc_info = {}
                    if os.path.exists(exc_file_path):
                        with (
                            contextlib.suppress(Exception),
                            open(exc_file_path, "rb") as f,
                        ):
                            exc_info = cloudpickle.load(f)

                    if (original_exception := exc_info.get("pickled_exception")) is not None:
                        # Re-raise the actual exception object if it was
                        # successfully pickled.
                        assert isinstance(original_exception, Exception)
                        raise original_exception

                    if (original_tb := exc_info.get("traceback")) is not None:
                        # Use string-based traceback for fallback case
                        raise AssertionError(
                            f"Test {func.__name__} failed when called with"
                            f" args {args} and kwargs {kwargs}"
                            f" (exit code: {_exitcode}):\n{original_tb}"
                        ) from None

                    # Fallback to the original generic error
                    raise AssertionError(
                        f"function {func.__name__} failed when called with"
                        f" args {args} and kwargs {kwargs}"
                        f" (exit code: {_exitcode})"
                    ) from None

    return wrapper


def spawn_new_process_for_each_test(f: Callable[_P, None]) -> Callable[_P, None]:
    """Decorator to spawn a new process for each test function."""

    @functools.wraps(f)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        # Check if we're already in a subprocess
        if os.environ.get("RUNNING_IN_SUBPROCESS") == "1":
            # If we are, just run the function directly
            return f(*args, **kwargs)

        import torch.multiprocessing as mp

        with suppress(RuntimeError):
            mp.set_start_method("spawn")

        # Get the module
        module_name = f.__module__

        # Create a process with environment variable set
        env = os.environ.copy()
        env["RUNNING_IN_SUBPROCESS"] = "1"

        with tempfile.TemporaryDirectory() as tempdir:
            output_filepath = os.path.join(tempdir, "new_process.tmp")

            # `cloudpickle` allows pickling complex functions directly
            input_bytes = cloudpickle.dumps((f, output_filepath))

            repo_root = str(VLLM_PATH.resolve())

            env = dict(env or os.environ)
            env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

            cmd = [sys.executable, "-m", f"{module_name}"]

            returned = subprocess.run(cmd, input=input_bytes, capture_output=True, env=env)

            # check if the subprocess is successful
            try:
                returned.check_returncode()
            except Exception as e:
                # wrap raised exception to provide more information
                raise RuntimeError(f"Error raised in subprocess:\n{returned.stderr.decode()}") from e

    return wrapper


def create_new_process_for_each_test(
    method: Literal["spawn", "fork"] | None = None,
) -> Callable[[Callable[_P, None]], Callable[_P, None]]:
    """Creates a decorator that runs each test function in a new process.

    Args:
        method: The process creation method. Can be either "spawn" or "fork".
               If not specified, it defaults to "spawn" on ROCm and XPU
               platforms and "fork" otherwise.

    Returns:
        A decorator to run test functions in separate processes.
    """
    if method is None:
        use_spawn = current_platform.is_rocm() or current_platform.is_xpu()
        method = "spawn" if use_spawn else "fork"

    assert method in ["spawn", "fork"], "Method must be either 'spawn' or 'fork'"

    if method == "fork":
        return fork_new_process_for_each_test

    return spawn_new_process_for_each_test


def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")

    return result


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local file to base64 format."""
    with open(file_path, "rb") as f:
        content = f.read()
        result = base64.b64encode(content).decode("utf-8")
    return result


def get_video_url_from_path(video_path: Optional[str]) -> str:
    """Convert a video path (local file or URL) to a video URL format for the API.

    If video_path is None or empty, returns the default URL.
    If video_path is a local file path, encodes it to base64 data URL.
    If video_path is a URL, returns it as-is.
    """
    if not video_path:
        # Default video URL
        return "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"

    # Check if it's a URL (starts with http:// or https://)
    if video_path.startswith(("http://", "https://")):
        return video_path

    # Otherwise, treat it as a local file path
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Detect video MIME type from file extension
    video_path_lower = video_path.lower()
    if video_path_lower.endswith(".mp4"):
        mime_type = "video/mp4"
    elif video_path_lower.endswith(".webm"):
        mime_type = "video/webm"
    elif video_path_lower.endswith(".mov"):
        mime_type = "video/quicktime"
    elif video_path_lower.endswith(".avi"):
        mime_type = "video/x-msvideo"
    elif video_path_lower.endswith(".mkv"):
        mime_type = "video/x-matroska"
    else:
        # Default to mp4 if extension is unknown
        mime_type = "video/mp4"

    video_base64 = encode_base64_content_from_file(video_path)
    return f"data:{mime_type};base64,{video_base64}"


def get_image_url_from_path(image_path: Optional[str]) -> str:
    """Convert an image path (local file or URL) to an image URL format for the API.

    If image_path is None or empty, returns the default URL.
    If image_path is a local file path, encodes it to base64 data URL.
    If image_path is a URL, returns it as-is.
    """
    if not image_path:
        # Default image URL
        return "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"

    # Check if it's a URL (starts with http:// or https://)
    if image_path.startswith(("http://", "https://")):
        return image_path

    # Otherwise, treat it as a local file path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Detect image MIME type from file extension
    image_path_lower = image_path.lower()
    if image_path_lower.endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    elif image_path_lower.endswith(".png"):
        mime_type = "image/png"
    elif image_path_lower.endswith(".gif"):
        mime_type = "image/gif"
    elif image_path_lower.endswith(".webp"):
        mime_type = "image/webp"
    else:
        # Default to jpeg if extension is unknown
        mime_type = "image/jpeg"

    image_base64 = encode_base64_content_from_file(image_path)
    return f"data:{mime_type};base64,{image_base64}"


def get_audio_url_from_path(audio_path: Optional[str]) -> str:
    """Convert an audio path (local file or URL) to an audio URL format for the API.

    If audio_path is None or empty, returns the default URL.
    If audio_path is a local file path, encodes it to base64 data URL.
    If audio_path is a URL, returns it as-is.
    """
    if not audio_path:
        # Default audio URL
        return AudioAsset("mary_had_lamb").url

    # Check if it's a URL (starts with http:// or https://)
    if audio_path.startswith(("http://", "https://")):
        return audio_path

    # Otherwise, treat it as a local file path
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Detect audio MIME type from file extension
    audio_path_lower = audio_path.lower()
    if audio_path_lower.endswith((".mp3", ".mpeg")):
        mime_type = "audio/mpeg"
    elif audio_path_lower.endswith(".wav"):
        mime_type = "audio/wav"
    elif audio_path_lower.endswith(".ogg"):
        mime_type = "audio/ogg"
    elif audio_path_lower.endswith(".flac"):
        mime_type = "audio/flac"
    elif audio_path_lower.endswith(".m4a"):
        mime_type = "audio/mp4"
    else:
        # Default to wav if extension is unknown
        mime_type = "audio/wav"

    audio_base64 = encode_base64_content_from_file(audio_path)
    return f"data:{mime_type};base64,{audio_base64}"
