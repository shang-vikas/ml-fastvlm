# pugsy_ai/pipelines/fastvlm/tmp_media.py
"""
Temporary media downloader/manager.

Usage:
    from .tmp_media import TempMedia
    with TempMedia(uri) as local_path:
        engine.process(local_path)
# On exit the file is removed automatically.
"""

from __future__ import annotations

import os
import tempfile
import shutil
import logging
from typing import Optional
import urllib.parse

logger = logging.getLogger("fastvlm.tmp_media")

# Configurable via env
DEFAULT_MAX_BYTES = int(os.getenv("FASTVLM_MAX_DOWNLOAD_SIZE_BYTES", str(2 * 1024**3)))  # 2 GiB default
DEFAULT_TIMEOUT = float(os.getenv("FASTVLM_DOWNLOAD_TIMEOUT_SECONDS", "300"))  # 5 min default
CHUNK_SIZE = 1024 * 1024  # 1 MiB


class TempMedia:
    """
    Context manager that yields a local filesystem path to the media.

    Supported sources:
      - Local file paths (no download)
      - http:// or https:// URLs (uses requests; streams to tempfile)
      - gs://bucket/path URIs (uses google-cloud-storage if installed)

    The file is removed on __exit__ unless it was already a local path (in that case it is not deleted).
    """

    def __init__(self, uri: str, max_bytes: Optional[int] = None, timeout: Optional[float] = None):
        self.uri = uri
        self.max_bytes = int(max_bytes) if max_bytes is not None else DEFAULT_MAX_BYTES
        self.timeout = float(timeout) if timeout is not None else DEFAULT_TIMEOUT
        self._tmp_path: Optional[str] = None
        self._is_local = False

    def __enter__(self) -> str:
        if not self.uri:
            raise ValueError("Empty media URI provided")

        # Local path
        if os.path.exists(self.uri):
            self._is_local = True
            return os.path.abspath(self.uri)

        parsed = urllib.parse.urlparse(self.uri)
        scheme = parsed.scheme.lower()

        if scheme in ("http", "https"):
            self._tmp_path = self._download_http(self.uri)
            return self._tmp_path
        elif scheme == "gs":
            try:
                self._tmp_path = self._download_gs(self.uri)
                return self._tmp_path
            except ImportError as exc:
                raise RuntimeError("google-cloud-storage is required to download gs:// URIs. "
                                   "Install with `pip install google-cloud-storage`") from exc
        else:
            # Unknown scheme: try to treat as local path (missing) -> error
            raise ValueError(f"Unsupported media URI scheme or file does not exist: {self.uri}")

    def __exit__(self, exc_type, exc, tb):
        # remove the temp file if we created one
        if self._tmp_path and not self._is_local:
            try:
                os.remove(self._tmp_path)
                logger.debug("Removed temporary media file %s", self._tmp_path)
            except Exception:
                logger.exception("Failed to remove temporary media file %s", self._tmp_path)
        # do not suppress exceptions
        return False

    def _download_http(self, url: str) -> str:
        try:
            import requests
        except Exception as e:
            raise RuntimeError("requests is required to download HTTP(S) URIs. Install with `pip install requests`") from e

        logger.info("Downloading HTTP(S) media: %s", url)
        tmp = tempfile.NamedTemporaryFile(prefix="fastvlm_", suffix=self._infer_suffix(url), delete=False)
        tmp_path = tmp.name
        bytes_written = 0

        try:
            with requests.get(url, stream=True, timeout=(10, self.timeout)) as r:
                r.raise_for_status()
                content_length = r.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        cl = int(content_length)
                        if cl > self.max_bytes:
                            raise ValueError(f"Remote resource size {cl} > allowed max {self.max_bytes}")
                    except ValueError:
                        # ignore if invalid header
                        pass

                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        bytes_written += len(chunk)
                        if bytes_written > self.max_bytes:
                            raise ValueError(f"Downloaded size exceeded maximum of {self.max_bytes} bytes")
                        tmp.write(chunk)
            tmp.flush()
            tmp.close()
            logger.debug("Downloaded %d bytes to %s", bytes_written, tmp_path)
            return tmp_path
        except Exception:
            # cleanup on failure
            try:
                tmp.close()
            except Exception:
                pass
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise

    def _download_gs(self, gs_uri: str) -> str:
        # lazy import to avoid hard dependency if you never use gs://
        from google.cloud import storage

        logger.info("Downloading GCS media: %s", gs_uri)
        parsed = urllib.parse.urlparse(gs_uri)
        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip("/")

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        tmp = tempfile.NamedTemporaryFile(prefix="fastvlm_gs_", suffix=self._infer_suffix(blob_name), delete=False)
        tmp_path = tmp.name
        tmp.close()

        # stream download to file, using chunks to avoid memory explosion
        with open(tmp_path, "wb") as fh:
            # we can use blob.download_to_file with chunking controlled by client library
            # but we'll use a streaming download via `download_as_bytes` chunks if available.
            # Use client library's built-in chunked download:
            # set chunk size on blob for resumable download
            # Note: chunk_size must be a multiple of 256 KiB; we use 1 MiB here.
            try:
                blob.chunk_size = CHUNK_SIZE
                downloader = storage.blob._ChunkedDownload(  # type: ignore[attr-defined]
                    client._http,  # type: ignore[attr-defined]
                    blob._get_download_url(),  # type: ignore[attr-defined]
                    fh,
                )
                # Use public API if present (some storage library versions differ). Fallback:
                try:
                    blob.download_to_file(fh, start=None, end=None, raw_download=False)
                except TypeError:
                    # fallback to download_as_bytes in chunks (inefficient)
                    data = blob.download_as_bytes()
                    if len(data) > self.max_bytes:
                        raise ValueError(f"Blob size {len(data)} exceeds max {self.max_bytes}")
                    fh.write(data)
            except Exception:
                # As a conservative fallback, try download_to_filename (may buffer)
                try:
                    blob.download_to_filename(tmp_path)
                except Exception:
                    # cleanup and bubble up
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    raise
        # Validate size
        final_size = os.path.getsize(tmp_path)
        if final_size > self.max_bytes:
            os.remove(tmp_path)
            raise ValueError(f"Downloaded GCS object too large: {final_size} > {self.max_bytes}")
        logger.debug("Downloaded GCS object to %s (%d bytes)", tmp_path, final_size)
        return tmp_path

    @staticmethod
    def _infer_suffix(name: str) -> str:
        # try to infer extension from URL or blob name
        _, ext = os.path.splitext(urllib.parse.urlparse(name).path)
        return ext or ""
