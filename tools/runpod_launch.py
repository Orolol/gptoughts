#!/usr/bin/env python3
"""Launch a training job on RunPod using the official Python SDK.

The script wraps ``runpod.create_pod`` with a CLI so you can trigger an on-demand
GPU pod, run your training command, and optionally wait until the pod becomes
available. You will need to export ``RUNPOD_API_KEY`` or pass ``--api-key``.

If an install bootstrap script is supplied (defaults to ``install.txt`` when
present), its commands execute inside the pod before the primary training
command so dependencies are set up automatically.
"""

from __future__ import annotations

import argparse
import base64
import json
import posixpath
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import runpod
import textwrap

# RunPod currently exposes these desiredStatus values for pods once deployed.
_TERMINAL_STATUSES = {"TERMINATED", "STOPPED", "CANCELLED", "FAILED"}


def _parse_key_value_pairs(pairs: Iterable[str]) -> Dict[str, str]:
    """Convert a list of ``KEY=VALUE`` strings into a dictionary."""
    env: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Cannot parse environment variable '{pair}'. Use KEY=VALUE format.")
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Environment variable '{pair}' is missing a key before '='.")
        env[key] = value
    return env


def _stringify_command(command: Optional[str]) -> str:
    """Normalize the command string fed into ``dockerArgs``."""
    if not command:
        return ""
    return command.strip()


def _load_install_script(install_path: Optional[str]) -> Optional[str]:
    """Read bootstrap commands that should run before the primary command."""
    if not install_path:
        return None

    try:
        with open(install_path, "r", encoding="utf-8") as handle:
            content = handle.read().strip()
    except OSError as exc:
        print(f"[runpod] Warning: failed to read install script {install_path}: {exc}")
        return None

    if not content:
        print(f"[runpod] Install script {install_path} is empty; skipping bootstrap step.")
        return None

    return content


def _compose_container_command(
    install_script: Optional[str],
    primary_command: str,
    wait_fragment: Optional[str] = None,
) -> str:
    """Combine install bootstrap commands with the primary command for the pod."""
    script_sections: List[str] = []

    if install_script:
        script_sections.append(install_script)

    if wait_fragment:
        script_sections.append(wait_fragment)

    if primary_command:
        script_sections.append(primary_command)

    if not script_sections:
        return ""

    script_body = "\n".join(script_sections)
    encoded = base64.b64encode(script_body.encode("utf-8")).decode("ascii")

    command = (
        "bash -lc 'set -eo pipefail; "
        "mkdir -p /tmp/runpod && "
        "printf %s \"" + encoded + "\" | base64 -d > /tmp/runpod/bootstrap.sh && "
        "chmod +x /tmp/runpod/bootstrap.sh && "
        "/tmp/runpod/bootstrap.sh'"
    )

    return command.replace('"', '\\"')


def _determine_ssh_key_path(user_supplied: Optional[str]) -> Optional[Path]:
    if user_supplied:
        candidate = Path(user_supplied).expanduser()
        if candidate.exists():
            return candidate
        print(f"[runpod] Warning: specified SSH key {candidate} not found. Falling back to defaults.")

    default_dir = Path(getattr(runpod, "SSH_KEY_PATH", os.path.expanduser("~/.runpod/ssh")))
    for filename in ("id_ed25519", "id_rsa"):
        default_key = default_dir / filename
        if default_key.exists():
            return default_key

    home_ssh = Path.home() / ".ssh"
    for filename in ("id_ed25519", "id_rsa"):
        fallback = home_ssh / filename
        if fallback.exists():
            return fallback

    return None


def _format_command(command: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _run_subprocess_with_retries(
    command: List[str],
    *,
    label: str,
    retries: int = 10,
    delay: int = 5,
) -> None:
    """Execute a subprocess command with retry logic."""

    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                command,
                check=False,
                text=True,
                capture_output=True,
            )
            if result.returncode == 0:
                return

            stderr_output = result.stderr.strip()
            stdout_output = result.stdout.strip()
            print(
                f"[runpod] Attempt {attempt}/{retries} for {label}."
                f" Command: {_format_command(command)}"
            )
            if stderr_output:
                print(f"[runpod] {label} stderr:\n{stderr_output}")
            if stdout_output:
                print(f"[runpod] {label} stdout:\n{stdout_output}")

            if attempt == retries:
                raise subprocess.CalledProcessError(
                    result.returncode,
                    command,
                    output=result.stdout,
                    stderr=result.stderr,
                )

            print(
                f"[runpod] Warning: {label} failed (attempt {attempt}/{retries}),"
                f" retrying in {delay}s..."
            )
        except subprocess.CalledProcessError as exc:
            # Should be rare; treat similarly.
            if attempt == retries:
                raise
            print(
                f"[runpod] Warning: {label} raised CalledProcessError (attempt {attempt}/{retries}): {exc}."
                f" Retrying in {delay}s..."
            )
        except KeyboardInterrupt:
            raise
        except FileNotFoundError:
            # For missing binaries, surface immediately.
            raise

        time.sleep(delay)


def _ensure_remote_directory(host: str, port: int, remote_dir: str, ssh_key: Optional[Path]) -> None:
    command = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=6",
        "-p",
        str(port),
    ]

    if ssh_key:
        command.extend(["-i", str(ssh_key)])

    command.extend([f"root@{host}", f"mkdir -p {shlex.quote(remote_dir)}"])

    _run_subprocess_with_retries(command, label="remote directory preparation")


def _wait_for_ssh_ready(
    host: str,
    port: int,
    username: str,
    ssh_key: Optional[Path],
    timeout: Optional[int],
    interval: int = 5,
) -> None:
    """Poll the SSH endpoint with a real handshake until ready or timeout."""

    if timeout is not None:
        deadline = time.monotonic() + timeout
    else:
        deadline = None

    attempt = 0
    base_command = [
        "ssh",
        # "-o",
        # "StrictHostKeyChecking=no",
        # "-o",
        # "UserKnownHostsFile=/dev/null",
        # "-o",
        # "BatchMode=yes",
        # "-o",
        # "ConnectTimeout=10",
        "-p",
        str(port),
    ]

    if ssh_key:
        base_command.extend(["-i", str(ssh_key)])

    target = f"{username}@{host}"

    while True:
        attempt += 1
        command = base_command + [target, "exit"]
        result = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            return

        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for SSH on {target}:{port}."
            )

        stderr_output = result.stderr.strip()
        stdout_output = result.stdout.strip()
        print(
            f"[runpod] SSH ready check attempt {attempt}. Command: {_format_command(command)}"
        )
        if stderr_output:
            print(f"[runpod] SSH wait stderr:\n{stderr_output}")
        if stdout_output:
            print(f"[runpod] SSH wait stdout:\n{stdout_output}")

        print(
            f"[runpod] Waiting for SSH availability on {target}:{port}"
            f" (attempt {attempt}, retry in {interval}s)..."
        )
        time.sleep(max(1, interval))


def _upload_checkpoint(
    local_path: Path,
    host: str,
    port: int,
    remote_path: str,
    ssh_key: Optional[Path],
) -> None:
    command = [
        "scp",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=10",
        "-P",
        str(port),
    ]

    if ssh_key:
        command.extend(["-i", str(ssh_key)])

    command.extend([str(local_path), f"root@{host}:{remote_path}"])

    _run_subprocess_with_retries(command, label="checkpoint upload")


def _wait_for_pod(pod_id: str, poll_interval: int, timeout: Optional[int]) -> dict:
    """Poll RunPod until the pod reports ``RUNNING`` or reaches a terminal state."""
    start_time = time.monotonic()
    last_snapshot = None

    while True:
        try:
            pod_info = runpod.get_pod(pod_id)
        except Exception as exc:  # noqa: BLE001 - surface but keep polling
            if last_snapshot != ("API_ERROR", None):
                last_snapshot = ("API_ERROR", None)
                print(f"[runpod] Warning: get_pod failed ({exc}). Retrying...")
            time.sleep(poll_interval)
            continue

        if not pod_info:
            if last_snapshot != ("PENDING", None):
                last_snapshot = ("PENDING", None)
                print(f"[runpod] Pod {pod_id} not yet visible. Waiting...")
            time.sleep(poll_interval)
            continue

        desired_status = pod_info.get("desiredStatus")
        machine_id = pod_info.get("machineId")

        snapshot = (desired_status, machine_id)
        if snapshot != last_snapshot:
            last_snapshot = snapshot
            runtime_info = pod_info.get("runtime") or {}
            runtime_ports = runtime_info.get("ports") or []
            port_summary = ", ".join(
                f"{port.get('type') or port.get('privatePort')} -> {port.get('publicPort')}" for port in runtime_ports
            )
            print(
                f"[runpod] status={desired_status} machine_id={machine_id or 'pending'}"
                f" ports=[{port_summary}]"
            )

        if desired_status == "RUNNING" and machine_id:
            return pod_info

        if desired_status in _TERMINAL_STATUSES:
            raise RuntimeError(
                f"Pod {pod_id} reached terminal status '{desired_status}'."
            )

        if timeout is not None and (time.monotonic() - start_time) > timeout:
            raise TimeoutError(
                f"Timed out after {timeout} seconds while waiting for pod {pod_id} to become RUNNING."
            )

        time.sleep(poll_interval)


def _wait_for_ssh_port(
    pod_id: str,
    poll_interval: int,
    timeout: Optional[int],
    initial_info: Optional[dict] = None,
):
    start_time = time.monotonic()
    pod_info = initial_info
    notice_printed = False

    while True:
        runtime_ports = []
        if pod_info:
            runtime_ports = ((pod_info.get("runtime") or {}).get("ports") or [])
            ssh_port = next(
                (
                    port
                    for port in runtime_ports
                    if port.get("type") == "ssh" or port.get("privatePort") == 22
                ),
                None,
            )
            if ssh_port and ssh_port.get("publicPort"):
                return pod_info, ssh_port, runtime_ports

        elapsed = time.monotonic() - start_time
        if timeout is not None and elapsed > timeout:
            return pod_info, None, runtime_ports

        if not notice_printed:
            print(f"[runpod] Waiting for SSH port assignment for pod {pod_id}...")
            notice_printed = True

        time.sleep(poll_interval)
        try:
            pod_info = runpod.get_pod(pod_id)
        except Exception as exc:  # noqa: BLE001 - continue polling on transient errors
            print(f"[runpod] Warning: get_pod failed while waiting for SSH port ({exc}). Retrying...")
            pod_info = None


def _format_ports(runtime_ports: List[dict]) -> str:
    if not runtime_ports:
        return "(no exposed ports yet)"

    formatted = []
    for port in runtime_ports:
        ip = port.get("ip") or "pending"
        public_port = port.get("publicPort")
        private_port = port.get("privatePort")
        port_type = port.get("type") or "port"
        formatted.append(f"{port_type}: {ip}:{public_port} -> {private_port}")
    return "\n  - ".join(formatted)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch a RunPod on-demand pod and start a training command.",
    )

    parser.add_argument("--api-key", help="RunPod API key. Defaults to RUNPOD_API_KEY env var.")
    parser.add_argument("--name", required=True, help="Name assigned to the pod.")
    parser.add_argument(
        "--image-name",
        help="Docker image to run (e.g. runpod/pytorch:2.2.0-py3.10). Required unless a template is provided.",
    )
    parser.add_argument(
        "--template-id",
        help="Optional RunPod template ID. Skip --image-name if your template already defines the image/command.",
    )
    parser.add_argument(
        "--command",
        help="Training command executed inside the container (wrap in quotes). Ignored when using a template with its own start command.",
    )
    parser.add_argument(
        "--install-script",
        help="Shell script executed inside the pod before the training command (defaults to install.txt when present).",
    )
    parser.add_argument(
        "--checkpoint-path",
        help="Local checkpoint file to upload to the pod before training starts.",
    )
    parser.add_argument(
        "--checkpoint-dest",
        help="Destination path inside the pod for the checkpoint file (default: /runpod-volume/checkpoints/<filename>).",
    )
    parser.add_argument(
        "--ssh-key",
        help="Private SSH key used for uploading checkpoints (defaults to ~/.runpod/ssh/id_rsa if present).",
    )
    parser.add_argument(
        "--ports",
        action="append",
        help="Expose additional container ports (format: 'HOSTPORT/TYPE', e.g., '8888/http'). Repeatable.",
    )
    parser.add_argument(
        "--gpu-type-id",
        help="GPU type identifier from runpod.get_gpus() (e.g. NVIDIA RTX 4090). Leave empty for CPU pods.",
    )
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs to request (default: 1).")
    parser.add_argument(
        "--cloud-type",
        choices=["ALL", "COMMUNITY", "SECURE"],
        default="ALL",
        help="Target cloud pool (default: ALL).",
    )
    parser.add_argument("--data-center-id", help="Optional data center ID to pin the pod location.")
    parser.add_argument("--country-code", help="ISO country code to bias placement (e.g. US, NL).")
    parser.add_argument("--min-vcpu", type=int, default=8, help="Minimum vCPU count (default: 8).")
    parser.add_argument("--min-memory", type=int, default=30, help="Minimum RAM in GB (default: 30).")
    parser.add_argument("--volume-gb", type=int, default=0, help="Persistent volume size in GB (default: 0).")
    parser.add_argument(
        "--container-disk-gb",
        type=int,
        default=None,
        help="Ephemeral container disk size in GB (default: RunPod default).",
    )
    parser.add_argument(
        "--volume-mount-path",
        default="/runpod-volume",
        help="Mount point for the persistent volume inside the container.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable to inject. Repeat for multiple values.",
    )
    parser.add_argument(
        "--allowed-cuda",
        action="append",
        default=[],
        metavar="CUDA_VERSION",
        help="Restrict placement to specific CUDA versions (repeat flag).",
    )
    parser.add_argument(
        "--network-volume-id",
        help="Attach an existing RunPod network volume by ID.",
    )
    parser.add_argument(
        "--no-public-ip",
        action="store_true",
        help="Disable public IP exposure for the pod (default: enabled).",
    )
    parser.add_argument(
        "--no-ssh",
        action="store_true",
        help="Do not start the SSH service inside the pod (default: start SSH).",
    )
    parser.add_argument("--min-download", type=int, help="Minimum download bandwidth (Mbps).")
    parser.add_argument("--min-upload", type=int, help="Minimum upload bandwidth (Mbps).")
    parser.add_argument("--poll-interval", type=int, default=5, help="Seconds between status checks (default: 15).")
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Maximum seconds to wait for RUNNING status. Use 0 to wait indefinitely.",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Do not wait for the pod to become RUNNING; exit after creation.",
    )

    args = parser.parse_args(argv)

    api_key = args.api_key or os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        parser.error("Set RUNPOD_API_KEY environment variable or pass --api-key.")

    if not args.image_name and not args.template_id:
        parser.error("Provide --image-name or --template-id so RunPod knows which image to launch.")

    if args.image_name and args.template_id:
        parser.error("Use either --image-name or --template-id, not both.")

    if not args.template_id and not args.command:
        parser.error("Provide --command when launching directly from an image.")

    if args.checkpoint_path and args.no_ssh:
        parser.error("Cannot upload a checkpoint when SSH is disabled. Remove --no-ssh to continue.")

    install_path = args.install_script
    if install_path is None:
        default_install = Path("install.txt")
        if default_install.exists():
            install_path = str(default_install)

    checkpoint_local_path: Optional[Path] = None
    checkpoint_wait_fragment: Optional[str] = None
    checkpoint_remote_path: Optional[str] = None
    checkpoint_remote_dir: Optional[str] = None

    port_entries: List[str] = []
    if args.ports:
        for entry in args.ports:
            for piece in entry.split(","):
                cleaned = piece.strip()
                if cleaned:
                    port_entries.append(cleaned)

    if args.checkpoint_path:
        checkpoint_local_path = Path(args.checkpoint_path).expanduser()
        if not checkpoint_local_path.is_file():
            parser.error(f"Checkpoint path {checkpoint_local_path} does not exist or is not a file.")

        if args.checkpoint_dest:
            dest_candidate = args.checkpoint_dest
        else:
            dest_candidate = "/runpod-volume/checkpoints"

        if dest_candidate.endswith("/"):
            checkpoint_remote_dir = dest_candidate.rstrip("/") or "/"
            checkpoint_remote_path = posixpath.join(checkpoint_remote_dir, checkpoint_local_path.name)
        else:
            basename = posixpath.basename(dest_candidate)
            if "." in basename and not dest_candidate.endswith(basename + "/"):
                checkpoint_remote_path = dest_candidate
                checkpoint_remote_dir = posixpath.dirname(dest_candidate) or "."
            else:
                checkpoint_remote_dir = dest_candidate
                checkpoint_remote_path = posixpath.join(checkpoint_remote_dir, checkpoint_local_path.name)

        wait_lines = textwrap.dedent(
            f"""
            mkdir -p {shlex.quote(checkpoint_remote_dir)}
            echo "[bootstrap] Waiting for checkpoint upload at {checkpoint_remote_path}"
            until [ -f {shlex.quote(checkpoint_remote_path)} ]; do
              sleep 5
            done
            echo "[bootstrap] Checkpoint detected at {checkpoint_remote_path}"
            export RUNPOD_RESUME_CHECKPOINT={shlex.quote(checkpoint_remote_path)}
            """
        ).strip()

        checkpoint_wait_fragment = wait_lines

    if not args.no_ssh:
        has_ssh_port = any(part.split("/")[0].strip() == "22" for part in port_entries)
        if not has_ssh_port:
            port_entries.append("22/tcp")

    ports_string = ",".join(port_entries) if port_entries else None

    try:
        env_vars = _parse_key_value_pairs(args.env)
    except ValueError as exc:  # pragma: no cover - exercised by CLI parsing
        parser.error(str(exc))

    install_script_content = _load_install_script(install_path)
    if install_script_content and install_path:
        print(f"[runpod] Will execute install bootstrap script before training: {install_path}")

    ssh_key_path = _determine_ssh_key_path(args.ssh_key) if checkpoint_local_path else None

    docker_args = _compose_container_command(
        install_script_content,
        _stringify_command(args.command),
        checkpoint_wait_fragment,
    )
    allowed_cuda = args.allowed_cuda or None
    timeout = None if args.timeout == 0 else args.timeout

    runpod.api_key = api_key

    if ports_string:
        print(f"[runpod] Requesting port mappings: {ports_string}")

    print("[runpod] Creating pod with requested configuration...")
    pod = runpod.create_pod(
        name=args.name,
        image_name=args.image_name or "",
        gpu_type_id=args.gpu_type_id,
        cloud_type=args.cloud_type,
        support_public_ip=not args.no_public_ip,
        start_ssh=True,
        data_center_id=args.data_center_id,
        country_code=args.country_code,
        gpu_count=args.gpu_count,
        volume_in_gb=args.volume_gb,
        container_disk_in_gb=args.container_disk_gb,
        min_vcpu_count=args.min_vcpu,
        min_memory_in_gb=args.min_memory,
        docker_args=docker_args,
        volume_mount_path=args.volume_mount_path,
        ports=ports_string,
        env=env_vars or None,
        template_id=args.template_id,
        network_volume_id=args.network_volume_id,
        allowed_cuda_versions=allowed_cuda,
        min_download=args.min_download,
        min_upload=args.min_upload,
    )

    pod_id = pod["id"]
    print(json.dumps(pod, indent=2))
    print(f"[runpod] Pod {pod_id} requested. Waiting for provisioning..." )

    if args.detach:
        print("[runpod] Detach requested; exiting without waiting for RUNNING status.")
        return 0

    try:
        pod_info = _wait_for_pod(pod_id, args.poll_interval, timeout)
    except (RuntimeError, TimeoutError) as exc:
        print(f"[runpod] ERROR: {exc}", file=sys.stderr)
        return 1

    runtime_info = pod_info.get("runtime") or {}
    runtime_ports = runtime_info.get("ports") or []

    ssh_port_entry = next(
        (
            port
            for port in runtime_ports
            if port.get("type") == "ssh" or port.get("privatePort") == 22
        ),
        None,
    )

    if checkpoint_local_path:
        if not ssh_port_entry or not ssh_port_entry.get("publicPort"):
            ssh_wait_timeout = timeout if timeout is not None else 10
            pod_info, ssh_port_entry, runtime_ports = _wait_for_ssh_port(
                pod_id,
                args.poll_interval,
                ssh_wait_timeout,
                initial_info=pod_info,
            )
            runtime_info = pod_info.get("runtime") or {}
            runtime_ports = runtime_info.get("ports") or []
            ssh_port_entry = next(
                (
                    port
                    for port in runtime_ports
                    if port.get("type") == "ssh" or port.get("privatePort") == 22
                ),
                None,
            )

        if not ssh_port_entry or not ssh_port_entry.get("publicPort"):
            print("[runpod] ERROR: SSH port info not available; cannot upload checkpoint.", file=sys.stderr)
            return 1

        remote_host = ssh_port_entry.get("ip") or "ssh.runpod.io"
        remote_port = int(ssh_port_entry.get("publicPort"))

        if ssh_key_path:
            print(f"[runpod] Using SSH key for upload: {ssh_key_path}")
        else:
            print("[runpod] SSH key not specified; relying on default SSH agent / config for upload.")

        port_wait_timeout = timeout if timeout is not None else 600
        ssh_username = "root"
        if remote_host.endswith("runpod.io"):
            ssh_username = pod_id

        try:
            _wait_for_ssh_ready(
                remote_host,
                remote_port,
                ssh_username,
                ssh_key_path,
                port_wait_timeout,
                interval=args.poll_interval,
            )
        except TimeoutError as exc:
            print(
                f"[runpod] WARNING: {exc}. Proceeding with upload retries anyway.",
                file=sys.stderr,
            )

        if checkpoint_remote_dir:
            print(f"[runpod] Preparing remote directory {checkpoint_remote_dir} for checkpoint upload...")
            try:
                _ensure_remote_directory(remote_host, remote_port, checkpoint_remote_dir, ssh_key_path)
            except subprocess.CalledProcessError as exc:
                print(f"[runpod] ERROR: Failed to prepare remote directory: {exc}", file=sys.stderr)
                return 1
            except FileNotFoundError as exc:
                print(f"[runpod] ERROR: SSH client not found ({exc}). Install openssh to enable uploads.", file=sys.stderr)
                return 1

        print(f"[runpod] Uploading checkpoint {checkpoint_local_path} -> {checkpoint_remote_path} ...")
        try:
            _upload_checkpoint(checkpoint_local_path, remote_host, remote_port, checkpoint_remote_path, ssh_key_path)
        except subprocess.CalledProcessError as exc:
            print(f"[runpod] ERROR: Checkpoint upload failed: {exc}", file=sys.stderr)
            return 1
        except FileNotFoundError as exc:
            print(f"[runpod] ERROR: SCP client not found ({exc}). Install openssh to enable uploads.", file=sys.stderr)
            return 1

        print("[runpod] Checkpoint upload completed. Training will resume once the bootstrap wait detects the file.")
        print(f"[runpod] RUNPOD_RESUME_CHECKPOINT inside pod: {checkpoint_remote_path}")

        runtime_info = pod_info.get("runtime") or {}
        runtime_ports = runtime_info.get("ports") or []

    print("[runpod] Pod is RUNNING. Connection details:")
    print(f"  Pod ID     : {pod_id}")
    print(f"  Machine ID : {pod_info.get('machineId')}")
    print(f"  GPU        : {pod_info.get('machine', {}).get('gpuDisplayName', 'n/a')}" )
    print(f"  Ports      :\n  - {_format_ports(runtime_ports)}")

    if ssh_port_entry and ssh_port_entry.get("publicPort"):
        ip_label = ssh_port_entry.get("ip") or "ssh.runpod.io"
        print("[runpod] Example SSH command:")
        print(f"  ssh -p {ssh_port_entry['publicPort']} root@{ip_label}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
