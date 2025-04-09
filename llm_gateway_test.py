#!/usr/bin/env python3
import asyncio
import contextlib
import os
import time
import traceback  # For enhanced error reporting

# --- Configuration ---
SHELL_COMMAND = "cd /home/ubuntu/llm_gateway_mcp_server && source .venv/bin/activate && python -m llm_gateway.cli.main run"
PROTOCOL_VERSION = "2025-03-25"
REQUEST_TIMEOUT = 60.0  # Increased timeout
BUFFER_LIMIT = 2**18  # 256 KiB

# Use orjson if available, fallback to standard json
try:
    import orjson as json_parser
    print("[INFO] Using orjson for JSON operations.")
except ImportError:
    import json as json_parser
    print("[INFO] Using standard json library.")

async def read_output(stream, name, output_list):
    """Read lines from stdout/stderr and print/store them."""
    while True:
        try:
            line_bytes = await stream.readline()
            if not line_bytes:
                print(f"[{name}] ### EOF ###", flush=True)
                break
            line_str_raw = line_bytes.decode('utf-8', errors='replace')
            print(f"[{name}] <<< {repr(line_str_raw)}", flush=True)  # Log raw line with repr
            line_str_stripped = line_str_raw.strip()
            if line_str_stripped:  # Only add non-empty lines to list
                output_list.append(line_str_stripped)
        except ValueError as e:
            if "Separator is found, but chunk is longer than limit" in str(e):
                print(f"[{name}] !!! Buffer limit ({BUFFER_LIMIT} bytes) exceeded reading line! Increase BUFFER_LIMIT.", flush=True)
            else:
                print(f"[{name}] !!! ValueError reading stream: {e}", flush=True)
            break
        except Exception as e:
            print(f"[{name}] !!! Error reading stream: {e}", flush=True)
            break

async def write_to_stdin(proc, data_bytes):
    """Write bytes to stdin and drain."""
    if proc.stdin is None or proc.stdin.is_closing():
        print("[CLIENT] !!! Stdin is closed or None. Cannot write.", flush=True)
        return False
    try:
        print(f"[CLIENT] >>> {repr(data_bytes)}", flush=True)  # Log raw bytes being sent
        proc.stdin.write(data_bytes)
        await proc.stdin.drain()
        return True
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
        print(f"[CLIENT] !!! Stdin write failed (Connection Error): {e}", flush=True)
        return False
    except Exception as e:
        print(f"[CLIENT] !!! Stdin write failed (Other Error): {e}", flush=True)
        return False

async def send_jsonrpc(proc, message_dict):
    """Serialize dict to JSON, add newline, encode, and send."""
    try:
        # Use orjson dumps if available (returns bytes), else standard json + encode
        if json_parser.__name__ == 'orjson':
            json_bytes = json_parser.dumps(message_dict) + b'\n'
        else:
            json_str = json_parser.dumps(message_dict) + '\n'
            json_bytes = json_str.encode('utf-8')

        method_or_id = message_dict.get('method', message_dict.get('id', 'N/A'))
        print(f"[CLIENT] >>> Sending JSON ({method_or_id})...", flush=True)
        return await write_to_stdin(proc, json_bytes)
    except Exception as e:
        print(f"[CLIENT] !!! Error serializing/sending JSON: {e}", flush=True)
        return False

async def wait_for_response(request_id, stdout_lines, mcp_timeout):
    """Wait for a specific JSON-RPC response ID in the output lines."""
    start_time = time.monotonic()
    last_checked_index = 0  # Optimize by not re-checking old lines
    while time.monotonic() - start_time < mcp_timeout:
        # Check only new lines added since last check
        current_len = len(stdout_lines)
        for i in range(last_checked_index, current_len):
            line = stdout_lines[i]
            try:
                msg = json_parser.loads(line)
                if isinstance(msg, dict) and msg.get("id") == request_id:
                    if "result" in msg:
                        print(f"[CLIENT] <<< Received OK response for ID {request_id}.", flush=True)
                        return msg  # Return the full response message
                    elif "error" in msg:
                        print(f"[CLIENT] <<< Received ERROR response for ID {request_id}: {msg['error']}", flush=True)
                        return msg  # Return the error response
            except Exception:
                # Just ignore non-JSON or invalid lines
                pass
        last_checked_index = current_len  # Update checked index
        await asyncio.sleep(0.05)  # Shorter sleep for faster checking

    print(f"[CLIENT] !!! Timeout waiting for response ID {request_id}.", flush=True)
    return None

async def main():
    print("[CLIENT] Starting server process via bash shell...")
    
    try:
        proc = await asyncio.create_subprocess_shell(
            SHELL_COMMAND,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=BUFFER_LIMIT,  # Apply the defined buffer limit
            env=os.environ.copy(),  # Inherit environment
            executable='/bin/bash'  # Explicitly specify bash
        )
        print(f"[CLIENT] Server process started with PID: {proc.pid}")

        stdout_lines = []
        stderr_lines = []

        # Start readers in the background
        stdout_reader = asyncio.create_task(read_output(proc.stdout, "STDOUT", stdout_lines))
        stderr_reader = asyncio.create_task(read_output(proc.stderr, "STDERR", stderr_lines))

        # Give server more time to start up before sending any messages
        print("[CLIENT] Waiting 3 seconds for server startup...")
        await asyncio.sleep(3.0)
        
        # 1. Send Initialize
        print("[CLIENT] Sending initialize request...")
        init_id = 1
        init_req = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "capabilities": {},
                "clientInfo": {"name": "llm-gateway-test", "version": "1.0.0"},
                "protocolVersion": PROTOCOL_VERSION
            },
            "id": init_id
        }
        if not await send_jsonrpc(proc, init_req):
            print("[CLIENT] Failed to send initialize request")
            return

        # Wait for InitializeResult
        print(f"[CLIENT] Waiting for initialize response (timeout: {REQUEST_TIMEOUT}s)...")
        response = await wait_for_response(init_id, stdout_lines, REQUEST_TIMEOUT)
        if response is None:
            print("[CLIENT] !!! Initialize response timed out")
            return
        elif "error" in response:
            print("[CLIENT] !!! Initialize response contained error")
            return
        else:
            print("[CLIENT] Initialize successful!")

        # 2. Send Initialized Notification
        print("[CLIENT] Sending initialized notification...")
        init_notif = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        if not await send_jsonrpc(proc, init_notif):
            print("[CLIENT] Failed to send initialized notification")
            return
        await asyncio.sleep(0.5)  # Increased pause after notification

        # 3. Send List Tools
        print("[CLIENT] Sending list_tools request...")
        list_tools_id = 2
        list_tools_req = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": list_tools_id
        }
        if not await send_jsonrpc(proc, list_tools_req):
            print("[CLIENT] Failed to send list_tools request")
            return

        # Wait for ListToolsResult
        print(f"[CLIENT] Waiting for list_tools response (timeout: {REQUEST_TIMEOUT}s)...")
        response = await wait_for_response(list_tools_id, stdout_lines, REQUEST_TIMEOUT)
        if response is None:
            print("[CLIENT] !!! list_tools response timed out")
        elif "error" in response:
            print("[CLIENT] !!! list_tools response contained error")
        else:
            print("[CLIENT] list_tools successful!")
            # Print the tools if available
            tools = response.get("result", {}).get("tools", [])
            print(f"[CLIENT] Found {len(tools)} tools")
            for tool in tools:
                print(f"[CLIENT] - {tool.get('name')}: {tool.get('description', 'No description')}")

        print("[CLIENT] Handshake completed successfully!")
        print("[CLIENT] Waiting for any remaining output...")
        await asyncio.sleep(5.0)

    except Exception as e:
        print(f"[CLIENT] !!! An error occurred: {e}", flush=True)
        print("[CLIENT] Exception details:", flush=True)
        traceback.print_exc()

    finally:
        print("[CLIENT] Cleaning up...")
        try:
            # Cancel readers
            if 'stdout_reader' in locals():
                stdout_reader.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stdout_reader
            
            if 'stderr_reader' in locals():
                stderr_reader.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stderr_reader

            # Terminate server process
            if 'proc' in locals() and proc.returncode is None:
                print("[CLIENT] Terminating server process...")
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    print("[CLIENT] Server did not terminate gracefully, killing...")
                    if proc.returncode is None:
                        proc.kill()
                except ProcessLookupError:
                    pass  # Process already exited
                except Exception as term_err:
                    print(f"[CLIENT] Error terminating: {term_err}")

                if proc.returncode is not None:
                    print(f"[CLIENT] Server process terminated with code {proc.returncode}.")
                else:
                    print("[CLIENT] Server process might still be running after kill attempt.")
            elif 'proc' in locals():
                print(f"[CLIENT] Server process already exited with code {proc.returncode}.")
        except Exception as cleanup_err:
            print(f"[CLIENT] Error during cleanup: {cleanup_err}")
            traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[CLIENT] Test interrupted by user.")
    except Exception as e:
        print(f"[CLIENT] Unhandled exception: {e}")
        traceback.print_exc()
