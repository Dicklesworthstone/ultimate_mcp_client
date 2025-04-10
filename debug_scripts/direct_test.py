import asyncio
import contextlib
import json
import os
import time

# --- Configuration ---
SHELL_COMMAND = "cd /home/ubuntu/llm_gateway_mcp_server && source .venv/bin/activate && python -m llm_gateway.cli.main run"
# Note: Removed the 'tee' redirection for simpler debugging
PROTOCOL_VERSION = "2025-03-25"
REQUEST_TIMEOUT = 45.0 # Keep increased timeout
# --- *** ADD BUFFER LIMIT *** ---
BUFFER_LIMIT = 2**18 # 256 KiB, adjust if needed (default is often 2**16 = 64 KiB)
# --- End Configuration ---


async def read_output(stream, name, output_list):
    """Read lines from stdout/stderr and print/store them."""
    # NOTE: The limit is applied when the stream is *created*,
    # this function just reads from the existing stream.
    while True:
        try:
            # This readline call now respects the limit set during stream creation
            line_bytes = await stream.readline()
            if not line_bytes:
                print(f"[{name}] ### EOF ###", flush=True)
                break
            line_str_raw = line_bytes.decode('utf-8', errors='replace')
            print(f"[{name}] RAW <<< {repr(line_str_raw)}", flush=True) # Log raw line with repr
            line_str_stripped = line_str_raw.strip()
            if line_str_stripped: # Only add non-empty lines to list
                output_list.append(line_str_stripped)
        except ValueError as e: # Catch the specific limit error from StreamReader
            if "Separator is found, but chunk is longer than limit" in str(e):
                 # Provide informative error message if limit is hit
                 print(f"[{name}] !!! Buffer limit ({BUFFER_LIMIT} bytes) exceeded reading line! Increase BUFFER_LIMIT in script.", flush=True)
            else:
                 # Log other ValueErrors if they occur
                 print(f"[{name}] !!! ValueError reading stream: {e}", flush=True)
            break # Stop reading on buffer limit or other ValueError
        except Exception as e:
            # Catch any other reading errors
            print(f"[{name}] !!! Error reading stream: {e}", flush=True)
            break

async def write_to_stdin(proc, data_bytes):
    """Write bytes to stdin and drain."""
    if proc.stdin is None or proc.stdin.is_closing():
        print("[CLIENT] !!! Stdin is closed or None. Cannot write.", flush=True)
        return False
    try:
        print(f"[CLIENT] RAW >>> {repr(data_bytes)}", flush=True) # Log raw bytes being sent
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
        json_str = json.dumps(message_dict) + '\n'
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
    last_checked_index = 0 # Optimize by not re-checking old lines
    while time.monotonic() - start_time < mcp_timeout:
        # Check only new lines added since last check
        current_len = len(stdout_lines)
        for i in range(last_checked_index, current_len):
            line = stdout_lines[i]
            try:
                msg = json.loads(line)
                if isinstance(msg, dict) and msg.get("id") == request_id:
                    if "result" in msg:
                        print(f"[CLIENT] <<< Received OK response for ID {request_id}.", flush=True)
                        return msg # Return the full response message
                    elif "error" in msg:
                        print(f"[CLIENT] <<< Received ERROR response for ID {request_id}: {msg['error']}", flush=True)
                        return msg # Return the error response
            except Exception: # Broad catch for JSON errors or non-dict messages
                pass # Ignore non-JSON or invalid lines for this check
        last_checked_index = current_len # Update checked index
        await asyncio.sleep(0.05) # Shorter sleep for faster checking

    print(f"[CLIENT] !!! Timeout waiting for response ID {request_id}.", flush=True)
    return None


async def main():
    print("Starting server process via bash shell...")
    # --- *** CORRECTED PROCESS CREATION WITH LIMIT *** ---
    proc = await asyncio.create_subprocess_shell(
            SHELL_COMMAND,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=BUFFER_LIMIT, # <--- Apply the defined buffer limit
            env=os.environ.copy(), # Inherit environment
            executable='/bin/bash' # <--- Explicitly specify bash
        )
    # --- *** END CORRECTION *** ---
    print(f"Server process started with PID: {proc.pid}")

    stdout_lines = []
    stderr_lines = []

    # Start readers in the background
    stdout_reader = asyncio.create_task(read_output(proc.stdout, "STDOUT", stdout_lines))
    stderr_reader = asyncio.create_task(read_output(proc.stderr, "STDERR", stderr_lines))

    response = None # Variable to hold responses

    try:
        await asyncio.sleep(1.5) # Give server a bit more time to maybe output startup messages

        # 1. Send Initialize
        init_id = 1
        init_req = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "capabilities": {},
                "clientInfo": {"name": "direct-test-client-v2", "version": "1.0.1"},
                "protocolVersion": PROTOCOL_VERSION
            },
            "id": init_id
        }
        if not await send_jsonrpc(proc, init_req): return # Exit if send fails

        # Wait for InitializeResult
        response = await wait_for_response(init_id, stdout_lines, REQUEST_TIMEOUT)
        if response is None or "error" in response:
             print("[CLIENT] !!! Failed to get successful InitializeResult.", flush=True)
             return # Stop if init fails

        # 2. Send Initialized Notification
        init_notif = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        if not await send_jsonrpc(proc, init_notif): return # Exit if send fails
        await asyncio.sleep(0.1) # Small pause after notification

        # 3. Send List Tools
        list_tools_id = 2
        list_tools_req = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": list_tools_id
        }
        if not await send_jsonrpc(proc, list_tools_req): return # Exit if send fails

        # Wait for ListToolsResult
        response = await wait_for_response(list_tools_id, stdout_lines, REQUEST_TIMEOUT)
        if response is None or "error" in response:
            print("[CLIENT] !!! Failed to get successful ListToolsResult.", flush=True)
            # Don't return yet, let cleanup happen
        else:
            print("[CLIENT] --- Handshake Successful (Initialize + ListTools) ---", flush=True)

        # Keep readers running for a bit longer to catch any delayed output
        print("[CLIENT] Waiting for any remaining output...")
        await asyncio.sleep(5.0)

    except Exception as e:
        print(f"[CLIENT] !!! An error occurred during the main test logic: {e}", flush=True)

    finally:
        print("[CLIENT] Cleaning up...")
        # Cancel readers
        stdout_reader.cancel()
        stderr_reader.cancel()
        with contextlib.suppress(asyncio.CancelledError): await stdout_reader
        with contextlib.suppress(asyncio.CancelledError): await stderr_reader

        # Terminate server process
        if proc.returncode is None:
            print("[CLIENT] Terminating server process...")
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                print("[CLIENT] Server did not terminate gracefully, killing...")
                if proc.returncode is None: proc.kill()
            except ProcessLookupError: pass # Process already exited
            except Exception as term_err: print(f"[CLIENT] Error terminating: {term_err}")

            if proc.returncode is not None:
                 print(f"[CLIENT] Server process terminated with code {proc.returncode}.")
            else:
                 print("[CLIENT] Server process might still be running after kill attempt.")
        else:
             print(f"[CLIENT] Server process already exited with code {proc.returncode}.")


if __name__ == "__main__":
    # Ensure necessary imports are at the top
    # (sys, contextlib should be added if not already present)
    import contextlib
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[CLIENT] Test interrupted by user.")
