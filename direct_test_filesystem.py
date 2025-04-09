import asyncio
import contextlib
import json
import os
import time

# --- Configuration for Filesystem Server ---
# Use 'npx' directly, path adaptation happens BEFORE this script runs (in main client)
FILESYSTEM_EXECUTABLE = "npx"
FILESYSTEM_ARGS = [
    "-y",
    "@modelcontextprotocol/server-filesystem",
    # Use the CORRECT Linux paths that the main client determined
    "/mnt/c/Users/jeffr/OneDrive/Desktop",
    "/mnt/c/Users/jeffr/Downloads"
]
PROTOCOL_VERSION = "2025-03-25"
REQUEST_TIMEOUT = 45.0 # Keep increased timeout
# Increased buffer limit, might not be needed for filesystem server, but safe
BUFFER_LIMIT = 2**18 # 256 KiB
# --- End Configuration ---

async def read_output(stream, name, output_list):
    """Read lines from stdout/stderr and print/store them."""
    while True:
        try:
            line_bytes = await stream.readline()
            if not line_bytes:
                print(f"[{name}] ### EOF ###", flush=True)
                break
            line_str_raw = line_bytes.decode('utf-8', errors='replace')
            print(f"[{name}] RAW <<< {repr(line_str_raw)}", flush=True)
            line_str_stripped = line_str_raw.strip()
            if line_str_stripped:
                output_list.append(line_str_stripped)
        except ValueError as e:
            if "Separator is found, but chunk is longer than limit" in str(e):
                 print(f"[{name}] !!! Buffer limit ({BUFFER_LIMIT} bytes) exceeded! Increase BUFFER_LIMIT.", flush=True)
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
        print(f"[CLIENT] RAW >>> {repr(data_bytes)}", flush=True)
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
        json_bytes = (json.dumps(message_dict) + '\n').encode('utf-8')
        method_or_id = message_dict.get('method', message_dict.get('id', 'N/A'))
        print(f"[CLIENT] >>> Sending JSON ({method_or_id})...", flush=True)
        return await write_to_stdin(proc, json_bytes)
    except Exception as e:
        print(f"[CLIENT] !!! Error serializing/sending JSON: {e}", flush=True)
        return False

async def wait_for_response(request_id, stdout_lines, mcp_timeout):
    """Wait for a specific JSON-RPC response ID in the output lines."""
    start_time = time.monotonic()
    last_checked_index = 0
    while time.monotonic() - start_time < mcp_timeout:
        current_len = len(stdout_lines)
        for i in range(last_checked_index, current_len):
            line = stdout_lines[i]
            try:
                msg = json.loads(line)
                if isinstance(msg, dict) and str(msg.get("id", -1)) == str(request_id): # Compare as strings for safety
                    if "result" in msg:
                        print(f"[CLIENT] <<< Received OK response for ID {request_id}.", flush=True)
                        return msg
                    elif "error" in msg:
                        print(f"[CLIENT] <<< Received ERROR response for ID {request_id}: {msg['error']}", flush=True)
                        return msg
            except Exception: pass
        last_checked_index = current_len
        await asyncio.sleep(0.05)
    print(f"[CLIENT] !!! Timeout waiting for response ID {request_id}.", flush=True)
    return None


async def main():
    print(f"Starting server process: {FILESYSTEM_EXECUTABLE} {' '.join(FILESYSTEM_ARGS)}")
    # --- *** Use create_subprocess_exec for npx *** ---
    proc = await asyncio.create_subprocess_exec(
            FILESYSTEM_EXECUTABLE,
            *FILESYSTEM_ARGS, # Unpack args list
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=BUFFER_LIMIT, # Apply buffer limit
            env=os.environ.copy() # Inherit environment
        )
    # --- *** END CHANGE *** ---
    print(f"Server process started with PID: {proc.pid}")

    stdout_lines = []
    stderr_lines = []
    stdout_reader = asyncio.create_task(read_output(proc.stdout, "STDOUT", stdout_lines))
    stderr_reader = asyncio.create_task(read_output(proc.stderr, "STDERR", stderr_lines))

    response = None
    handshake_ok = False
    try:
        await asyncio.sleep(1.5)

        # 1. Send Initialize
        init_id = 1
        init_req = {
            "jsonrpc": "2.0", "method": "initialize", "id": init_id,
             "params": {
                 "capabilities": {},
                 "clientInfo": {"name": "fs-test-client", "version": "1.0.0"},
                 "protocolVersion": PROTOCOL_VERSION
            }
        }
        if not await send_jsonrpc(proc, init_req): return
        response = await wait_for_response(init_id, stdout_lines, REQUEST_TIMEOUT)
        if response is None or "error" in response:
             print("[CLIENT] !!! Failed Initialize.", flush=True)
             return

        # --- Check Capabilities from Initialize Result ---
        server_capabilities = response.get("result", {}).get("capabilities", {})
        print(f"[CLIENT] Server Capabilities: {server_capabilities}", flush=True)
        has_tools = "tools" in server_capabilities # Check only for presence of 'tools' key
        has_resources = "resources" in server_capabilities # Check for 'resources' key
        # --- End Capability Check ---

        # 2. Send Initialized Notification
        init_notif = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        if not await send_jsonrpc(proc, init_notif): return
        await asyncio.sleep(0.1)

        # 3. Try List Tools (only if advertised)
        list_tools_ok = False
        if has_tools:
            print("[CLIENT] Server advertises tools capability. Attempting tools/list...")
            list_tools_id = 2
            list_tools_req = {"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": list_tools_id}
            if not await send_jsonrpc(proc, list_tools_req): return
            response = await wait_for_response(list_tools_id, stdout_lines, REQUEST_TIMEOUT)
            if response is None:
                 print("[CLIENT] !!! Timeout on tools/list.", flush=True)
                 # Decide if this is fatal for handshake - for now, let's continue
                 list_tools_ok = False
            elif "error" in response:
                 # Specifically check for Method not found, which is expected for this server
                 if response["error"].get("code") == -32601:
                      print("[CLIENT] --- Confirmed: tools/list returned 'Method not found' as expected. ---", flush=True)
                      list_tools_ok = True # Mark as "ok" because this error is expected/handled
                 else:
                      print("[CLIENT] !!! Received unexpected error on tools/list.", flush=True)
                      list_tools_ok = False
            else:
                 print("[CLIENT] --- Received unexpected SUCCESS response for tools/list! ---", flush=True)
                 # print(response) # Optional: print the unexpected tool list
                 list_tools_ok = True
        else:
            print("[CLIENT] Server does NOT advertise tools capability. Skipping tools/list.", flush=True)
            list_tools_ok = True # Consider handshake ok if tools aren't expected

        # --- Consider handshake OK if initialize worked and tools check passed (or was skipped) ---
        if list_tools_ok: # Check if the tools step was ok (either succeeded, skipped, or expected error)
             handshake_ok = True
             print("[CLIENT] --- Core Handshake Sufficiently Completed ---", flush=True)
        else:
             print("[CLIENT] !!! Handshake Failed due to unexpected tools/list result.", flush=True)
             # No need to proceed if handshake failed here

        # --- **** Optional: Try resources/list if handshake OK **** ---
        if handshake_ok and has_resources:
            print("[CLIENT] Server advertises resources capability. Attempting resources/list...")
            list_res_id = 3
            list_res_req = {"jsonrpc": "2.0", "method": "resources/list", "params": {}, "id": list_res_id}
            if not await send_jsonrpc(proc, list_res_req): return

            # Wait for ListResourcesResult
            response = await wait_for_response(list_res_id, stdout_lines, REQUEST_TIMEOUT)
            if response is None or "error" in response:
                print("[CLIENT] !!! Failed to get successful ListResourcesResult.", flush=True)
            else:
                print("[CLIENT] --- Received ListResourcesResult ---", flush=True)
                # Optionally print the result content
                # print(response.get("result"))

        # Keep readers running for a bit longer
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
        # Terminate server process (same as before)
        if proc.returncode is None:
            print("[CLIENT] Terminating server process...")
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                if proc.returncode is None: proc.kill()
            except ProcessLookupError: pass
            except Exception as term_err: print(f"[CLIENT] Error terminating: {term_err}")
            print(f"[CLIENT] Server process terminated code: {proc.returncode}.")
        else: print(f"[CLIENT] Server process already exited code: {proc.returncode}.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[CLIENT] Test interrupted by user.")
