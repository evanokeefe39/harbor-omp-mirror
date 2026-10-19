/**
 * A minimal, dependency-free MCP stdio server.
 *
 * Speaks newline-delimited JSON-RPC 2.0 on stdin/stdout, which is exactly the
 * framing omp's stdio transport uses ("Messages are newline-delimited JSON",
 * `src/mcp/transports/stdio.ts`; frames are written as
 * `${JSON.stringify(response)}\n`).
 *
 * Launched from the agent directory's `mcp.json` as:
 *
 *   "command": "${HOME}/.bun/bin/bun"
 *   "args":    ["${HOME}/.omp/agent/mcp/harbor-hello-mcp.js"]
 *
 * omp expands `${VAR}` / `${VAR:-default}` in a discovered `mcp.json` at load
 * time (`expandEnvVarsDeep`, `src/discovery/helpers.ts`, applied by the native
 * provider `src/discovery/builtin.ts` and the standalone reader
 * `src/discovery/mcp-json.ts`), and on POSIX then hands argv to `Bun.spawn`
 * verbatim with no shell (`resolveStdioSpawnCommand`: `cmd: [config.command,
 * ...args]`). So both paths must be written as `${HOME}/…`, never `$HOME/…` or
 * `~/…`, and neither is quoted — argv needs no quoting because no shell sees it.
 *
 * Exposes one tool, `echo`. omp registers it as `mcp__harbor_fixture_live_echo`
 * (server `harbor-fixture-live` + tool `echo`, per `createMCPToolName`).
 */

const SERVER_NAME = "harbor-hello-mcp";
const SERVER_VERSION = "0.1.0";
const DEFAULT_PROTOCOL_VERSION = "2025-11-25";
const MARKER = "HARBOR_OMP_FIXTURE_MCP";

const TOOL = {
	name: "echo",
	title: "Echo",
	description: `Echo a string back, prefixed with ${MARKER}.`,
	inputSchema: {
		type: "object",
		properties: {
			text: { type: "string", description: "Text to echo back." },
		},
		required: [],
	},
	annotations: { readOnlyHint: true, idempotentHint: true, openWorldHint: false },
};

function send(message) {
	process.stdout.write(`${JSON.stringify(message)}\n`);
}

function respond(id, result) {
	send({ jsonrpc: "2.0", id, result });
}

function respondError(id, code, message) {
	send({ jsonrpc: "2.0", id, error: { code, message } });
}

function handle(message) {
	const { id, method, params } = message;
	// JSON-RPC notification: no `id`. Never answered, even on failure.
	const isNotification = id === undefined || id === null;

	switch (method) {
		case "initialize": {
			const requested = typeof params?.protocolVersion === "string" ? params.protocolVersion : undefined;
			respond(id, {
				protocolVersion: requested || DEFAULT_PROTOCOL_VERSION,
				capabilities: { tools: { listChanged: false } },
				serverInfo: { name: SERVER_NAME, version: SERVER_VERSION },
				instructions:
					`${MARKER}: the harbor-fixture-live MCP server is loaded and serving tools. ` +
					"Call mcp__harbor_fixture_live_echo to observe it.",
			});
			return;
		}
		case "notifications/initialized":
		case "notifications/cancelled":
			return;
		case "ping":
			if (!isNotification) respond(id, {});
			return;
		case "tools/list":
			respond(id, { tools: [TOOL] });
			return;
		case "tools/call": {
			const name = params?.name;
			if (name !== TOOL.name) {
				respond(id, {
					content: [{ type: "text", text: `Unknown tool: ${String(name)}` }],
					isError: true,
				});
				return;
			}
			const text = typeof params?.arguments?.text === "string" ? params.arguments.text : "";
			respond(id, { content: [{ type: "text", text: `${MARKER} echo: ${text}` }], isError: false });
			return;
		}
		default:
			if (!isNotification) respondError(id, -32601, `Method not found: ${String(method)}`);
	}
}

const decoder = new TextDecoder();
let buffer = "";

function consume(chunkText) {
	buffer += chunkText;
	let newline = buffer.indexOf("\n");
	while (newline !== -1) {
		const line = buffer.slice(0, newline).trim();
		buffer = buffer.slice(newline + 1);
		newline = buffer.indexOf("\n");
		if (!line) continue;
		let message;
		try {
			message = JSON.parse(line);
		} catch {
			continue; // malformed frame: drop it, never crash the transport
		}
		try {
			handle(message);
		} catch (error) {
			if (message?.id !== undefined && message?.id !== null) {
				respondError(message.id, -32603, String(error));
			}
		}
	}
}

async function main() {
	const stdin = typeof Bun !== "undefined" && Bun.stdin ? Bun.stdin.stream() : process.stdin;
	for await (const chunk of stdin) {
		consume(typeof chunk === "string" ? chunk : decoder.decode(chunk, { stream: true }));
	}
}

main().catch(error => {
	process.stderr.write(`${MARKER} server failed: ${String(error)}\n`);
	process.exit(1);
});
