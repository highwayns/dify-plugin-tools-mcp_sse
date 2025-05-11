import json
import logging
from abc import ABC, abstractmethod
from queue import Queue, Empty
from threading import Event, Thread
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from httpx import Response
from httpx_sse import connect_sse
import jwt  # 需要安装 PyJWT

class McpAuthClientMixin:
    def __init__(self, login_url: str, username: str, password: str):
        self.login_url = login_url
        self.username = username
        self.password = password
        self.token = None

    def authenticate(self) -> str:
        """Authenticate to MCPHub and retrieve JWT token"""
        response = httpx.post(
            self.login_url,
            headers={"Content-Type": "application/json"},
            json={"username": self.username, "password": self.password},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        if "token" not in data:
            raise Exception("MCPHub login failed: No token returned")
        self.token = data["token"]
        return self.token

    def get_auth_headers(self) -> dict[str, str]:
        if not self.token:
            self.authenticate()
        return {"Authorization": f"Bearer {self.token}"}

class McpClient(ABC):
    """Interface for MCP client."""

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def initialize(self):
        raise NotImplementedError

    @abstractmethod
    def list_tools(self):
        raise NotImplementedError

    @abstractmethod
    def call_tool(self, tool_name: str, tool_args: dict):
        raise NotImplementedError


def remove_request_params(url: str) -> str:
    return urljoin(url, urlparse(url).path)


class McpSseClient(McpClient):
    """
    HTTP with SSE transport MCP client.
    """

    def __init__(self, name: str, url: str,
                 headers: dict[str, Any] | None = None,
                 timeout: float = 50,
                 sse_read_timeout: float = 50,
                 ):
        self.name = name
        self.url = url
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout
        self.endpoint_url = None
        #self.client = httpx.Client(headers=headers)
        auth_headers = headers or {}
        if isinstance(self, McpAuthClientMixin):
            auth_headers = self.get_auth_headers()
        if headers:
            auth_headers.update(headers)  # headers 的优先级更高
        self.client = httpx.Client(headers=auth_headers)

        self._request_id = 0
        self.message_queue = Queue()
        self.response_ready = Event()
        self.should_stop = Event()
        self._listen_thread = None
        self._connected = Event()
        self._error_event = Event()
        self._thread_exception = None
        self.connect()

    def _listen_messages(self) -> None:
        try:
            logging.info(f"{self.name} - Connecting to SSE endpoint: {remove_request_params(self.url)}")
            with connect_sse(
                    client=self.client,
                    method="GET",
                    url=self.url,
                    timeout=httpx.Timeout(self.timeout, read=self.sse_read_timeout),
            ) as event_source:
                event_source.response.raise_for_status()
                logging.debug(f"{self.name} - SSE connection established")
                for sse in event_source.iter_sse():
                    logging.debug(f"{self.name} - Received SSE event: {sse.event}")
                    if self.should_stop.is_set():
                        break
                    match sse.event:
                        case "endpoint":
                            self.endpoint_url = urljoin(self.url, sse.data)
                            logging.info(f"{self.name} - Received endpoint URL: {self.endpoint_url}")
                            self._connected.set()
                            url_parsed = urlparse(self.url)
                            endpoint_parsed = urlparse(self.endpoint_url)
                            if (url_parsed.netloc != endpoint_parsed.netloc
                                    or url_parsed.scheme != endpoint_parsed.scheme):
                                error_msg = f"{self.name} - Endpoint origin does not match connection origin: {self.endpoint_url}"
                                logging.error(error_msg)
                                raise ValueError(error_msg)
                        case "message":
                            message = json.loads(sse.data)
                            logging.debug(f"{self.name} - Received server message: {message}")
                            self.message_queue.put(message)
                            self.response_ready.set()
                        case _:
                            logging.warning(f"{self.name} - Unknown SSE event: {sse.event}")
        except Exception as e:
            self._thread_exception = e
            self._error_event.set()
            self._connected.set()

    def send_message(self, data: dict):
        if not self.endpoint_url:
            if self._thread_exception:
                raise ConnectionError(f"{self.name} - MCP Server connection failed: {self._thread_exception}")
            else:
                raise RuntimeError(f"{self.name} - Please call connect() first")
        logging.debug(f"{self.name} - Sending client message: {data}")
        response = self.client.post(
            url=self.endpoint_url,
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=self.timeout
        )
        response.raise_for_status()
        logging.debug(f"{self.name} - Client message sent successfully: {response.status_code}")
        if "id" in data:
            message_id = data["id"]
            while True:
                self.response_ready.wait()
                self.response_ready.clear()
                try:
                    while True:
                        message = self.message_queue.get_nowait()
                        if "id" in message and message["id"] == message_id:
                            self._request_id += 1
                            return message
                        self.message_queue.put(message)
                except Empty:
                    pass
        return {}

    def connect(self) -> None:
        self._listen_thread = Thread(target=self._listen_messages, daemon=True)
        self._listen_thread.start()
        while True:
            if self._error_event.is_set():
                if isinstance(self._thread_exception, httpx.HTTPStatusError):
                    raise ConnectionError(f"{self.name} - MCP Server connection failed: {self._thread_exception}") \
                        from self._thread_exception
                else:
                    raise self._thread_exception
            if self._connected.wait(timeout=0.1):
                break
            if not self._listen_thread.is_alive():
                raise ConnectionError(f"{self.name} - MCP Server SSE listener thread died unexpectedly!")

    def close(self) -> None:
        try:
            self.should_stop.set()
            self.client.close()
            if self._listen_thread and self._listen_thread.is_alive():
                self._listen_thread.join(timeout=10)
        except Exception as e:
            raise Exception(f"{self.name} - MCP Server connection close failed: {str(e)}")

    def initialize(self):
        init_data = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "MCP HTTP with SSE Client",
                    "version": "1.0.0"
                }
            }
        }
        response = self.send_message(init_data)
        if "error" in response:
            raise Exception(f"MCP Server initialize error: {response['error']}")
        notify_data = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        response = self.send_message(notify_data)
        if "error" in response:
            raise Exception(f"MCP Server notifications/initialized error: {response['error']}")

    def list_tools(self):
        tools_data = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/list",
            "params": {}
        }
        response = self.send_message(tools_data)
        if "error" in response:
            raise Exception(f"MCP Server tools/list error: {response['error']}")
        return response.get("result", {}).get("tools", [])

    def call_tool(self, tool_name: str, tool_args: dict):
        call_data = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": tool_args
            }
        }
        response = self.send_message(call_data)
        if "error" in response:
            raise Exception(f"MCP Server tools/call error: {response['error']}")
        return response.get("result", {}).get("content", [])

class SecureMcpSseClient(McpAuthClientMixin, McpSseClient):
    def __init__(self, name, url, login_url, username, password,
                 timeout=50, sse_read_timeout=50):
        McpAuthClientMixin.__init__(self, login_url, username, password)
        McpSseClient.__init__(self, name, url,
                              headers=self.get_auth_headers(),
                              timeout=timeout,
                              sse_read_timeout=sse_read_timeout)

class McpStreamableHttpClient(McpClient):
    """
    Streamable HTTP transport MCP client.
    """

    def __init__(self, name: str, url: str,
                 headers: dict[str, Any] | None = None,
                 timeout: float = 50,
                 ):
        self.name = name
        self.url = url
        self.timeout = timeout
        auth_headers = headers or {}
        if isinstance(self, McpAuthClientMixin):
            auth_headers = self.get_auth_headers()
        if headers:
            auth_headers.update(headers)  # headers 的优先级更高
        self.client = httpx.Client(headers=auth_headers)
        self.session_id = None  # 初始化为空    
    def close(self) -> None:
        try:
            self.client.close()
        except Exception as e:
            raise Exception(f"{self.name} - MCP Server connection close failed: {str(e)}")

    def send_message(self, data: dict) -> Response:
        headers = {"Content-Type": "application/json"}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        logging.debug(f"{self.name} - Sending client message: {data}")
        response = self.client.post(
            url=self.url,
            json=data,
            headers=headers,
            timeout=self.timeout
        )
        logging.debug(f"{self.name} - Client message sent successfully: {response.status_code}")
        return response

    def initialize(self):
        init_data = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "MCP Streamable HTTP Client",
                    "version": "1.0.0"
                }
            }
        }
        response = self.send_message(init_data)
        # 不再从 response.headers 获取 session_id
        notify_data = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        self.send_message(notify_data)

    def list_tools(self):
        tools_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
        response = self.send_message(tools_data)
        response_data = response.json()
        if "error" in response_data:
            raise Exception(f"MCP Server tools/list error: {response_data['error']}")
        return response_data.get("result", {}).get("tools", [])

    def call_tool(self, tool_name: str, tool_args: dict):
        call_data = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": tool_args
            }
        }
        response = self.send_message(call_data)
        response_data = response.json()
        if "error" in response_data:
            raise Exception(f"MCP Server tools/call error: {response_data['error']}")
        return response_data.get("result", {}).get("content", [])

class SecureMcpHttpClient(McpAuthClientMixin, McpStreamableHttpClient):
    def __init__(self, name, url, login_url, username, password,
                 timeout=50):
        McpAuthClientMixin.__init__(self, login_url, username, password)
        McpStreamableHttpClient.__init__(self, name, url,
                                         headers=self.get_auth_headers(),
                                         timeout=timeout)

class McpClients:
    def __init__(self, servers_config: dict[str, Any]):
        if "mcpServers" in servers_config:
            servers_config = servers_config["mcpServers"]
        self._clients = {
            name: self.init_client(name, config)
            for name, config in servers_config.items()
        }
        for client in self._clients.values():
            client.initialize()
        self._tools = {}

    @staticmethod
    def init_client(name: str, config: dict[str, Any]) -> McpClient:
        transport = "sse"
        if "transport" in config:
            transport = config["transport"]
        if transport == "streamable_http":
            return SecureMcpHttpClient(
                name=name,
                url=config.get("url"),
                #headers=config.get("headers", None),
                timeout=config.get("timeout", 50),
                login_url=config.get("login_url"),
                username=config.get("login_user"),
                password=config.get("login_password"),
            )
        return SecureMcpSseClient(
            name=name,
            url=config.get("url"),
            #headers=config.get("headers", None),
            timeout=config.get("timeout", 50),
            sse_read_timeout=config.get("sse_read_timeout", 50),
            login_url=config.get("login_url"),
            username=config.get("login_user"),
            password=config.get("login_password"),
        )

    def fetch_tools(self) -> list[dict]:
        try:
            all_tools = []
            for server_name, client in self._clients.items():
                tools = client.list_tools()
                all_tools.extend(tools)
                self._tools[server_name] = tools
            return all_tools
        except Exception as e:
            raise RuntimeError(f"Error fetching tools: {str(e)}")

    def execute_tool(self, tool_name: str, tool_args: dict[str, Any]):
        if not self._tools:
            self.fetch_tools()
        tool_clients = {}
        for server_name, tools in self._tools.items():
            for tool in tools:
                if server_name in self._clients:
                    tool_clients[tool["name"]] = self._clients[server_name]
        client = tool_clients.get(tool_name, None)
        try:
            if not client:
                raise Exception(f"there is not a tool named {tool_name}")
            result = client.call_tool(tool_name, tool_args)
            if isinstance(result, dict) and "progress" in result:
                progress = result["progress"]
                total = result["total"]
                percentage = (progress / total) * 100
                logging.info(
                    f"Progress: {progress}/{total} "
                    f"({percentage:.1f}%)"
                )
            return f"Tool execution result: {result}"
        except Exception as e:
            error_msg = f"Error executing tool: {str(e)}"
            logging.error(error_msg)
            return error_msg

    def close(self) -> None:
        for client in self._clients.values():
            try:
                client.close()
            except Exception as e:
                logging.error(e)

if __name__ == "__main__":
    # 示例参数，请根据实际情况修改
    login_url = "http://localhost:3004/auth/login"
    username = "admin"
    password = "admin123"

    print("Testing McpAuthClientMixin authentication...")
    auth_client = McpAuthClientMixin(login_url, username, password)
    try:
        token = auth_client.authenticate()
        print("Authentication successful. Token:", token)
        headers = auth_client.get_auth_headers()
        print("Auth headers:", headers)
    except Exception as e:
        print("Authentication failed:", str(e))

    # 测试 McpClients
    print("\nTesting McpClients...")
    servers_config = {
        "mcpServers": {
            "local_server": {
                "url": "http://localhost:3004/sse",
                "login_url": login_url,
                "login_user": username,
                "login_password": password,
                "transport": "sse",  # 或 "streamable_http"
                "timeout": 30
            }
        }
    }
    try:
        mcp_clients = McpClients(servers_config)
        print("McpClients initialized successfully.")

        # 获取工具列表
        tools = mcp_clients.fetch_tools()
        print(f"Fetched tools: {tools}")

        # 如果有工具，尝试执行第一个工具
        if tools:
            tool_name = tools[0]["name"]
            print(f"Executing tool: {tool_name}")
            # 示例参数，根据实际工具调整
            tool_args = {}
            result = mcp_clients.execute_tool(tool_name, tool_args)
            print(f"Tool execution result: {result}")
        else:
            print("No tools found on the server.")

        mcp_clients.close()
        print("McpClients closed successfully.")
    except Exception as e:
        print("McpClients test failed:", str(e))
