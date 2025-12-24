"""
Network utilities for port allocation and connectivity checks.

Consolidates port-finding logic from Docker and Local controllers.
"""

import socket
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def find_available_port(
    start_port: int,
    end_port: int,
    host: str = "0.0.0.0"
) -> Optional[int]:
    """
    Find an available port in the specified range.
    
    Tests each port by attempting to bind a socket. This is the standard
    way to check port availability.
    
    Args:
        start_port: Start of port range (inclusive)
        end_port: End of port range (inclusive)
        host: Host interface to bind to
        
    Returns:
        Available port number, or None if no ports available
        
    Example:
        >>> port = find_available_port(8000, 8100)
        >>> if port:
        ...     print(f"Using port {port}")
        ... else:
        ...     print("No available ports")
    """
    for port in range(start_port, end_port + 1):
        try:
            # Try to bind to the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
                # If bind succeeds, port is available
                logger.debug("Port %d is available", port)
                return port
                
        except OSError:
            # Port is in use, try next one
            logger.debug("Port %d is in use", port)
            continue
            
    logger.warning("No available ports in range %d-%d", start_port, end_port)
    return None


def is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    """
    Check if a specific port is available.
    
    Args:
        port: Port number to check
        host: Host interface to check
        
    Returns:
        True if port is available, False if in use
        
    Example:
        >>> if is_port_available(8000):
        ...     server.start(port=8000)
        ... else:
        ...     print("Port 8000 is already in use")
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
    except OSError:
        return False


def find_available_ports(
    count: int,
    start_port: int,
    end_port: int,
    host: str = "0.0.0.0"
) -> list[int]:
    """
    Find multiple available ports in the specified range.
    
    Args:
        count: Number of ports needed
        start_port: Start of port range
        end_port: End of port range
        host: Host interface to bind to
        
    Returns:
        List of available port numbers (may be shorter than count if not enough available)
        
    Example:
        >>> ports = find_available_ports(3, 8000, 8100)
        >>> print(f"Allocated ports: {ports}")
        Allocated ports: [8000, 8001, 8005]
    """
    available_ports = []
    
    for port in range(start_port, end_port + 1):
        if is_port_available(port, host):
            available_ports.append(port)
            
            if len(available_ports) >= count:
                break
                
    if len(available_ports) < count:
        logger.warning(
            "Only found %d available ports, needed %d",
            len(available_ports),
            count
        )
        
    return available_ports


def get_local_ip() -> str:
    """
    Get the local IP address of this machine.
    
    Uses a trick of connecting to a public IP (doesn't actually send data)
    to determine which interface would be used for external connections.
    
    Returns:
        Local IP address string (e.g., "192.168.1.100")
        Falls back to "127.0.0.1" if detection fails
        
    Example:
        >>> ip = get_local_ip()
        >>> print(f"Local IP: {ip}")
        Local IP: 192.168.1.100
    """
    try:
        # Create a socket and connect to a public IP (doesn't actually connect)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            local_ip = sock.getsockname()[0]
            return local_ip
    except Exception:
        return "127.0.0.1"


def check_port_connectivity(
    host: str,
    port: int,
    timeout: float = 2.0
) -> tuple[bool, Optional[str]]:
    """
    Check if a port is reachable (accepts connections).
    
    Different from is_port_available() - this checks if a service
    is actually listening on the port.
    
    Args:
        host: Host to connect to
        port: Port to connect to
        timeout: Connection timeout in seconds
        
    Returns:
        (is_reachable, error_message) tuple:
        - is_reachable: True if connection succeeded
        - error_message: None if OK, error description if not
        
    Example:
        >>> reachable, error = check_port_connectivity("localhost", 8000)
        >>> if reachable:
        ...     print("Service is up!")
        ... else:
        ...     print(f"Cannot connect: {error}")
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect((host, port))
            return True, None
            
    except socket.timeout:
        return False, f"Connection timeout after {timeout}s"
    except ConnectionRefusedError:
        return False, "Connection refused (port not listening)"
    except OSError as e:
        return False, f"OS error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"
