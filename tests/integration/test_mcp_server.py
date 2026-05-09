"""
Integration tests for MCP Server stdio communication.

Tests verify:
- JSON-RPC 2.0 protocol compliance
- stdout only contains MCP messages
- stderr contains logs
- initialize handshake works
- Multimodal content (text + images) assembly
"""

import json
import subprocess
import sys
import base64
import tempfile
from pathlib import Path
from src.core.response.multimodal_assembler import MultimodalAssembler


def test_server_initialize():
    """Test server can complete initialize handshake."""
    # Prepare initialize request
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }

    # Start server process
    server_path = Path(__file__).parent.parent.parent / "src" / "mcp_server" / "server.py"
    proc = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Send request
    stdout, stderr = proc.communicate(input=json.dumps(request) + "\n", timeout=5)

    # Verify stdout contains valid JSON-RPC response
    lines = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
    assert len(lines) >= 1, "Should have at least one response line"

    response = json.loads(lines[0])
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert "result" in response
    assert response["result"]["protocolVersion"] == "2024-11-05"
    assert "capabilities" in response["result"]
    assert "serverInfo" in response["result"]

    # Verify stderr contains logs (not empty)
    assert len(stderr) > 0, "stderr should contain logs"
    assert "initialize" in stderr.lower() or "server" in stderr.lower()


def test_server_stdout_not_polluted():
    """Test stdout only contains JSON-RPC messages, no log pollution."""
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    }

    server_path = Path(__file__).parent.parent.parent / "src" / "mcp_server" / "server.py"
    proc = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = proc.communicate(input=json.dumps(request) + "\n", timeout=5)

    # Every line in stdout should be valid JSON
    lines = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
    for line in lines:
        data = json.loads(line)  # Should not raise
        assert "jsonrpc" in data, "Every stdout line should be JSON-RPC message"

    # stderr should have logs
    assert len(stderr) > 0


def test_server_invalid_request():
    """Test server handles invalid JSON-RPC request."""
    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "unknown_method",
        "params": {}
    }

    server_path = Path(__file__).parent.parent.parent / "src" / "mcp_server" / "server.py"
    proc = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = proc.communicate(input=json.dumps(request) + "\n", timeout=5)

    lines = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
    assert len(lines) >= 1

    response = json.loads(lines[0])
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 3
    assert "error" in response
    assert response["error"]["code"] == -32601  # Method not found


def test_server_malformed_json():
    """Test server handles malformed JSON input."""
    server_path = Path(__file__).parent.parent.parent / "src" / "mcp_server" / "server.py"
    proc = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = proc.communicate(input="not valid json\n", timeout=5)

    lines = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
    assert len(lines) >= 1

    response = json.loads(lines[0])
    assert response["jsonrpc"] == "2.0"
    assert "error" in response
    assert response["error"]["code"] == -32700  # Parse error


def test_server_multiple_requests():
    """Test server can handle multiple sequential requests."""
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"}
            }
        },
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "initialized",
            "params": {}
        }
    ]

    input_data = "\n".join(json.dumps(req) for req in requests) + "\n"

    server_path = Path(__file__).parent.parent.parent / "src" / "mcp_server" / "server.py"
    proc = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = proc.communicate(input=input_data, timeout=5)

    lines = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
    assert len(lines) >= 2, "Should have responses for both requests"

    # First response: initialize
    resp1 = json.loads(lines[0])
    assert resp1["id"] == 10
    assert "result" in resp1

    # Second response: initialized (notification, may not have response)
    # Or error if not initialized yet
    resp2 = json.loads(lines[1])
    assert resp2["jsonrpc"] == "2.0"


def test_multimodal_assembler_with_images():
    """Test multimodal assembler can load and encode images."""
    # Create temporary image file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        images_dir = tmpdir_path / "images"
        images_dir.mkdir()

        # Create a simple 1x1 PNG image (smallest valid PNG)
        png_data = (
            b'\x89PNG\r\n\x1a\n'  # PNG signature
            b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
            b'\x08\x02\x00\x00\x00\x90wS\xde'  # IHDR chunk
            b'\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01'
            b'\r\n-\xb4'  # IDAT chunk
            b'\x00\x00\x00\x00IEND\xaeB`\x82'  # IEND chunk
        )

        image_path = images_dir / "test_image.png"
        image_path.write_bytes(png_data)

        # Create assembler
        assembler = MultimodalAssembler(images_base_dir=str(tmpdir_path))

        # Create retrieval results with image reference
        retrieval_results = [
            {
                "text": "This is a test document with an image.",
                "metadata": {
                    "images": [
                        {
                            "id": "test_img_001",
                            "path": "images/test_image.png",
                            "page": 1,
                            "text_offset": 20,
                            "text_length": 15
                        }
                    ]
                }
            }
        ]

        # Assemble content
        content = assembler.assemble(retrieval_results)

        # Verify content structure
        assert len(content) == 2, "Should have text and image content"

        # Check text content
        text_content = content[0]
        assert text_content["type"] == "text"
        assert "test document" in text_content["text"]

        # Check image content
        image_content = content[1]
        assert image_content["type"] == "image"
        assert "data" in image_content
        assert "mimeType" in image_content
        assert image_content["mimeType"] == "image/png"

        # Verify base64 encoding
        decoded = base64.b64decode(image_content["data"])
        assert decoded == png_data, "Decoded image should match original"


def test_multimodal_assembler_missing_image():
    """Test multimodal assembler handles missing image files gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        assembler = MultimodalAssembler(images_base_dir=tmpdir)

        retrieval_results = [
            {
                "text": "Document with missing image",
                "metadata": {
                    "images": [
                        {
                            "id": "missing_img",
                            "path": "images/nonexistent.png"
                        }
                    ]
                }
            }
        ]

        content = assembler.assemble(retrieval_results)

        # Should only have text content, image skipped
        assert len(content) == 1
        assert content[0]["type"] == "text"


def test_multimodal_assembler_no_images():
    """Test multimodal assembler with text-only results."""
    assembler = MultimodalAssembler()

    retrieval_results = [
        {
            "text": "Plain text document",
            "metadata": {}
        }
    ]

    content = assembler.assemble(retrieval_results)

    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Plain text document"


def test_multimodal_assembler_multiple_images():
    """Test multimodal assembler with multiple images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        images_dir = tmpdir_path / "images"
        images_dir.mkdir()

        # Create two test images
        png_data = (
            b'\x89PNG\r\n\x1a\n'
            b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
            b'\x08\x02\x00\x00\x00\x90wS\xde'
            b'\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01'
            b'\r\n-\xb4'
            b'\x00\x00\x00\x00IEND\xaeB`\x82'
        )

        (images_dir / "img1.png").write_bytes(png_data)
        (images_dir / "img2.png").write_bytes(png_data)

        assembler = MultimodalAssembler(images_base_dir=str(tmpdir_path))

        retrieval_results = [
            {
                "text": "Document with two images",
                "metadata": {
                    "images": [
                        {"id": "img1", "path": "images/img1.png"},
                        {"id": "img2", "path": "images/img2.png"}
                    ]
                }
            }
        ]

        content = assembler.assemble(retrieval_results)

        # Should have 1 text + 2 images
        assert len(content) == 3
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"
        assert content[2]["type"] == "image"


def test_multimodal_assembler_mime_types():
    """Test multimodal assembler detects correct MIME types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        images_dir = tmpdir_path / "images"
        images_dir.mkdir()

        # Create test files with different extensions
        test_data = b"fake image data"
        (images_dir / "test.png").write_bytes(test_data)
        (images_dir / "test.jpg").write_bytes(test_data)
        (images_dir / "test.gif").write_bytes(test_data)

        assembler = MultimodalAssembler(images_base_dir=str(tmpdir_path))

        # Test PNG
        result = assembler._load_image({"path": "images/test.png"})
        assert result["mimeType"] == "image/png"

        # Test JPEG
        result = assembler._load_image({"path": "images/test.jpg"})
        assert result["mimeType"] == "image/jpeg"

        # Test GIF
        result = assembler._load_image({"path": "images/test.gif"})
        assert result["mimeType"] == "image/gif"

