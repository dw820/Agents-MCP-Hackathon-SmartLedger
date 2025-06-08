#!/usr/bin/env python3
"""
SmartLedger Service Runner
Manages both the Gradio web interface and MCP server
"""

import subprocess
import threading
import time
import sys
import signal
import os
from pathlib import Path

def run_gradio_app():
    """Run the Gradio web interface"""
    print("🚀 Starting Gradio web interface...")
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("📱 Gradio interface stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Gradio app failed: {e}")

def run_mcp_server():
    """Run the MCP server"""
    print("🔧 Starting MCP server...")
    try:
        subprocess.run([sys.executable, "mcp_server.py"], check=True)
    except KeyboardInterrupt:
        print("🔧 MCP server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ MCP server failed: {e}")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['gradio', 'modal', 'llama-index', 'mcp']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("📦 Install with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main service runner"""
    print("🎯 SmartLedger Service Manager")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not set in environment")
        print("   Copy .env.example to .env and add your API key")
    
    if not os.getenv('MODAL_TOKEN'):
        print("⚠️  Warning: MODAL_TOKEN not set in environment")
        print("   Run 'modal token new' to setup Modal authentication")
    
    mode = input("\nSelect mode:\n1. Web interface only\n2. MCP server only\n3. Both services\nChoice (1-3): ").strip()
    
    try:
        if mode == "1":
            print("\n🚀 Starting Gradio web interface...")
            run_gradio_app()
            
        elif mode == "2":
            print("\n🔧 Starting MCP server...")
            run_mcp_server()
            
        elif mode == "3":
            print("\n🚀 Starting both services...")
            
            # Start Gradio in a thread
            gradio_thread = threading.Thread(target=run_gradio_app, daemon=True)
            gradio_thread.start()
            
            # Wait a moment for Gradio to start
            time.sleep(3)
            
            # Start MCP server in main thread
            run_mcp_server()
            
        else:
            print("❌ Invalid choice. Exiting.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services...")
        
    print("✅ SmartLedger services stopped")

if __name__ == "__main__":
    main()