#!/usr/bin/env python3
"""
API Key setup utility for the RAG system.
"""

import os
import sys
from pathlib import Path

def setup_api_key():
    """Interactive setup for Anthropic API key."""
    print("🔑 RAG System API Key Setup")
    print("=" * 40)
    
    env_file = Path("../.env")
    
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Creating .env file...")
        with open(env_file, "w") as f:
            f.write("# Copy this file to .env and add your actual API key\n")
            f.write("ANTHROPIC_API_KEY=your-anthropic-api-key-here\n")
        print(f"✅ Created {env_file}")
    
    print("\nTo fix the 'query failed' issue:")
    print("1. Get your Anthropic API key from: https://console.anthropic.com/")
    print("2. Edit the .env file and replace 'your-anthropic-api-key-here' with your actual API key")
    print("3. Restart the application")
    
    print(f"\n📁 Edit this file: {env_file.absolute()}")
    print("\nExample .env content:")
    print("ANTHROPIC_API_KEY=sk-ant-api03-...")

if __name__ == "__main__":
    setup_api_key()