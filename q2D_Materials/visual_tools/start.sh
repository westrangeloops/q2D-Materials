#!/bin/bash
# Start script for Vue.js + FastAPI Layer Architect

echo "🚀 Starting Layer Architect (Vue.js + FastAPI)"
echo ""
echo "Backend will run on: http://localhost:8000"
echo "Open index.html in your browser or serve it with:"
echo "  python3 -m http.server 8080"
echo ""
echo "Then open: http://localhost:8080/q2D_Materials/visual_tools/index.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
python3 backend.py

