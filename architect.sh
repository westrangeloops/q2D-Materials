#!/bin/bash
# Quick launcher for Vue.js + FastAPI Layer Architect

echo "🎨 Starting Vue.js Layer Architect..."
echo ""
echo "This is the FAST version (Vue.js + FastAPI)"
echo "NOT the Streamlit version"
echo ""

cd "$(dirname "$0")/q2D_Materials/visual_tools"

# Start backend in background
echo "Starting backend on http://localhost:8000..."
python3 backend.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Start frontend server
echo "Starting frontend server on http://localhost:8080..."
echo ""
echo "✅ Backend: http://localhost:8000"
echo "✅ Frontend: http://localhost:8080/index.html"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Start HTTP server in foreground
python3 -m http.server 8080

# Cleanup on exit
kill $BACKEND_PID 2>/dev/null

