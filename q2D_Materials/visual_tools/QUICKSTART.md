# Quick Start - Vue.js Layer Architect

## 🚀 Simplest Way (One Command!)

```bash
cd q2D_Materials/visual_tools
python3 backend.py
```

Then open **`http://localhost:8000`** in your browser!

The backend now serves the HTML file directly, so you don't need a separate HTTP server.

## Using Nix

```bash
python3 q2D_Materials/visual_tools/backend.py
```

Then open **`http://localhost:8000`** in your browser.

## ⚠️ Important Notes

- **This is NOT Streamlit** - The Vue.js version runs on ports 8000 (backend) and 8080 (frontend)
- **Streamlit version** runs on port 8501: `streamlit run layer_builder.py`
- The Vue.js version is **much faster** and has **better performance**

## 🔧 Troubleshooting

If you see CORS errors, make sure:
1. Backend is running on port 8000
2. Frontend is being served (not just opened as file://)
3. The API URL in `index.html` matches your backend URL

