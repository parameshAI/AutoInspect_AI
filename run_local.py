import uvicorn
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    
    print("=" * 60)
    print("AutoInspect AI is starting...")
    print(f"Server will be available at: http://{host}:{port}")
    print("=" * 60)
    
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
