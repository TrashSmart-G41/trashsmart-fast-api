from fastapi import FastAPI
from .api.clustering_routes import router as clustering_router

def create_app() -> FastAPI:
    app = FastAPI(title="TrashSmart Clustering Service", version="1.0.0")
    app.include_router(clustering_router, prefix="/api")
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)