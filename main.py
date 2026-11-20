from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

app = FastAPI( 
    title="Singapore CBD Digital Twin Matrix Compute Engine", 
    version="1.0.0", 
    description="Vectorized 4x4 matrix, 4D temporal, and spatial calculations" 
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TransformRequest(BaseModel):
    point: list[float]  # [x, y, z]
    translation: list[float]  # [tx, ty, tz]
    rotation_y: float  # Angle in radians

class ExtrudePolygonRequest(BaseModel):
    polygon: list[list[list[float]]]  # [[ [x, z], ... ], ...] (outer ring, then holes)
    height: float
    storey: int
    
@app.get("/health") 
def health_check(): 
    return { 
        "service": "FastAPI Matrix Compute Engine", 
        "status": "HEALTHY", 
        "numpy_version": np.__version__ 
    }
    
@app.post("/matrix/extrude")
def extrude_polygon(payload: ExtrudePolygonRequest):
    if not payload.polygon or any(len(ring) < 3 for ring in payload.polygon):
        raise HTTPException(status_code=422, detail="Each polygon ring must contain at least three points.")
    if any(len(point) != 2 for ring in payload.polygon for point in ring):
        raise HTTPException(status_code=422, detail="Polygon points must be [x, z] coordinate pairs.")
    coordinates = np.array([point for ring in payload.polygon for point in ring])
    if not np.isfinite(coordinates).all():
        raise HTTPException(status_code=422, detail="Polygon coordinates must be finite.")
    if not np.isfinite(payload.height) or payload.height <= 0:
        raise HTTPException(status_code=422, detail="Building height must be a positive finite number.")
    if payload.storey < 1:
        raise HTTPException(status_code=422, detail="Building storeys must be a positive integer.")

    levels = np.linspace(0.0, payload.height, payload.storey + 1)
    Height = np.identity(4)
    Height[1, 3] = payload.height
    extruded_polygon = []
    for ring in payload.polygon:
        points = np.array([[x, 0.0, z, 1.0] for x, z in ring])
        extruded_polygon.append((points @ Height.T)[:, :3].tolist())

    return {
        "original": payload.polygon,
        "extruded": extruded_polygon,
        "matrix": Height.tolist(),
        "levels": levels.tolist()
    }

@app.post("/matrix/transform") 
def transform_point(payload: TransformRequest): 
    # Homogeneous 4-vector representation 
    p = np.array([payload.point[0], payload.point[1], payload.point[2], 1.0]) 
    # 4x4 Translation Matrix 
    T = np.identity(4) 
    T[0, 3] = payload.translation[0] 
    T[1, 3] = payload.translation[1] 
    T[2, 3] = payload.translation[2] 
    # 4x4 Rotation Matrix around Y-axis 
    theta = payload.rotation_y 
    R = np.identity(4) 
    R[0, 0] = np.cos(theta) 
    R[0, 2] = np.sin(theta) 
    R[2, 0] = -np.sin(theta) 
    R[2, 2] = np.cos(theta) 
    # Composite Affine Matrix: M = T * R 
    M = np.dot(T, R) 
    transformed = np.dot(M, p) 
    return { 
        "original": payload.point, 
        "transformed": transformed[:3].tolist(), 
        "matrix": M.tolist() 
    }