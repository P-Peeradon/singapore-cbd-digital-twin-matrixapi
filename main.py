from fastapi import FastAPI
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
    
@app.get("/health") 
def health_check(): 
    return { 
        "service": "FastAPI Matrix Compute Engine", 
        "status": "HEALTHY", 
        "numpy_version": np.__version__ 
    }
    
@app.post("/matrix/extrude")
def extrude_polygon(payload: TransformRequest):
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
    extruded_point = np.dot(M, p)
    return {
        "original": payload.point,
        "extruded": extruded_point[:3].tolist(),
        "matrix": M.tolist()
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