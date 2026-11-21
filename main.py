from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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

class ExtrusionMatrixRequest(BaseModel): 
    # Polygon represented as a 3 x n matrix (rows: [x], [y], [z]) 
    polygon: list[list[float]] = Field(..., description="3 x n matrix representing polygon coordinates") 
    height: float = Field(..., gt=0.0, description="Total extrusion height in meters") 
    storeys: int = Field(default=1, ge=1, description="Number of storeys / vertical segments") 

class ExtrusionMatrixResponse(BaseModel): 
    original: list[list[float]] 
    elevation_levels: list[float] 
    prism: list[list[float]] 
    # 3 x (n \* (storeys + 1)) matrix 
    vertex_count: int
    
@app.get("/health") 
def health_check(): 
    return { 
        "service": "FastAPI Matrix Compute Engine", 
        "status": "HEALTHY", 
        "numpy_version": np.__version__ 
    }
    
@app.post("/matrix/extrude", 
    response_model=ExtrusionMatrixResponse, 
    status_code=status.HTTP_200_OK, 
    summary="Compute 3D Prism Cartesian Lifting via Matrix Operations" 
) 
def compute_extrusion_matrix(payload: ExtrusionMatrixRequest): 
    # 1. Convert input polygon to NumPy array (shape: 3 x n) 
    poly = np.asarray(payload.polygon, dtype=np.float64) 
    if poly.shape[0] != 3: 
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail=f"Input polygon matrix must be 3 x n, received shape {poly.shape}" 
        ) 
    n = poly.shape[1] 
    k = payload.storeys 
    
    # 2. Construct discrete elevation levels: {0, h, 2h, ..., H} of length k + 1 
    elevations = np.linspace(0.0, payload.height, k + 1) 
    
    # 3. Perform Cartesian product lifting across all (k + 1) elevation tiers 
    prism_tiers = [] 
    for y_val in elevations: 
        tier = poly.copy() 
        tier[1, :] = y_val 
        # Set vertical Y elevation row 
        prism_tiers.append(tier) 
    
    # 4. Concatenate horizontally along columns -> shape: 3 x (n \* (k + 1)) 
    prism_matrix = np.hstack(prism_tiers) 
    return ExtrusionMatrixResponse(
        original=poly.tolist(), 
        elevation_levels=elevations.tolist(), 
        prism=prism_matrix.tolist(), 
        vertex_count=prism_matrix.shape[1] 
    )

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