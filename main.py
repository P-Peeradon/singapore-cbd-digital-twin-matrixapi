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

class BoundingSphere(BaseModel): 
    id: str 
    center: list[float] # [x, y, z] 
    radius: float 

class FrustumCullingRequest(BaseModel): 
    mvp_matrix: list[list[float]] # 4x4 MVP Matrix 
    spheres: list[BoundingSphere] 

class FrustumCullingResponse(BaseModel): 
    total_objects: int 
    visible_objects: list[str] 
    culled_objects: list[str]
    
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

@app.post("/matrix/frustum-culling", response_model=FrustumCullingResponse) 
def perform_frustum_culling(payload: FrustumCullingRequest): 
    mvp = np.array(payload.mvp_matrix, dtype=np.float64) 
    if mvp.shape != (4, 4): 
        raise ValueError("MVP matrix must be 4x4") 
    
    # Extract 6 frustum planes (Left, Right, Bottom, Top, Near, Far) from MVP matrix 
    planes = np.zeros((6, 4)) 
    planes[0] = mvp[3, :] + mvp[0, :] # Left 
    planes[1] = mvp[3, :] - mvp[0, :] # Right 
    planes[2] = mvp[3, :] + mvp[1, :] # Bottom 
    planes[3] = mvp[3, :] - mvp[1, :] # Top 
    planes[4] = mvp[3, :] + mvp[2, :] # Near 
    planes[5] = mvp[3, :] - mvp[2, :] # Far 
    
    # Normalize plane equations (Ax + By + Cz + D = 0) 
    for i in range(6): 
        norm = np.linalg.norm(planes[i, :3]) 
        if norm > 0: 
            planes[i] /= norm 
        
        visible = [] 
        culled = [] 
        for sphere in payload.spheres: 
            center = np.array([sphere.center[0], sphere.center[1], sphere.center[2], 1.0]) 
            r = sphere.radius 
            is_visible = True 
            
            for i in range(6): 
                # Distance from sphere center to plane 
                dist = np.dot(planes[i], center) 
                if dist < -r: 
                    is_visible = False 
                    break 
                
            if is_visible: visible.append(sphere.id) 
            else: culled.append(sphere.id) 
                
            return FrustumCullingResponse(
                total_objects=len(payload.spheres), 
                visible_objects=visible, 
                culled_objects=culled 
            )