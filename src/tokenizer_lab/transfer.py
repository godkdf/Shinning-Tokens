"""Small, explicit embedding transfer baselines; no implicit model downloads."""
from __future__ import annotations
import numpy as np

def subtoken_average(embeddings,token_sequences):
    matrix=np.asarray(embeddings)
    if matrix.ndim!=2:raise ValueError("Embedding matrix must have shape [vocabulary, width]")
    results=[]
    for ids in token_sequences:
        if not ids or any(not isinstance(i,(int,np.integer)) or i<0 or i>=len(matrix) for i in ids):
            raise ValueError("Every new token needs nonempty valid parent IDs")
        results.append(matrix[ids].astype(np.float32).mean(axis=0))
    return np.stack(results) if results else np.empty((0,matrix.shape[1]),dtype=np.float32)

def ridge_mapping(source,target,*,regularization=.01):
    x=np.asarray(source,dtype=np.float64);y=np.asarray(target,dtype=np.float64)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(x)<2:raise ValueError("Matched 2D source and target anchors are required")
    if regularization<=0:raise ValueError("Regularization must be positive")
    xm=x.mean(axis=0);ym=y.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-8]=1
    z=(x-xm)/scale;centered=y-ym
    if z.shape[1]<=len(z):
        weight=np.linalg.solve(z.T@z+regularization*np.eye(z.shape[1]),z.T@centered)
    else:
        weight=z.T@np.linalg.solve(z@z.T+regularization*np.eye(len(z)),centered)
    return {"source_mean":xm.astype(np.float32),"source_scale":scale.astype(np.float32),"weight":weight.astype(np.float32),"target_mean":ym.astype(np.float32)}

def apply_mapping(source,mapping):
    return ((np.asarray(source,dtype=np.float32)-mapping["source_mean"])/mapping["source_scale"])@mapping["weight"]+mapping["target_mean"]

def mapping_metrics(predicted,target):
    p=np.asarray(predicted,dtype=np.float64);t=np.asarray(target,dtype=np.float64)
    if p.shape!=t.shape or p.ndim!=2:raise ValueError("Prediction and target shapes must match")
    denom=np.linalg.norm(p,axis=1)*np.linalg.norm(t,axis=1)
    valid=denom>0;cos=np.sum(p*t,axis=1)[valid]/denom[valid]
    return {"anchors":len(p),"coordinate_mse":float(np.mean((p-t)**2)),"mean_cosine":float(cos.mean()) if len(cos) else None,
            "median_cosine":float(np.median(cos)) if len(cos) else None,"scope":"Vector reconstruction only; not evidence of generation quality."}
