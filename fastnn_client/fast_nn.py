from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import grpc
import numpy as np

from . import fastnn_pb2
from . import fastnn_pb2_grpc

from scipy.spatial import KDTree

# (tbp.monty) slapd@LUCID:~/tbp$ code .
# (tbp.monty) slapd@LUCID:~/tbp$ cat /etc/resolv.conf | grep nameserver
# nameserver 10.255.255.254

DEFAULT_ENDPOINT = "172.31.32.1:50051"  # whatever you’re using from WSL


def _as_points_f32(points: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D numpy array of shape (N,2) or (N,3); got ndim={arr.ndim}")

    if arr.shape[1] not in (2, 3):
        raise ValueError(f"{name} must have 2 or 3 columns; got shape={arr.shape}")

    return np.ascontiguousarray(arr)


@dataclass(frozen=True)
class _Split:
    x: list[float]
    y: list[float]
    z: Optional[list[float]]  # None means "omit field" (2D)


def _split_xyz(points_f32: np.ndarray) -> _Split:
    x = points_f32[:, 0].tolist()
    y = points_f32[:, 1].tolist()

    if points_f32.shape[1] == 3:
        z = points_f32[:, 2].tolist()
        return _Split(x=x, y=y, z=z)

    return _Split(x=x, y=y, z=None)


class FastNn:
    """
    Mimics the minimal surface of scipy.spatial.KDTree:

        nn = FastNn(source_points)
        dist, idx = nn.query(query_points, k=10)

    Notes:
      - Supports 2D (N,2)/(Q,2) by omitting z in requests (server treats as z=0).
      - Returns SciPy-like shapes:
          k == 1  -> (Q,), (Q,)
          k > 1   -> (Q,k), (Q,k)
    """

    def __init__(self, source_points: np.ndarray):
        self._source_points = _as_points_f32(source_points, name="source_points")
        self._channel = None
        self._stub = None
        self._id = None
        self._closed = False

        self._local_kdtree = KDTree(source_points)  # for small queries, to avoid gRPC overhead

        self._connect_and_create()

    def _connect_and_create(self) -> None:
        if self._closed:
            raise RuntimeError("FastNn is closed")

        self._channel = grpc.insecure_channel(DEFAULT_ENDPOINT)
        self._stub = fastnn_pb2_grpc.FastNnServiceStub(self._channel)

        split = _split_xyz(self._source_points)
        create_kwargs = {"x": split.x, "y": split.y}
        if split.z is not None:
            create_kwargs["z"] = split.z

        resp = self._stub.Create(fastnn_pb2.CreateRequest(**create_kwargs))
        self._id = resp.id
        print(f"FastNn id: {self._id}")

    def query(self, search_locations: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        if self._closed:
            raise RuntimeError("FastNn is closed")
        if self._id is None:
            # e.g., after unpickling if caller used it before _connect_and_create ran
            self._connect_and_create()
        if k <= 0:
            raise ValueError(f"k must be >= 1; got {k}")

        
            

        # DECISION # DECISION # DECISION # DECISION # DECISION 

        local_limit = 600  # empirically, gRPC overhead dominates for small queries; this is a heuristic threshold  

        if(len(search_locations) <= local_limit):
            return self._local_kdtree.query(search_locations, k=k)






        qpts = _as_points_f32(search_locations, name="search_locations")
        q = qpts.shape[0]
        split = _split_xyz(qpts)

        query_kwargs = {"id": self._id, "x": split.x, "y": split.y, "k": int(k)}
        if split.z is not None:
            query_kwargs["z"] = split.z

        resp = self._stub.Query(fastnn_pb2.QueryRequest(**query_kwargs))

        expected = q * k
        idx = np.asarray(resp.indices, dtype=np.int32).reshape(q, k)
        dist = np.asarray(resp.distances, dtype=np.float32).reshape(q, k)

        if k == 1:
            return dist[:, 0], idx[:, 0]

        return dist, idx

    def close(self) -> None:
        if self._closed:
            return

        print(f"FastNn id: {self._id} closing...")

        try:
            if self._stub is not None and self._id is not None:
                self._stub.Destroy(fastnn_pb2.DestroyRequest(id=self._id))
        finally:
            if self._channel is not None:
                self._channel.close()
            self._closed = True
            

    # ---- Pickle support (KDTree-like) ----

    def __getstate__(self):
        # Only store the deterministic data needed to recreate the engine.
        # Do NOT store gRPC channel/stub/engine id.
        return {
            "source_points": self._source_points,
            "closed": self._closed,
        }

    def __setstate__(self, state):
        self._source_points = _as_points_f32(state["source_points"], name="source_points")
        self._closed = bool(state.get("closed", False))
        self._channel = None
        self._stub = None
        self._id = None

        if not self._closed:
            self._connect_and_create()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass