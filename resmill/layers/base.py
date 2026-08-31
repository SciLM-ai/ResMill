import numpy as np


class Layer:
    """Base class for geological layers. Defines grid geometry and surfaces."""

    def __init__(self, nx, ny, nz, x_len, y_len, z_len, top_depth, dip=0.0, kzkx=0.1):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.x_len = x_len
        self.y_len = y_len
        self.z_len = z_len
        self.dx = x_len / nx
        self.dy = y_len / ny
        self.dz = z_len / nz
        self.top_depth = top_depth
        self.dip = dip
        self.kzkx = kzkx

        # Grid edges (nx+1, ny+1)
        x = np.linspace(0, x_len, nx + 1)
        y = np.linspace(0, y_len, ny + 1)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')

        # Depth surfaces with dip
        self.z1 = self.Y * np.tan(np.radians(dip)) + top_depth
        self.z2 = self.z1 + z_len
        self.zz = [self.z1, self.z2]

        # Property arrays — populated by create_geology()
        self.poro_mat = None
        self.perm_mat = None
        self.active = np.ones((nx, ny, nz), dtype=int)

    def create_geology(self):
        raise NotImplementedError("Subclasses must implement create_geology()")

    def to_grdecl(self, path, **kwargs):
        """Export this layer as a corner-point GRDECL file (see resmill.export.to_grdecl)."""
        from ..export import to_grdecl
        return to_grdecl(self, path, **kwargs)
