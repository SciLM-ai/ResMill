import numpy as np


class Reservoir:
    """Stacks multiple Layer objects into a single reservoir model.

    ``layers`` are listed top to bottom (each layer's base depth must meet
    the next layer's ``top_depth``); ``self.layers`` and ``self.zz`` keep
    that listed order. The concatenated property arrays follow the
    package-wide vertical convention instead: the k index increases
    upward, so global k=0 is the base of the deepest layer and k=nz-1
    the top of the shallowest.
    """

    def __init__(self, layers):
        if not isinstance(layers, list):
            layers = [layers]
        self.layers = layers
        self.n_layers = len(layers)

        # Validate lateral dimensions match
        nx0, ny0 = layers[0].nx, layers[0].ny
        x0, y0 = layers[0].x_len, layers[0].y_len
        for layer in layers:
            if layer.nx != nx0 or layer.ny != ny0:
                raise ValueError("All layers must have same nx, ny")
            if layer.x_len != x0 or layer.y_len != y0:
                raise ValueError("All layers must have same x_len, y_len")

        self.nx, self.ny = nx0, ny0
        self.x_len, self.y_len = x0, y0
        self.X, self.Y = layers[0].X, layers[0].Y
        self.nz = sum(l.nz for l in layers)
        self.z_len = sum(l.z_len for l in layers)
        self.top_depth = layers[0].top_depth
        self.zz = [zz for layer in layers for zz in layer.zz]

        # Validate z-continuity
        for i in range(len(layers) - 1):
            if not np.allclose(layers[i].zz[-1], layers[i + 1].zz[0]):
                raise ValueError(
                    f"Bottom of layer {i} does not match top of layer {i+1}"
                )

        # Concatenate property arrays along z, deepest layer first: within
        # a layer k=0 is the layer base, so this keeps k increasing upward
        # through the whole stack (and cross-sections render right side up).
        self.poro_mat = np.concatenate([l.poro_mat for l in reversed(layers)], axis=2)
        self.perm_mat = np.concatenate([l.perm_mat for l in reversed(layers)], axis=2)
        self.active = np.concatenate([l.active for l in reversed(layers)], axis=2)

    def to_grdecl(self, path, **kwargs):
        """Export the stacked model as a corner-point GRDECL file (see resmill.export.to_grdecl)."""
        from .export import to_grdecl
        return to_grdecl(self, path, **kwargs)
